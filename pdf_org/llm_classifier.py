from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

from .cache import JsonCache
from .models import CategoryOptimizationResult, DocumentClassification, PdfRecord
from .utils import extract_json_object, normalize_category_name, stable_hash_for_categories

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Pluggable LLM interface for document classification and category optimization."""

    @abstractmethod
    def classify_document(self, record: PdfRecord) -> DocumentClassification:
        raise NotImplementedError

    @abstractmethod
    def optimize_categories(self, raw_categories: list[str]) -> CategoryOptimizationResult:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLMClient):
    """OpenAI-compatible chat-completions client with robust JSON parsing and caching."""

    def __init__(
        self,
        model: str,
        cache: JsonCache,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: int = 60,
    ):
        self.model = model
        self.cache = cache
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_s = timeout_s

    def classify_document(self, record: PdfRecord) -> DocumentClassification:
        cached = self.cache.get_document(record.sha256)
        if cached:
            return self._build_classification(cached, from_cache=True)

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set; falling back to deterministic local classification")
            fallback = self._fallback_classification(record, "No API key configured")
            self.cache.set_document(record.sha256, self._classification_to_payload(fallback))
            return fallback

        prompt = self._classification_prompt(record)
        raw = self._chat_json(prompt)
        parsed = extract_json_object(raw or "")

        if not parsed:
            fallback = self._fallback_classification(record, "Malformed LLM JSON response")
            fallback.raw_response = raw
            self.cache.set_document(record.sha256, self._classification_to_payload(fallback))
            return fallback

        result = self._build_classification(parsed, raw_response=raw)
        self.cache.set_document(record.sha256, self._classification_to_payload(result))
        return result

    def optimize_categories(self, raw_categories: list[str]) -> CategoryOptimizationResult:
        cleaned = [normalize_category_name(c) for c in raw_categories if c.strip()]
        if not cleaned:
            return CategoryOptimizationResult(
                category_mapping={},
                normalized_categories=[],
                short_reason="No categories provided.",
            )

        categories_hash = stable_hash_for_categories(cleaned)
        cached = self.cache.get_optimization(categories_hash)
        if cached:
            return self._build_optimization(cached, from_cache=True)

        if not self.api_key:
            mapping = {c: c for c in sorted(set(cleaned))}
            fallback = CategoryOptimizationResult(
                category_mapping=mapping,
                normalized_categories=sorted(set(mapping.values())),
                short_reason="No API key configured; identity mapping used.",
            )
            self.cache.set_optimization(categories_hash, self._optimization_to_payload(fallback))
            return fallback

        prompt = self._optimization_prompt(sorted(set(cleaned)))
        raw = self._chat_json(prompt)
        parsed = extract_json_object(raw or "")

        if not parsed:
            mapping = {c: c for c in sorted(set(cleaned))}
            fallback = CategoryOptimizationResult(
                category_mapping=mapping,
                normalized_categories=sorted(set(mapping.values())),
                short_reason="Malformed LLM JSON response; identity mapping used.",
                raw_response=raw,
            )
            self.cache.set_optimization(categories_hash, self._optimization_to_payload(fallback))
            return fallback

        result = self._build_optimization(parsed, raw_response=raw)
        if not result.category_mapping:
            result.category_mapping = {c: c for c in sorted(set(cleaned))}
            result.normalized_categories = sorted(set(result.category_mapping.values()))
        self.cache.set_optimization(categories_hash, self._optimization_to_payload(result))
        return result

    def _chat_json(self, prompt: str) -> str | None:
        if requests is None:
            logger.error("requests package is not installed; cannot call LLM API")
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a precise document organization assistant. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM request failed: %s", exc)
            return None

    def _classification_prompt(self, record: PdfRecord) -> str:
        preview = record.text_preview[:4000]
        title_hint = record.metadata_title or ""
        return (
            "Classify this PDF document and infer a likely true title. "
            "Return JSON with keys: title, document_type, topic, suggested_category, confidence, short_reason. "
            "document_type must be one of: research_paper, book, notes, slides, report, manual, thesis, other.\n\n"
            f"Metadata title: {title_hint}\n"
            f"Relative path: {record.relative_path}\n"
            f"Page count: {record.page_count}\n"
            f"Text preview:\n{preview}"
        )

    def _optimization_prompt(self, categories: list[str]) -> str:
        return (
            "You are optimizing folder taxonomy for a PDF library. "
            "Merge near-duplicates, reduce fragmentation, and normalize names with snake_case. "
            "Return JSON object with keys: category_mapping (object raw->optimized), normalized_categories (array), short_reason.\n\n"
            f"Raw categories:\n{json.dumps(categories, ensure_ascii=False)}"
        )

    def _fallback_classification(self, record: PdfRecord, reason: str) -> DocumentClassification:
        title = (record.metadata_title or Path(record.file_path).stem).strip() or "untitled"
        return DocumentClassification(
            title=title,
            document_type="other",
            topic="unknown",
            suggested_category="uncategorized",
            confidence=0.1,
            short_reason=reason,
            raw_response=None,
        )

    @staticmethod
    def _build_classification(payload: dict[str, Any], raw_response: str | None = None, from_cache: bool = False) -> DocumentClassification:
        confidence_raw = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except Exception:  # noqa: BLE001
            confidence = 0.0

        return DocumentClassification(
            title=str(payload.get("title") or "untitled").strip() or "untitled",
            document_type=str(payload.get("document_type") or "other").strip() or "other",
            topic=str(payload.get("topic") or "unknown").strip() or "unknown",
            suggested_category=normalize_category_name(str(payload.get("suggested_category") or "uncategorized")),
            confidence=max(0.0, min(1.0, confidence)),
            short_reason=str(payload.get("short_reason") or "").strip(),
            raw_response=raw_response if raw_response is not None else payload.get("raw_response"),
            from_cache=from_cache,
        )

    @staticmethod
    def _build_optimization(payload: dict[str, Any], raw_response: str | None = None, from_cache: bool = False) -> CategoryOptimizationResult:
        mapping_raw = payload.get("category_mapping") or {}
        mapping: dict[str, str] = {}
        if isinstance(mapping_raw, dict):
            for raw_cat, opt_cat in mapping_raw.items():
                raw_norm = normalize_category_name(str(raw_cat))
                opt_norm = normalize_category_name(str(opt_cat))
                mapping[raw_norm] = opt_norm

        normalized_raw = payload.get("normalized_categories") or []
        normalized = []
        if isinstance(normalized_raw, list):
            normalized = sorted({normalize_category_name(str(c)) for c in normalized_raw})
        if not normalized and mapping:
            normalized = sorted(set(mapping.values()))

        return CategoryOptimizationResult(
            category_mapping=mapping,
            normalized_categories=normalized,
            short_reason=str(payload.get("short_reason") or "").strip(),
            raw_response=raw_response if raw_response is not None else payload.get("raw_response"),
            from_cache=from_cache,
        )

    @staticmethod
    def _classification_to_payload(result: DocumentClassification) -> dict[str, Any]:
        return {
            "title": result.title,
            "document_type": result.document_type,
            "topic": result.topic,
            "suggested_category": result.suggested_category,
            "confidence": result.confidence,
            "short_reason": result.short_reason,
            "raw_response": result.raw_response,
        }

    @staticmethod
    def _optimization_to_payload(result: CategoryOptimizationResult) -> dict[str, Any]:
        return {
            "category_mapping": result.category_mapping,
            "normalized_categories": result.normalized_categories,
            "short_reason": result.short_reason,
            "raw_response": result.raw_response,
        }

