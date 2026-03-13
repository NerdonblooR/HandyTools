from __future__ import annotations

import json
import logging
import os

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

from .cache import JsonCache
from .models import PdfRecord
from .utils import pick_reasonable_label, slugify_folder_name

logger = logging.getLogger(__name__)


class ClusterNamer:
    """Use an LLM sparingly to produce short human-readable cluster folder names."""

    def __init__(
        self,
        model: str,
        cache: JsonCache,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: int = 45,
    ):
        self.model = model
        self.cache = cache
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_s = timeout_s

    def name_cluster(self, cluster_key: str, records: list[PdfRecord]) -> tuple[str, str]:
        """Return (folder_name, short_reason) with cache and deterministic fallback."""
        cached = self.cache.get_cluster_name(cluster_key)
        if cached:
            return str(cached.get("folder_name") or "misc"), "from_cache"

        fallback = slugify_folder_name(pick_reasonable_label("\n".join(r.semantic_text for r in records)))
        if not self.api_key or requests is None:
            self.cache.set_cluster_name(cluster_key, {"folder_name": fallback})
            return fallback, "fallback_without_llm"

        prompt = self._prompt(records)
        response = self._chat(prompt)
        folder = self._parse_name(response) or fallback
        folder = slugify_folder_name(folder)
        self.cache.set_cluster_name(cluster_key, {"folder_name": folder, "raw": response})
        return folder, "llm_named"

    def _chat(self, prompt: str) -> str | None:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return concise JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to name cluster with LLM: %s", exc)
            return None

    @staticmethod
    def _prompt(records: list[PdfRecord]) -> str:
        sample = []
        for r in records[:8]:
            sample.append(
                {
                    "title": r.detected_title,
                    "metadata_title": r.metadata_title,
                    "snippet": r.content_snippet[:400],
                }
            )

        return (
            "Generate a short folder label (2-4 words) for a cluster of related PDFs. "
            "Avoid generic words like document/files/misc. Return JSON: {\"folder_name\":\"...\"}.\n"
            f"Cluster sample:\n{json.dumps(sample, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_name(raw: str | None) -> str | None:
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("folder_name"):
                return str(parsed["folder_name"]).strip()
        except json.JSONDecodeError:
            return None
        return None
