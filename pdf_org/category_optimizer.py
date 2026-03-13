from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict

from .llm_classifier import BaseLLMClient
from .models import CategoryOptimizationResult, DocumentClassification
from .utils import normalize_category_name

logger = logging.getLogger(__name__)


def optimize_categories(
    classifications: list[DocumentClassification],
    llm_client: BaseLLMClient,
    min_docs_per_category: int = 2,
    max_docs_per_category: int = 25,
) -> CategoryOptimizationResult:
    """Run category optimization and rebalance category sizes between configured thresholds."""
    if not classifications:
        return CategoryOptimizationResult(
            category_mapping={},
            normalized_categories=[],
            short_reason="No classifications provided.",
        )

    min_size = max(1, int(min_docs_per_category))
    max_size = max(min_size, int(max_docs_per_category))

    raw_categories = [c.suggested_category for c in classifications if c.suggested_category]
    base_optimization = llm_client.optimize_categories(raw_categories)
    global_mapping = base_optimization.category_mapping or {}

    docs: list[dict[str, str]] = []
    for idx, classification in enumerate(classifications):
        raw = normalize_category_name(classification.suggested_category or "uncategorized")
        category = normalize_category_name(global_mapping.get(raw, raw))
        docs.append(
            {
                "id": str(idx),
                "raw_category": raw,
                "category": category,
                "topic": classification.topic,
                "doc_type": classification.document_type,
                "title": classification.title,
            }
        )

    result_docs, operations = _rebalance_categories(docs, llm_client=llm_client, min_size=min_size, max_size=max_size)

    category_mapping: dict[str, str] = {}
    document_mapping: dict[str, str] = {}
    category_sizes: dict[str, int] = {}

    counts = Counter(doc["category"] for doc in result_docs)
    for doc in result_docs:
        category_mapping[doc["raw_category"]] = doc["category"]
        document_mapping[doc["id"]] = doc["category"]
    for cat, size in counts.items():
        category_sizes[cat] = size

    short_reason = (
        f"Optimized categories with size constraints [{min_size}, {max_size}] "
        f"in {len(operations)} merge/split operation(s)."
    )

    return CategoryOptimizationResult(
        category_mapping=category_mapping,
        document_category_mapping=document_mapping,
        normalized_categories=sorted(category_sizes),
        category_sizes=dict(sorted(category_sizes.items())),
        operations_log=operations,
        short_reason=short_reason,
        raw_response=base_optimization.raw_response,
        from_cache=base_optimization.from_cache,
    )


def _rebalance_categories(
    docs: list[dict[str, str]],
    llm_client: BaseLLMClient,
    min_size: int,
    max_size: int,
    max_iterations: int = 25,
) -> tuple[list[dict[str, str]], list[str]]:
    operations: list[str] = []
    for _ in range(max_iterations):
        by_category = _group_by_category(docs)
        oversized = [(cat, items) for cat, items in by_category.items() if len(items) > max_size]
        undersized = [(cat, items) for cat, items in by_category.items() if len(items) < min_size]

        if not oversized and (not undersized or len(by_category) <= 1):
            break

        if oversized:
            cat, items = max(oversized, key=lambda pair: len(pair[1]))
            split_groups = _split_category_items(items, max_size=max_size)
            if len(split_groups) <= 1:
                oversized = []
            else:
                names = _generate_subcategory_names(cat, split_groups, llm_client)
                for group, new_name in zip(split_groups, names, strict=False):
                    for doc in group:
                        doc["category"] = new_name
                log_line = f"SPLIT: {cat} ({len(items)} docs) -> " + ", ".join(
                    f"{name} ({len(group)})" for name, group in zip(names, split_groups, strict=False)
                )
                logger.info(log_line)
                operations.append(log_line)
                continue

        if undersized and len(by_category) > 1:
            cat, items = min(undersized, key=lambda pair: len(pair[1]))
            target = _find_most_similar_category(cat, by_category, max_combined_size=max_size)
            if not target:
                log_line = (
                    f"SKIP-MERGE: {cat} ({len(items)}) has no compatible target "
                    f"without exceeding max_docs_per_category={max_size}."
                )
                logger.info(log_line)
                operations.append(log_line)
                break
            merged_items = by_category[target] + items
            merged_name = _generate_merged_category_name(target, cat, merged_items, llm_client)
            for doc in merged_items:
                doc["category"] = merged_name
            log_line = f"MERGE: {cat} ({len(items)}) + {target} ({len(by_category[target])}) -> {merged_name} ({len(merged_items)})"
            logger.info(log_line)
            operations.append(log_line)
            continue

        break

    return docs, operations


def _group_by_category(docs: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for doc in docs:
        groups[doc["category"]].append(doc)
    return dict(groups)


def _split_category_items(items: list[dict[str, str]], max_size: int) -> list[list[dict[str, str]]]:
    if len(items) <= max_size:
        return [items]

    k = max(2, math.ceil(len(items) / max_size))
    sorted_items = sorted(items, key=lambda d: (d.get("topic", ""), d.get("doc_type", ""), d.get("title", "")))
    buckets = [[] for _ in range(k)]
    for idx, item in enumerate(sorted_items):
        buckets[idx % k].append(item)
    return [bucket for bucket in buckets if bucket]


def _generate_subcategory_names(base_category: str, groups: list[list[dict[str, str]]], llm_client: BaseLLMClient) -> list[str]:
    names: list[str] = []
    for idx, group in enumerate(groups, start=1):
        candidates = [
            f"{base_category}_{_most_common_token(group, 'topic')}",
            f"{base_category}_{_most_common_token(group, 'doc_type')}",
            f"{base_category}_{idx}",
        ]
        snippets = [f"{item.get('title','')} | {item.get('topic','')} | {item.get('doc_type','')}" for item in group[:8]]
        proposed = llm_client.suggest_category_name(candidates, snippets)
        name = normalize_category_name(proposed)
        if not name or name in names:
            name = normalize_category_name(f"{base_category}_{idx}")
        names.append(name)
    return names


def _generate_merged_category_name(target: str, source: str, merged_items: list[dict[str, str]], llm_client: BaseLLMClient) -> str:
    candidates = [target, source, f"{target}_{source}"]
    snippets = [f"{item.get('title','')} | {item.get('topic','')}" for item in merged_items[:10]]
    return normalize_category_name(llm_client.suggest_category_name(candidates, snippets))


def _find_most_similar_category(
    source: str,
    groups: dict[str, list[dict[str, str]]],
    max_combined_size: int,
) -> str | None:
    source_vec = _category_vector(groups.get(source, []), source)
    source_size = len(groups.get(source, []))
    best_name = None
    best_score = float("-inf")
    for name, items in groups.items():
        if name == source:
            continue
        score = _cosine_similarity(source_vec, _category_vector(items, name))
        combined = source_size + len(items)
        if combined <= max_combined_size and score > best_score:
            best_score = score
            best_name = name
    return best_name


def _category_vector(items: list[dict[str, str]], category_name: str) -> list[float]:
    vector = [0.0] * 48
    for token in _tokenize(category_name):
        vector[hash(token) % len(vector)] += 1.8
    for item in items:
        for key in ("topic", "doc_type", "title"):
            for token in _tokenize(item.get(key, "")):
                vector[hash(token) % len(vector)] += 1.0
    return vector


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return -1.0
    return dot / (mag1 * mag2)


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in (text or ""))
    return [part for part in cleaned.split() if len(part) > 1]


def _most_common_token(items: list[dict[str, str]], field_name: str) -> str:
    tokens = []
    for item in items:
        tokens.extend(_tokenize(item.get(field_name, "")))
    if not tokens:
        return "group"
    return Counter(tokens).most_common(1)[0][0]
