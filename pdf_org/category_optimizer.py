from __future__ import annotations

from .llm_classifier import BaseLLMClient
from .models import CategoryOptimizationResult, DocumentClassification


def optimize_categories(
    classifications: list[DocumentClassification],
    llm_client: BaseLLMClient,
) -> CategoryOptimizationResult:
    """Run second-pass optimization on raw categories inferred per document."""
    raw_categories = [c.suggested_category for c in classifications if c.suggested_category]
    return llm_client.optimize_categories(raw_categories)
