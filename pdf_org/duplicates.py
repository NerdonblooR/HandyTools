from __future__ import annotations

import math
import re
from collections import Counter

from .models import DuplicateCandidate, PdfRecord


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)]


def _vec(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def find_duplicate_candidates(records: list[PdfRecord], near_threshold: float = 0.94) -> list[DuplicateCandidate]:
    """Return exact and likely duplicate pairs without deleting files."""
    if len(records) < 2:
        return []

    out: list[DuplicateCandidate] = []

    by_sha: dict[str, list[PdfRecord]] = {}
    for r in records:
        by_sha.setdefault(r.sha256, []).append(r)

    for sha, members in by_sha.items():
        if len(members) < 2:
            continue
        for i in range(len(members) - 1):
            out.append(
                DuplicateCandidate(
                    left_sha256=members[i].sha256,
                    right_sha256=members[i + 1].sha256,
                    reason="exact_sha256_match",
                    similarity=1.0,
                )
            )

    vectors = [_vec(r.semantic_text or r.detected_title) for r in records]
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i].sha256 == records[j].sha256:
                continue
            score = _cosine(vectors[i], vectors[j])
            if score >= near_threshold:
                out.append(
                    DuplicateCandidate(
                        left_sha256=records[i].sha256,
                        right_sha256=records[j].sha256,
                        reason="high_text_similarity",
                        similarity=score,
                    )
                )

    return out
