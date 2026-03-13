from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from .models import PdfRecord

logger = logging.getLogger(__name__)


@dataclass
class ClusterPlan:
    """Intermediate cluster structure before naming."""

    cluster_id: str
    member_sha256: list[str]


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)]


def _vectorize(texts: list[str]) -> list[Counter[str]]:
    return [Counter(_tokenize(text)) for text in texts]


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _distance_threshold(size: int) -> float:
    if size < 8:
        return 0.6
    if size < 25:
        return 0.65
    return 0.7


def build_clusters(records: list[PdfRecord], min_cluster_size: int = 2) -> tuple[list[ClusterPlan], dict[str, int]]:
    """Cluster records with simple graph-connected-components on cosine similarity."""
    if not records:
        return [], {}

    vecs = _vectorize([r.semantic_text or r.detected_title for r in records])
    threshold = _distance_threshold(len(records))

    adjacency: dict[int, set[int]] = {i: set() for i in range(len(records))}
    similarity: dict[tuple[int, int], float] = {}
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            score = _cosine(vecs[i], vecs[j])
            similarity[(i, j)] = score
            if score >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    grouped = _connected_components(adjacency, records)
    merged = _merge_small_clusters(grouped, similarity, records, min_cluster_size)

    plans = [
        ClusterPlan(cluster_id=f"cluster_{idx + 1}", member_sha256=members)
        for idx, members in enumerate(sorted(merged.values(), key=lambda m: (-len(m), m[0])))
    ]

    membership: dict[str, int] = {}
    for idx, plan in enumerate(plans):
        for sha in plan.member_sha256:
            membership[sha] = idx
    logger.info("Created %s semantic clusters", len(plans))
    return plans, membership


def _connected_components(adjacency: dict[int, set[int]], records: list[PdfRecord]) -> dict[int, list[str]]:
    visited: set[int] = set()
    grouped: dict[int, list[str]] = {}
    cluster_id = 0

    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        members: list[str] = []
        while stack:
            cur = stack.pop()
            members.append(records[cur].sha256)
            for nxt in adjacency[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        grouped[cluster_id] = members
        cluster_id += 1
    return grouped


def _merge_small_clusters(
    grouped: dict[int, list[str]], similarity: dict[tuple[int, int], float], records: list[PdfRecord], min_cluster_size: int
) -> dict[int, list[str]]:
    index_of_sha = {r.sha256: i for i, r in enumerate(records)}
    clusters = {k: v[:] for k, v in grouped.items()}

    small_ids = [cid for cid, members in clusters.items() if len(members) < min_cluster_size]
    for cid in small_ids:
        members = clusters.get(cid)
        if not members:
            continue

        best_target = None
        best_score = -1.0
        for other_id, other_members in clusters.items():
            if other_id == cid or not other_members:
                continue
            score = _average_similarity(members, other_members, similarity, index_of_sha)
            if score > best_score:
                best_score = score
                best_target = other_id

        if best_target is not None and best_score >= 0.38:
            clusters[best_target].extend(members)
            clusters[cid] = []

    return {cid: members for cid, members in clusters.items() if members}


def _average_similarity(
    left: list[str], right: list[str], similarity: dict[tuple[int, int], float], index_of_sha: dict[str, int]
) -> float:
    scores = []
    for l in left:
        for r in right:
            i, j = sorted((index_of_sha[l], index_of_sha[r]))
            scores.append(similarity.get((i, j), 0.0))
    return sum(scores) / max(len(scores), 1)


def refine_cluster_sizes(
    plans: list[ClusterPlan],
    records_by_sha: dict[str, PdfRecord],
    large_cluster_threshold: int = 15,
) -> list[ClusterPlan]:
    """Split very large clusters and keep tiny-cluster count controlled."""
    refined: list[ClusterPlan] = []
    next_id = 1
    for plan in plans:
        if len(plan.member_sha256) <= large_cluster_threshold:
            refined.append(ClusterPlan(cluster_id=f"cluster_{next_id}", member_sha256=plan.member_sha256))
            next_id += 1
            continue

        sub_records = [records_by_sha[s] for s in plan.member_sha256]
        subplans, _ = build_clusters(sub_records, min_cluster_size=3)
        for sub in subplans:
            refined.append(ClusterPlan(cluster_id=f"cluster_{next_id}", member_sha256=sub.member_sha256))
            next_id += 1
    return refined
