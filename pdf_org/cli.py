from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from .cache import JsonCache
from .duplicates import find_duplicate_candidates
from .extractor import discover_pdfs, extract_pdf_record
from .llm_classifier import ClusterNamer
from .models import Cluster, FileReportItem, RunReport, RunSummary
from .organizer import execute_action, plan_file_action
from .semantic_grouping import build_clusters, refine_cluster_sizes
from .utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster-based PDF organizer")
    parser.add_argument("root_dir", type=Path, help="Root directory containing PDFs")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview actions without changing files")
    mode.add_argument("--apply", action="store_true", help="Apply move and rename operations")

    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument("--cache-dir", type=Path, default=Path(".pdf_org_cache"), help="Cache directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model used only for cluster naming")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages to extract per PDF")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Minimum target cluster size")
    parser.add_argument("--large-cluster-threshold", type=int, default=15, help="Split clusters larger than this")
    parser.add_argument("--duplicate-threshold", type=float, default=0.94, help="Near-duplicate cosine threshold")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


def _cluster_cache_key(member_sha256: list[str]) -> str:
    payload = "\n".join(sorted(member_sha256))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    root_dir: Path = args.root_dir.resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Root directory not found or not a directory: {root_dir}")

    apply = bool(args.apply)
    if not args.apply and not args.dry_run:
        apply = False

    cache = JsonCache(args.cache_dir)
    namer = ClusterNamer(model=args.model, cache=cache)

    started_at = RunReport.now_iso()
    summary = RunSummary()
    errors: list[str] = []
    warnings: list[str] = []

    pdf_paths = discover_pdfs(root_dir)
    summary.scanned_files = len(pdf_paths)
    logger.info("Discovered %s PDF files", len(pdf_paths))

    records = [extract_pdf_record(path=path, root_dir=root_dir, max_pages=args.max_pages) for path in pdf_paths]
    records_by_sha = {r.sha256: r for r in records}

    for record in records:
        if record.error:
            summary.extracted_failed += 1
            warnings.append(f"Extraction issue for {record.relative_path}: {record.error}")
        else:
            summary.extracted_success += 1

    cluster_plans, _ = build_clusters(records, min_cluster_size=args.min_cluster_size)
    cluster_plans = refine_cluster_sizes(
        cluster_plans,
        records_by_sha=records_by_sha,
        large_cluster_threshold=args.large_cluster_threshold,
    )

    clusters: list[Cluster] = []
    cluster_name_by_sha: dict[str, tuple[str, str]] = {}
    for plan in cluster_plans:
        members = [records_by_sha[sha] for sha in plan.member_sha256 if sha in records_by_sha]
        folder_name, reason = namer.name_cluster(_cluster_cache_key(plan.member_sha256), members)
        clusters.append(Cluster(cluster_id=plan.cluster_id, member_sha256=plan.member_sha256, folder_name=folder_name, short_reason=reason))
        for sha in plan.member_sha256:
            cluster_name_by_sha[sha] = (plan.cluster_id, folder_name)

    summary.clusters_created = len(clusters)

    duplicates = find_duplicate_candidates(records, near_threshold=args.duplicate_threshold)
    summary.exact_duplicates = sum(1 for d in duplicates if d.reason == "exact_sha256_match")
    summary.likely_duplicates = sum(1 for d in duplicates if d.reason != "exact_sha256_match")

    file_reports: list[FileReportItem] = []
    for record in records:
        cluster_id, cluster_name = cluster_name_by_sha.get(record.sha256, ("cluster_uncategorized", "misc"))
        action = plan_file_action(root_dir=root_dir, pdf=record, folder_name=cluster_name)
        action = execute_action(action, apply=apply)

        summary.actions_planned += 1
        if action.executed:
            summary.actions_executed += 1
        if action.error:
            summary.actions_failed += 1
            errors.append(f"Action failed for {record.relative_path}: {action.error}")

        logger.info("[%s] %s -> folder=%s target=%s", "APPLY" if apply else "DRY-RUN", record.relative_path, action.folder_name, action.target_path)

        file_reports.append(FileReportItem(pdf=record, cluster_id=cluster_id, cluster_name=cluster_name, action=action))

    finished_at = RunReport.now_iso()
    report = RunReport(
        started_at=started_at,
        finished_at=finished_at,
        root_dir=str(root_dir),
        dry_run=not apply,
        model=args.model,
        max_pages=args.max_pages,
        summary=summary,
        clusters=clusters,
        files=file_reports,
        duplicates=duplicates,
        errors=errors,
        warnings=warnings,
    )

    if args.output:
        args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote report to %s", args.output)

    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
