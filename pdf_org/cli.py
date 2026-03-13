from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .cache import JsonCache
from .category_optimizer import optimize_categories
from .extractor import discover_pdfs, extract_pdf_record
from .llm_classifier import OpenAICompatibleLLM
from .models import FileReportItem, RunReport, RunSummary
from .organizer import execute_action, plan_file_action
from .utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-driven PDF organizer")
    parser.add_argument("root_dir", type=Path, help="Root directory containing PDFs")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview actions without changing files")
    mode.add_argument("--apply", action="store_true", help="Apply move and rename operations")

    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument("--cache-dir", type=Path, default=Path(".pdf_org_cache"), help="Cache directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages to extract for preview")
    parser.add_argument("--min-docs-per-category", type=int, default=2, help="Minimum documents per final category")
    parser.add_argument("--max-docs-per-category", type=int, default=25, help="Maximum documents per final category")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    root_dir: Path = args.root_dir.resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Root directory not found or not a directory: {root_dir}")

    apply = bool(args.apply)
    if not args.apply and not args.dry_run:
        apply = False

    ensure_dir(args.cache_dir)
    cache = JsonCache(args.cache_dir)
    llm = OpenAICompatibleLLM(model=args.model, cache=cache)

    started_at = RunReport.now_iso()
    summary = RunSummary()
    errors: list[str] = []
    warnings: list[str] = []

    pdf_paths = discover_pdfs(root_dir)
    summary.scanned_files = len(pdf_paths)
    logger.info("Discovered %s PDF files", len(pdf_paths))

    records = []
    for path in pdf_paths:
        record = extract_pdf_record(path=path, root_dir=root_dir, max_pages=args.max_pages)
        records.append(record)
        if record.error:
            summary.extracted_failed += 1
            warnings.append(f"Extraction issue for {record.relative_path}: {record.error}")
        else:
            summary.extracted_success += 1

    classifications_by_sha = {}
    classified_entries: list[int] = []
    classifications_in_order = []
    for record_index, record in enumerate(records):
        try:
            classification = llm.classify_document(record)
            classifications_by_sha[record.sha256] = classification
            classifications_in_order.append(classification)
            classified_entries.append(record_index)
            summary.classified_success += 1
        except Exception as exc:  # noqa: BLE001
            summary.classified_failed += 1
            errors.append(f"Classification failure for {record.relative_path}: {exc}")

    optimization = optimize_categories(
        classifications_in_order,
        llm,
        min_docs_per_category=args.min_docs_per_category,
        max_docs_per_category=args.max_docs_per_category,
    )
    mapping = optimization.category_mapping or {}
    per_doc_mapping = optimization.document_category_mapping or {}
    per_record_mapping: dict[int, str] = {}
    for classification_idx_str, optimized_category in per_doc_mapping.items():
        try:
            classification_idx = int(classification_idx_str)
        except ValueError:
            continue
        if classification_idx < 0 or classification_idx >= len(classified_entries):
            continue
        record_index = classified_entries[classification_idx]
        per_record_mapping[record_index] = optimized_category

    file_reports: list[FileReportItem] = []
    for idx, record in enumerate(records):
        classification = classifications_by_sha.get(record.sha256)
        if not classification:
            file_reports.append(
                FileReportItem(
                    pdf=record,
                    classification=None,
                    optimized_category="uncategorized",
                    action=None,
                )
            )
            continue

        optimized_category = per_record_mapping.get(idx)
        action = plan_file_action(root_dir, record, classification, mapping, optimized_category=optimized_category)
        action = execute_action(action, apply=apply)

        summary.actions_planned += 1
        if action.executed:
            summary.actions_executed += 1
        if action.error:
            summary.actions_failed += 1
            errors.append(f"Action failed for {record.relative_path}: {action.error}")

        logger.info(
            "[%s] %s -> category=%s target=%s",
            "APPLY" if apply else "DRY-RUN",
            record.relative_path,
            action.optimized_category,
            action.target_path,
        )

        file_reports.append(
            FileReportItem(
                pdf=record,
                classification=classification,
                optimized_category=action.optimized_category,
                action=action,
            )
        )

    finished_at = RunReport.now_iso()
    report = RunReport(
        started_at=started_at,
        finished_at=finished_at,
        root_dir=str(root_dir),
        dry_run=not apply,
        model=args.model,
        max_pages=args.max_pages,
        summary=summary,
        category_optimization=optimization,
        files=file_reports,
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
