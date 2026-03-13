from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .models import DocumentClassification, FileAction, PdfRecord
from .utils import ensure_dir, normalize_category_name, sanitize_filename, unique_target_path

logger = logging.getLogger(__name__)


def plan_file_action(
    root_dir: Path,
    pdf: PdfRecord,
    classification: DocumentClassification,
    category_mapping: dict[str, str],
) -> FileAction:
    """Plan a safe move+rename action based on optimized category and title."""
    raw_category = normalize_category_name(classification.suggested_category)
    optimized_category = normalize_category_name(category_mapping.get(raw_category, "uncategorized"))

    category_dir = root_dir / optimized_category
    title = sanitize_filename(classification.title) if classification.title else sanitize_filename(Path(pdf.file_path).stem)
    target = unique_target_path(category_dir, title, ext=Path(pdf.file_path).suffix.lower() or ".pdf")

    return FileAction(
        source_path=pdf.file_path,
        target_path=str(target),
        optimized_category=optimized_category,
        title_used=title,
        dry_run=True,
        executed=False,
    )


def execute_action(action: FileAction, apply: bool) -> FileAction:
    """Execute or dry-run a single action with robust error capture."""
    action.dry_run = not apply
    if not apply:
        return action

    src = Path(action.source_path)
    dst = Path(action.target_path)
    try:
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
        action.executed = True
        return action
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed moving %s -> %s: %s", src, dst, exc)
        action.error = str(exc)
        action.executed = False
        return action
