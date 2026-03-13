from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .models import FileAction, PdfRecord
from .utils import ensure_dir, sanitize_filename, slugify_folder_name, unique_target_path

logger = logging.getLogger(__name__)


def plan_file_action(root_dir: Path, pdf: PdfRecord, folder_name: str) -> FileAction:
    """Plan safe move/rename action from final cluster folder and detected title."""
    folder = slugify_folder_name(folder_name)
    category_dir = root_dir / folder
    title = sanitize_filename(pdf.detected_title) if pdf.detected_title else sanitize_filename(Path(pdf.file_path).stem)
    target = unique_target_path(category_dir, title, ext=Path(pdf.file_path).suffix.lower() or ".pdf")

    return FileAction(
        source_path=pdf.file_path,
        target_path=str(target),
        folder_name=folder,
        title_used=title,
        dry_run=True,
        executed=False,
    )


def execute_action(action: FileAction, apply: bool) -> FileAction:
    """Execute (or preview) action and capture errors."""
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
