from __future__ import annotations

import logging
import re
from pathlib import Path

from .models import PdfRecord
from .utils import sanitize_filename, sha256_file

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore
except Exception:  # noqa: BLE001
    fitz = None


def discover_pdfs(root_dir: Path) -> list[Path]:
    """Recursively discover PDFs in deterministic order."""
    paths = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(paths, key=lambda p: str(p.relative_to(root_dir)).lower())


def detect_title(metadata_title: str | None, first_page_text: str, path: Path) -> str:
    """Detect a likely human-readable title from metadata/text/file name."""
    if metadata_title and len(metadata_title.strip()) > 3:
        return sanitize_filename(metadata_title.strip())

    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    candidates = [line for line in lines[:10] if 6 <= len(line) <= 150 and not line.lower().startswith("arxiv:")]
    if candidates:
        best = max(candidates, key=lambda s: len(re.findall(r"[A-Za-z]", s)))
        return sanitize_filename(best)

    return sanitize_filename(path.stem)


def extract_pdf_record(path: Path, root_dir: Path, max_pages: int = 3) -> PdfRecord:
    """Extract resilient text/metadata fields from a single PDF."""
    relative = str(path.relative_to(root_dir))
    size = path.stat().st_size
    digest = sha256_file(path)

    if fitz is None:
        return PdfRecord(
            file_path=str(path),
            relative_path=relative,
            file_size=size,
            sha256=digest,
            page_count=None,
            metadata_title=None,
            first_page_text="",
            content_snippet="",
            detected_title=sanitize_filename(path.stem),
            error="PyMuPDF not installed; cannot extract PDF content.",
        )

    try:
        doc = fitz.open(path)
        page_count = doc.page_count
        metadata_title = (doc.metadata or {}).get("title")

        text_chunks: list[str] = []
        first_page_text = ""
        for i in range(min(page_count, max_pages)):
            try:
                page_text = doc.load_page(i).get_text("text").strip()
                if i == 0:
                    first_page_text = page_text
                if page_text:
                    text_chunks.append(page_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed extracting page %s from %s: %s", i, path, exc)

        snippet = "\n".join(text_chunks)[:6000]
        title = detect_title(metadata_title=metadata_title, first_page_text=first_page_text, path=path)
        return PdfRecord(
            file_path=str(path),
            relative_path=relative,
            file_size=size,
            sha256=digest,
            page_count=page_count,
            metadata_title=metadata_title.strip() if metadata_title else None,
            first_page_text=first_page_text,
            content_snippet=snippet,
            detected_title=title,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF extraction failed for %s", path)
        return PdfRecord(
            file_path=str(path),
            relative_path=relative,
            file_size=size,
            sha256=digest,
            page_count=None,
            metadata_title=None,
            first_page_text="",
            content_snippet="",
            detected_title=sanitize_filename(path.stem),
            error=f"Extraction failed: {exc}",
        )
