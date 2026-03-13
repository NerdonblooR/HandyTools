from __future__ import annotations

import logging
from pathlib import Path

from .models import PdfRecord
from .utils import sha256_file

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore
except Exception:  # noqa: BLE001
    fitz = None


def discover_pdfs(root_dir: Path) -> list[Path]:
    """Recursively discover PDFs in stable deterministic order."""
    paths = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(paths, key=lambda p: str(p.relative_to(root_dir)).lower())


def extract_pdf_record(path: Path, root_dir: Path, max_pages: int = 3) -> PdfRecord:
    """Extract metadata + preview from a single PDF, returning resilient partial data on failure."""
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
            text_preview="",
            error="PyMuPDF not installed; cannot extract PDF content.",
        )

    try:
        doc = fitz.open(path)
        page_count = doc.page_count
        metadata_title = (doc.metadata or {}).get("title")

        text_chunks: list[str] = []
        for i in range(min(page_count, max_pages)):
            try:
                text_chunks.append(doc.load_page(i).get_text("text"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed extracting page %s from %s: %s", i, path, exc)

        preview = "\n".join(chunk.strip() for chunk in text_chunks if chunk.strip())
        return PdfRecord(
            file_path=str(path),
            relative_path=relative,
            file_size=size,
            sha256=digest,
            page_count=page_count,
            metadata_title=metadata_title.strip() if metadata_title else None,
            text_preview=preview,
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
            text_preview="",
            error=f"Extraction failed: {exc}",
        )
