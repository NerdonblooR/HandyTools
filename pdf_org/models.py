from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PdfRecord:
    """Raw extracted PDF information used for classification and reporting."""

    file_path: str
    relative_path: str
    file_size: int
    sha256: str
    page_count: int | None
    metadata_title: str | None
    text_preview: str
    error: str | None = None


@dataclass
class DocumentClassification:
    """Structured first-pass classification result from the LLM."""

    title: str
    document_type: str
    topic: str
    suggested_category: str
    confidence: float
    short_reason: str
    raw_response: str | None = None
    from_cache: bool = False


@dataclass
class CategoryOptimizationResult:
    """Global category simplification result from the LLM."""

    category_mapping: dict[str, str]
    normalized_categories: list[str]
    short_reason: str
    document_category_mapping: dict[str, str] = field(default_factory=dict)
    operations_log: list[str] = field(default_factory=list)
    category_sizes: dict[str, int] = field(default_factory=dict)
    raw_response: str | None = None
    from_cache: bool = False


@dataclass
class FileAction:
    """Planned or executed filesystem move/rename operation."""

    source_path: str
    target_path: str
    optimized_category: str
    title_used: str
    dry_run: bool
    executed: bool
    error: str | None = None


@dataclass
class FileReportItem:
    """Combined per-file result entry in the final run report."""

    pdf: PdfRecord
    classification: DocumentClassification | None
    optimized_category: str
    action: FileAction | None


@dataclass
class RunSummary:
    """Summary counts for run-level reporting."""

    scanned_files: int = 0
    extracted_success: int = 0
    extracted_failed: int = 0
    classified_success: int = 0
    classified_failed: int = 0
    actions_planned: int = 0
    actions_executed: int = 0
    actions_failed: int = 0


@dataclass
class RunReport:
    """Top-level JSON report payload for the CLI run."""

    started_at: str
    finished_at: str
    root_dir: str
    dry_run: bool
    model: str
    max_pages: int
    summary: RunSummary
    category_optimization: CategoryOptimizationResult
    files: list[FileReportItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
