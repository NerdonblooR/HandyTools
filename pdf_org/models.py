from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PdfRecord:
    """Extracted content and metadata for a PDF."""

    file_path: str
    relative_path: str
    file_size: int
    sha256: str
    page_count: int | None
    metadata_title: str | None
    first_page_text: str
    content_snippet: str
    detected_title: str
    error: str | None = None

    @property
    def semantic_text(self) -> str:
        """Concise text used for semantic vectorization and clustering."""
        return "\n".join(
            part
            for part in [self.detected_title, self.metadata_title or "", self.first_page_text, self.content_snippet]
            if part
        )


@dataclass
class Cluster:
    """A semantically coherent group of PDFs."""

    cluster_id: str
    member_sha256: list[str]
    folder_name: str
    short_reason: str


@dataclass
class DuplicateCandidate:
    """Likely duplicate relation between two files."""

    left_sha256: str
    right_sha256: str
    reason: str
    similarity: float


@dataclass
class FileAction:
    """Planned or executed move operation for a single PDF."""

    source_path: str
    target_path: str
    folder_name: str
    title_used: str
    dry_run: bool
    executed: bool
    error: str | None = None


@dataclass
class FileReportItem:
    """Per-file result in the run report."""

    pdf: PdfRecord
    cluster_id: str
    cluster_name: str
    action: FileAction | None


@dataclass
class RunSummary:
    """Aggregate counters for the run."""

    scanned_files: int = 0
    extracted_success: int = 0
    extracted_failed: int = 0
    clusters_created: int = 0
    actions_planned: int = 0
    actions_executed: int = 0
    actions_failed: int = 0
    exact_duplicates: int = 0
    likely_duplicates: int = 0


@dataclass
class RunReport:
    """Top-level report payload for the CLI run."""

    started_at: str
    finished_at: str
    root_dir: str
    dry_run: bool
    model: str
    max_pages: int
    summary: RunSummary
    clusters: list[Cluster] = field(default_factory=list)
    files: list[FileReportItem] = field(default_factory=list)
    duplicates: list[DuplicateCandidate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
