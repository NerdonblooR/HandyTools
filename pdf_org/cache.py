from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import ensure_dir


class JsonCache:
    """Simple JSON-file cache for document and optimization LLM responses."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.doc_dir = cache_dir / "documents"
        self.opt_dir = cache_dir / "optimizations"
        ensure_dir(self.doc_dir)
        ensure_dir(self.opt_dir)

    def get_document(self, sha256: str) -> dict[str, Any] | None:
        return self._read_json(self.doc_dir / f"{sha256}.json")

    def set_document(self, sha256: str, payload: dict[str, Any]) -> None:
        self._write_json(self.doc_dir / f"{sha256}.json", payload)

    def get_optimization(self, categories_hash: str) -> dict[str, Any] | None:
        return self._read_json(self.opt_dir / f"{categories_hash}.json")

    def set_optimization(self, categories_hash: str, payload: dict[str, Any]) -> None:
        self._write_json(self.opt_dir / f"{categories_hash}.json", payload)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
