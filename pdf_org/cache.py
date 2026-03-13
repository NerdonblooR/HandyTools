from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import ensure_dir


class JsonCache:
    """Simple JSON-file cache for reusable model outputs."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cluster_name_dir = cache_dir / "cluster_names"
        ensure_dir(self.cluster_name_dir)

    def get_cluster_name(self, cluster_key: str) -> dict[str, Any] | None:
        return self._read_json(self.cluster_name_dir / f"{cluster_key}.json")

    def set_cluster_name(self, cluster_key: str, payload: dict[str, Any]) -> None:
        self._write_json(self.cluster_name_dir / f"{cluster_key}.json", payload)

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
