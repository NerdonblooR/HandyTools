from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_for_categories(categories: list[str]) -> str:
    normalized = [normalize_category_name(c) for c in categories if c.strip()]
    payload = "\n".join(sorted(set(normalized)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_category_name(value: str) -> str:
    cleaned = re.sub(r"\s+", "_", value.strip().lower())
    cleaned = re.sub(r"[^a-z0-9_-]", "", cleaned)
    return cleaned or "uncategorized"


def sanitize_filename(name: str, max_len: int = 180) -> str:
    cleaned = name.strip().replace("/", " ").replace("\\", " ")
    cleaned = re.sub(r"[<>:\"|?*\x00-\x1F]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = "untitled"
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_file"
    return cleaned[:max_len].rstrip(" .") or "untitled"


def unique_target_path(target_dir: Path, filename_no_ext: str, ext: str = ".pdf") -> Path:
    candidate = target_dir / f"{filename_no_ext}{ext}"
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        candidate = target_dir / f"{filename_no_ext}_{idx}{ext}"
        if not candidate.exists():
            return candidate
        idx += 1


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Attempt to parse a JSON object from free-form LLM text."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None
