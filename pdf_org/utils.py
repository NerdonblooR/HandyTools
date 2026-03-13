from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "were",
    "have",
    "has",
    "into",
    "using",
    "study",
    "analysis",
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


def sanitize_filename(name: str, max_len: int = 180) -> str:
    cleaned = name.strip().replace("/", " ").replace("\\", " ")
    cleaned = re.sub(r"[<>:\"|?*\x00-\x1F]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = "untitled"
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_file"
    return cleaned[:max_len].rstrip(" .") or "untitled"


def slugify_folder_name(name: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s_-]", "", name).strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:max_len] or "misc"


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


def pick_reasonable_label(text: str) -> str:
    """Deterministic fallback label from frequent non-trivial words."""
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)]
    filtered = [t for t in tokens if t not in STOPWORDS]
    if not filtered:
        return "misc"

    freq: dict[str, int] = {}
    for token in filtered:
        freq[token] = freq.get(token, 0) + 1

    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return "_".join(word for word, _ in top)
