# HandyTools

## `pdf_org` — LLM-driven PDF organizer

`pdf_org` is a local-first CLI for organizing large PDF libraries.

It scans a directory recursively, extracts metadata and text previews from each PDF, classifies each file with an OpenAI-compatible model, globally normalizes category names, and then plans or applies file move/rename operations.

---

## What it does

- Recursively discovers `.pdf` files under a root folder.
- Extracts metadata/title/page count and a text preview (first N pages).
- Classifies each PDF into a category with an LLM.
- Runs a second pass to optimize category taxonomy (merge/normalize duplicates).
- Plans target file paths and folder names.
- Supports safe preview mode (`--dry-run`) and execution mode (`--apply`).
- Writes a full JSON report to stdout and optionally to a file.
- Caches document classification and category optimization responses locally.

---

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`
- Optional but recommended: `OPENAI_API_KEY`

If no API key is configured, `pdf_org` still runs with deterministic fallback classification (`uncategorized`) and identity category mapping.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

### 1) Configure environment variables

```bash
export OPENAI_API_KEY="your_api_key"
# optional (defaults to https://api.openai.com/v1)
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 2) Run in preview mode (recommended first)

```bash
python -m pdf_org.cli /path/to/pdf/library --dry-run --output report.json
```

This shows planned actions without modifying files.

### 3) Apply changes

```bash
python -m pdf_org.cli /path/to/pdf/library --apply --output report.json
```

This performs move/rename operations.

---

## CLI usage

```bash
python -m pdf_org.cli ROOT_DIR [--dry-run | --apply] [options]
```

### Positional argument

- `ROOT_DIR` — Root directory containing PDFs.

### Modes

- `--dry-run` — Preview actions without changing files.
- `--apply` — Execute planned move/rename operations.

If neither flag is set, behavior is dry-run by default.

### Options

- `--output PATH` — Write JSON report to this path.
- `--cache-dir PATH` — Cache directory (default: `.pdf_org_cache`).
- `--model NAME` — LLM model name (default: `gpt-4o-mini`).
- `--max-pages N` — Max pages used for text preview extraction (default: `3`).
- `--verbose` — Enable verbose logging.

### Example commands

```bash
# Preview with verbose logs
python -m pdf_org.cli ~/Documents/PDFs --dry-run --verbose

# Apply changes with custom model and cache location
python -m pdf_org.cli ~/Documents/PDFs --apply --model gpt-4o-mini --cache-dir .pdf_org_cache

# Preview and save report
python -m pdf_org.cli ~/Documents/PDFs --dry-run --output run_report.json
```

---

## Report output

`pdf_org` always prints a JSON report to stdout. With `--output`, it also writes the same report to disk.

The report includes:

- Run metadata (`started_at`, `finished_at`, `root_dir`, `model`, `dry_run`)
- Summary counts (scan/extract/classify/action success/failure)
- Category optimization result and mapping
- Per-file extraction/classification/action details
- Aggregated warnings and errors

---

## Cache behavior

By default, cache files are stored in `.pdf_org_cache/`.

- Document classifications are cached by PDF SHA256.
- Category optimization is cached by a stable hash of raw category sets.

This reduces repeated API calls across runs.

---

## Typical workflow

1. Run `--dry-run` and inspect the JSON report.
2. Optionally tune `--max-pages` or `--model`.
3. Run again with `--apply`.
4. Keep the report for audit/history.

---

## Troubleshooting

- **No API key set**: tool falls back to `uncategorized` behavior by design.
- **LLM response malformed**: tool uses fallback classification or identity mapping and continues.
- **Network/API failures**: request errors are logged; run report records failures.
- **Extraction issues for some PDFs**: run continues; warnings include per-file extraction errors.

---

## Notes on behavior changes (vs keyword-based approach)

- Hardcoded keyword taxonomy logic has been removed.
- Categories are now inferred and then globally normalized in a second pass.
- Operations are gated behind explicit `--apply`.
- Metadata title and content preview both influence title/category inference.
