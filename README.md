# HandyTools

## `pdf_org` — cluster-first PDF organizer

`pdf_org` recursively scans a folder tree, extracts PDF text, clusters semantically similar files, uses an LLM only to name each cluster folder, then renames/moves files safely.

## Why this design

- **Clustering is primary**: no hard-coded categories.
- **LLM usage is minimal**: only for cluster naming (with cache + fallback).
- **Safe by default**: dry-run unless `--apply` is explicitly set.
- **Nested folders supported**: recursive discovery remains on.
- **Duplicates are reported, never auto-deleted**.

## Pipeline

1. Discover all PDFs recursively.
2. Extract metadata, first-page text, and snippet (malformed files handled gracefully).
3. Detect likely title (metadata → first-page heuristic → filename fallback).
4. Build lightweight token-frequency semantic vectors.
5. Cluster via cosine-similarity graph connected components.
6. Merge very small clusters when semantically close.
7. Split oversized clusters into smaller coherent subclusters.
8. Name each final cluster folder with LLM (or deterministic fallback).
9. Rename files using detected title and move into final cluster folder.
10. Report exact and likely duplicates.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

- `PyMuPDF`: reliable PDF extraction.
- No heavy ML dependency required: clustering uses a lightweight token-cosine graph approach.
- `requests`: OpenAI-compatible API call for cluster naming.

## CLI

```bash
python -m pdf_org.cli ROOT_DIR [--dry-run | --apply] [options]
```

### Important flags

- `--dry-run`: preview only (default behavior if no mode passed).
- `--apply`: execute move/rename operations.
- `--model`: LLM model for cluster naming only.
- `--max-pages`: pages used for extraction.
- `--min-cluster-size`: merge pressure for tiny clusters.
- `--large-cluster-threshold`: split very large clusters.
- `--duplicate-threshold`: near-duplicate similarity threshold.
- `--output`: save JSON report.
- `--verbose`: debug logging.

## Usage examples

```bash
# Safe preview
python -m pdf_org.cli ~/PDFs --dry-run --output report.json

# Preview with custom clustering controls
python -m pdf_org.cli ~/PDFs --dry-run --min-cluster-size 3 --large-cluster-threshold 18

# Apply organization
python -m pdf_org.cli ~/PDFs --apply --output applied_report.json
```

## Notes

- If `OPENAI_API_KEY` is missing, naming falls back to deterministic labels.
- Duplicate detection is advisory; manual review is recommended.
- Output report includes clusters, per-file actions, and duplicate candidates.
