# HandyTools

## pdf_org (LLM-driven PDF organizer)

`pdf_org` is a local-first CLI that recursively scans PDFs, extracts metadata/text previews, asks an LLM for document classification, globally optimizes category taxonomy, and then plans/applies move+rename actions.

### Features

- Recursive deterministic PDF discovery
- Resilient extraction with PyMuPDF
- Two-pass LLM workflow:
  - per-document classification
  - global category optimization
- Safe `--dry-run` mode (default behavior unless `--apply`)
- Local JSON cache for document and taxonomy calls
- JSON run report with metadata, actions, errors, and warnings

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
python -m pdf_org.cli /path/to/library --dry-run --output report.json
python -m pdf_org.cli /path/to/library --apply --model gpt-4o-mini --cache-dir .pdf_org_cache --verbose
```

### LLM configuration

Set environment variables:

```bash
export OPENAI_API_KEY="..."
# optional
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

If no API key is set, the tool falls back to deterministic `uncategorized` behavior and still produces a report.

### Migration notes from the old keyword classifier

- Hard-coded keyword taxonomies were removed in favor of LLM-inferred categories.
- Category folders are now determined globally through a second optimization pass.
- Title detection now combines PDF metadata + LLM inference.
- All file operations are now explicitly gated behind `--apply`.
- LLM responses are cached by PDF SHA256 and category-set hash to reduce repeated API calls.
