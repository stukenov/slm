# Processing Pipeline

Converts raw HTML tars (`kaznet-crawl-raw`) → structured parquet (`kaznet-processed`).

## Setup

```bash
cd process
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run pipeline

```bash
# Process all pending batches
python pipeline.py

# Process only 10 batches (for testing)
python pipeline.py --limit 10

# Dry run — show what would be processed
python pipeline.py --dry-run

# One specific batch
python pipeline.py --batch batch_0001_1_20260408_123141.tar.gz
```

## Output schema (per row)

| Column | Description |
|--------|-------------|
| `domain` | Site domain (e.g. `turkystan.kz`) |
| `url` | Source URL (reconstructed from domain) |
| `title` | Page `<title>` tag |
| `language` | Detected language code (`kk`, `ru`, `en`, ...) |
| `lang_score` | Confidence 0–1 |
| `text` | Clean plain text (trafilatura) |
| `text_len` | Character count |
| `markdown` | HTML→Markdown (markdownify) |
| `source_batch` | Which tar file this came from |

## Add a new column (agent)

1. Create `agents/my_agent.py`:

```python
from agents.base import BaseAgent

class MyAgent(BaseAgent):
    columns = ["my_column"]

    def process(self, row: dict) -> dict:
        text = row.get("text", "")
        return {"my_column": text[:100]}  # your logic here

AGENT = MyAgent()
```

2. Run it on existing processed data:

```bash
# Add column to all rows
python run_agent.py --agent agents/my_agent.py

# Only on Kazakh rows
python run_agent.py --agent agents/my_agent.py --filter-lang kk

# Test on 2 files first
python run_agent.py --agent agents/my_agent.py --limit 2 --dry-run
```

The agent reads each parquet, adds the column, uploads back. Skips files that already have the column.

## Available agents

| Agent | Columns |
|-------|---------|
| `TitleAgent` | `title` |
| `TextAgent` | `text`, `text_len` |
| `MarkdownAgent` | `markdown` |
| `LanguageAgent` | `language`, `lang_score` |
| `agents/summarize_example.py` | `summary` (example) |
