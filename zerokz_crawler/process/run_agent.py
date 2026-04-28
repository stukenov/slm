"""
Run an additional agent on the already-processed dataset.

This lets you add new columns (summarize, NER, classify, etc.)
without reprocessing from raw HTML.

Usage:
  python run_agent.py --agent agents/summarize.py --column summary
  python run_agent.py --agent agents/ner.py --filter-lang kk

The agent script must define:
  AGENT = YourAgent()   # instance of BaseAgent

Flow:
  1. Stream rows from HF processed repo (parquet files)
  2. Apply agent to each row
  3. Upload updated parquet with new columns back to processed repo
     (overwrites the same file with added columns)
"""

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

PROCESSED_REPO = "stukenov/kaznet-processed"
STATE_FILE = Path(__file__).parent / "agent_state.json"


def load_agent(agent_path: str):
    spec = importlib.util.spec_from_file_location("custom_agent", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "AGENT"):
        raise ValueError(f"{agent_path} must define AGENT = YourAgent()")
    return mod.AGENT


def list_parquet_files(api: HfApi) -> list[str]:
    files = api.list_repo_files(PROCESSED_REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def process_file(api: HfApi, hf_path: str, agent, filter_lang: str | None, dry_run: bool):
    filename = Path(hf_path).name
    print(f"\nProcessing {filename}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        local = api.hf_hub_download(
            repo_id=PROCESSED_REPO,
            filename=hf_path,
            repo_type="dataset",
            local_dir=tmpdir,
        )
        df = pd.read_parquet(local)

    # Check if column already exists
    new_cols = agent.columns
    already_done = all(c in df.columns for c in new_cols)
    if already_done:
        print(f"  Columns {new_cols} already exist — skipping")
        return

    # Filter by language if requested
    mask = pd.Series([True] * len(df))
    if filter_lang and "language" in df.columns:
        mask = df["language"] == filter_lang
        print(f"  Filtering to lang={filter_lang}: {mask.sum()}/{len(df)} rows")

    # Apply agent
    results = []
    for i, row in enumerate(df[mask].itertuples(index=False)):
        row_dict = row._asdict()
        result = agent.safe_process(row_dict)
        results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{mask.sum()}...")

    # Merge new columns back
    result_df = pd.DataFrame(results, index=df[mask].index)
    for col in new_cols:
        df[col] = None
        df.loc[mask, col] = result_df[col]

    if dry_run:
        print(f"  [DRY] would upload {filename} with columns {new_cols}")
        print(df[new_cols].head(3).to_string())
        return

    # Upload back
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = f.name
    try:
        df.to_parquet(tmp, index=False, engine="pyarrow")
        size_mb = os.path.getsize(tmp) / 1024 / 1024
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=f"data/{filename}",
            repo_id=PROCESSED_REPO,
            repo_type="dataset",
        )
        print(f"  Uploaded {filename} ({size_mb:.1f} MB) with {new_cols}")
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="Path to agent .py file defining AGENT=")
    parser.add_argument("--filter-lang", help="Only process rows with this language (e.g. kk)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    agent = load_agent(args.agent)
    print(f"Agent: {agent.__class__.__name__}, columns: {agent.columns}")

    api = HfApi()
    files = list_parquet_files(api)
    print(f"Found {len(files)} parquet files in processed repo")

    for i, hf_path in enumerate(files):
        process_file(api, hf_path, agent, args.filter_lang, args.dry_run)
        if args.limit and i + 1 >= args.limit:
            break


if __name__ == "__main__":
    main()
