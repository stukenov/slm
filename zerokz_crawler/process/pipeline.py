"""
Main processing pipeline.

Flow:
  HF raw repo (kaznet-crawl-raw)
    → download tar one by one (oldest first)
    → extract HTML files
    → run each HTML through agents (text, markdown, language, ...)
    → upload processed rows as parquet to HF processed repo
    → mark batch as done in state.json

Usage:
  python pipeline.py                  # process all pending batches
  python pipeline.py --limit 5        # process only 5 batches
  python pipeline.py --batch batch_0001_1_20260408_123141.tar.gz  # specific batch
  python pipeline.py --dry-run        # show what would be processed
"""

import argparse
import json
import os
import re
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from huggingface_hub import HfApi

from agents import DEFAULT_AGENTS

# ── Config ────────────────────────────────────────────────────────────────────

RAW_REPO = "stukenov/kaznet-crawl-raw"
PROCESSED_REPO = "stukenov/kaznet-processed"
STATE_FILE = Path(__file__).parent / "state.json"
BATCH_SIZE = 500          # rows per parquet upload chunk


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"done": [], "total_rows": 0, "last_processed": None}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── HF helpers ────────────────────────────────────────────────────────────────

def list_raw_batches(api: HfApi) -> list[str]:
    """Return sorted list of tar filenames on the raw repo (oldest first)."""
    files = api.list_repo_files(RAW_REPO, repo_type="dataset")
    tars = sorted(f for f in files if f.startswith("data/") and f.endswith(".tar.gz"))
    return tars  # e.g. ["data/batch_0001_1_20260408_123141.tar.gz", ...]


def download_tar(api: HfApi, hf_path: str, dest_dir: Path) -> Path:
    """Download a tar from HF raw repo to dest_dir."""
    filename = Path(hf_path).name
    local = dest_dir / filename
    url = api.hf_hub_download(
        repo_id=RAW_REPO,
        filename=hf_path,
        repo_type="dataset",
        local_dir=str(dest_dir),
    )
    return Path(url)


def upload_parquet(api: HfApi, df: pd.DataFrame, name: str, retries: int = 5):
    """Upload a DataFrame as parquet to the processed repo with retry on transient errors."""
    api.create_repo(PROCESSED_REPO, repo_type="dataset", exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = f.name
    try:
        df.to_parquet(tmp, index=False, engine="pyarrow")
        size_mb = os.path.getsize(tmp) / 1024 / 1024
        print(f"  Uploading {name} ({len(df)} rows, {size_mb:.1f} MB)...")
        for attempt in range(retries):
            try:
                api.upload_file(
                    path_or_fileobj=tmp,
                    path_in_repo=f"data/{name}",
                    repo_id=PROCESSED_REPO,
                    repo_type="dataset",
                )
                break  # success
            except Exception as e:
                if attempt < retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"  Upload failed (attempt {attempt+1}/{retries}): {e}")
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
    finally:
        os.unlink(tmp)


# ── URL helpers ───────────────────────────────────────────────────────────────

def filename_to_url(domain: str, filename: str) -> str:
    """Best-effort reconstruct URL from domain + scrapy filename."""
    # scrapy saves path_segments as underscores, no perfect reverse — return domain
    return f"https://{domain}/"


def extract_domain_from_path(site_dirname: str) -> str:
    """Convert crawled_sites dir name back to domain."""
    return site_dirname.replace("_", ".", 2) if "_" in site_dirname else site_dirname


# ── Row builder ───────────────────────────────────────────────────────────────

def build_base_row(domain: str, filename: str, html: str, batch_name: str) -> dict:
    """Build the base row before agent processing."""
    return {
        "domain": domain,
        "url": f"https://{domain}/",   # best we can do without seen_urls.txt
        "source_batch": batch_name,
        "html": html,
    }


def process_html_file(domain: str, fname: str, html: str, batch_name: str) -> dict:
    """Run all agents on one HTML file and return complete row."""
    row = build_base_row(domain, fname, html, batch_name)

    for agent in DEFAULT_AGENTS:
        result = agent.safe_process(row)
        row.update(result)

    # Drop raw HTML from output (too large; keep if you want by removing this line)
    row.pop("html", None)

    return row


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_tar(api: HfApi, hf_path: str, state: dict, dry_run: bool = False):
    batch_name = Path(hf_path).name  # e.g. batch_0001_1_20260408_123141.tar.gz

    if batch_name in state["done"]:
        print(f"  [SKIP] {batch_name} already processed")
        return

    if dry_run:
        print(f"  [DRY]  would process {batch_name}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {batch_name}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Download
        print(f"  Downloading...")
        t0 = time.time()
        local_tar = download_tar(api, hf_path, tmp)
        print(f"  Downloaded in {time.time()-t0:.1f}s ({local_tar.stat().st_size/1024/1024:.1f} MB)")

        # 2. Extract
        print(f"  Extracting...")
        extract_dir = tmp / "extracted"
        extract_dir.mkdir()
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(extract_dir)

        # 3. Find all HTML files
        html_files = list(extract_dir.rglob("*.html"))
        print(f"  Found {len(html_files)} HTML files")

        if not html_files:
            print("  No HTML files — skipping")
            state["done"].append(batch_name)
            save_state(state)
            return

        # 4. Process
        rows = []
        for i, html_path in enumerate(html_files):
            # Extract domain from path: extracted/domain_dir/html/file.html
            parts = html_path.relative_to(extract_dir).parts
            site_dirname = parts[0] if parts else "unknown"
            domain = extract_domain_from_path(site_dirname)

            try:
                html = html_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if len(html) < 100:  # skip empty/tiny pages
                continue

            row = process_html_file(domain, html_path.name, html, batch_name)

            # Skip pages with no extracted text
            if not row.get("text"):
                continue

            rows.append(row)

            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{len(html_files)}...")

        print(f"  Extracted {len(rows)} rows with text")

        if not rows:
            state["done"].append(batch_name)
            save_state(state)
            return

        # 5. Upload in chunks
        df = pd.DataFrame(rows)

        # Reorder columns nicely
        col_order = [
            "domain", "url", "title", "language", "lang_score",
            "text", "text_len", "markdown",
            "docling_md", "docling_json",
            "source_batch",
        ]
        df = df[[c for c in col_order if c in df.columns]]

        # Split into chunks if large
        for chunk_start in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[chunk_start:chunk_start + BATCH_SIZE]
            chunk_suffix = f"_part{chunk_start // BATCH_SIZE + 1}" if len(df) > BATCH_SIZE else ""
            parquet_name = batch_name.replace(".tar.gz", f"{chunk_suffix}.parquet")
            upload_parquet(api, chunk, parquet_name)

        # 6. Mark done
        state["done"].append(batch_name)
        state["total_rows"] = state.get("total_rows", 0) + len(rows)
        state["last_processed"] = batch_name
        save_state(state)

        print(f"  Done! {len(rows)} rows uploaded.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max batches to process (0=all)")
    parser.add_argument("--batch", type=str, help="Process specific batch file")
    parser.add_argument("--worker", type=str, default="", help="Worker id, e.g. '1/4' — process 1st quarter of batches")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    state = load_state()

    print(f"State: {len(state['done'])} batches done, {state.get('total_rows', 0)} total rows")

    if args.batch:
        hf_path = f"data/{args.batch}" if not args.batch.startswith("data/") else args.batch
        process_tar(api, hf_path, state, dry_run=args.dry_run)
        return

    batches = list_raw_batches(api)
    pending = [b for b in batches if Path(b).name not in state["done"]]

    # Slice for parallel workers: --worker 1/4 means this pod handles 1st quarter
    if args.worker:
        idx, total = map(int, args.worker.split("/"))
        chunk = len(pending) // total
        start = (idx - 1) * chunk
        end = start + chunk if idx < total else len(pending)
        pending = pending[start:end]
        print(f"Worker {idx}/{total}: batches {start}–{end} ({len(pending)} total)")

    print(f"Batches on HF: {len(batches)} total, {len(pending)} pending\n")

    count = 0
    for hf_path in pending:
        process_tar(api, hf_path, state, dry_run=args.dry_run)
        count += 1
        if args.limit and count >= args.limit:
            print(f"\nReached limit of {args.limit} batches.")
            break

    print(f"\nFinished. Total rows processed: {state.get('total_rows', 0)}")


if __name__ == "__main__":
    main()
