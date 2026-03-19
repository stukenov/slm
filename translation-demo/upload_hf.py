#!/usr/bin/env python3
"""Upload translated FineWeb-Edu EN→KK parquet to HuggingFace."""

import argparse
from datasets import Dataset
from huggingface_hub import HfApi

REPO_ID = "saken-tukenov/sozkz-fineweb-edu-en-kk-1m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path", help="Path to parquet file")
    parser.add_argument("--repo-id", default=REPO_ID, help="HuggingFace repo ID")
    args = parser.parse_args()

    print(f"Loading {args.parquet_path}...")
    ds = Dataset.from_parquet(args.parquet_path)
    print(f"Loaded {len(ds)} rows")
    print(f"Columns: {ds.column_names}")
    print(f"Sample: {ds[0]['text_kk'][:200]}...")

    print(f"\nPushing to {args.repo_id}...")
    ds.push_to_hub(args.repo_id, private=False)
    print("Done!")


if __name__ == "__main__":
    main()
