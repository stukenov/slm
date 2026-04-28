#!/usr/bin/env python3
"""Merge generated GEC chunks and optionally upload as a dataset."""

from __future__ import annotations

import argparse
import glob
import json
import os


def line_key(item: dict) -> tuple[str, str]:
    return (item.get("input", ""), item.get("output", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./gec_qwen_cloudrift")
    parser.add_argument("--output", default=None)
    parser.add_argument("--repo", default="stukenov/sozkz-corpus-synthetic-kk-gec-qwen35-v1")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    chunk_paths = sorted(glob.glob(os.path.join(args.input_dir, "chunks", "*.jsonl")))
    if not chunk_paths:
        print("No chunk files found.")
        return

    output = args.output or os.path.join(args.input_dir, "merged.jsonl")
    seen = set()
    total = 0
    dupes = 0

    with open(output, "w") as out:
        for path in chunk_paths:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = line_key(item)
                    if key in seen:
                        dupes += 1
                        continue
                    seen.add(key)
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total += 1

    print(f"Merged {len(chunk_paths)} files: {total} unique rows ({dupes} duplicates removed)")
    print(f"Saved to {output}")

    if args.upload:
        token = os.environ.get("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
        from huggingface_hub import HfApi, create_repo

        create_repo(args.repo, token=token, exist_ok=True, repo_type="dataset")
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=output,
            path_in_repo="data/train.jsonl",
            repo_id=args.repo,
            repo_type="dataset",
            token=token,
        )
        progress = os.path.join(args.input_dir, "progress.json")
        if os.path.exists(progress):
            api.upload_file(
                path_or_fileobj=progress,
                path_in_repo="progress.json",
                repo_id=args.repo,
                repo_type="dataset",
                token=token,
            )
        print(f"Uploaded to {args.repo}")


if __name__ == "__main__":
    main()
