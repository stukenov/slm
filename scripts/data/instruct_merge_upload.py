#!/usr/bin/env python3
"""Merge instruct chunks and upload to HuggingFace.

Usage:
    python3 instruct_merge_upload.py --input_dir /root/instruct_kk --repo stukenov/sozkz-corpus-instruct-kk-alpaca-v1
"""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/root/instruct_kk")
    parser.add_argument("--repo", default="stukenov/sozkz-corpus-instruct-kk-alpaca-v1")
    parser.add_argument("--output", default=None, help="Merged output file (default: input_dir/merged.jsonl)")
    parser.add_argument("--upload", action="store_true", help="Upload to HF")
    args = parser.parse_args()

    output = args.output or os.path.join(args.input_dir, "merged.jsonl")
    chunks = sorted(glob.glob(os.path.join(args.input_dir, "chunks", "chunk_*.jsonl")))

    if not chunks:
        print("No chunks found!")
        return

    # Merge and deduplicate
    seen = set()
    total = 0
    dupes = 0

    with open(output, "w") as out:
        for chunk_file in chunks:
            with open(chunk_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = item.get("instruction_kk", "")
                    if key in seen:
                        dupes += 1
                        continue
                    seen.add(key)
                    out.write(line)
                    total += 1

    print(f"Merged {len(chunks)} chunks: {total} unique pairs ({dupes} duplicates removed)")
    print(f"Saved to {output}")

    if args.upload:
        token = os.environ.get("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
        from huggingface_hub import HfApi, create_repo
        create_repo(args.repo, token=token, exist_ok=True, repo_type="dataset")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=output,
            path_in_repo="data/train.jsonl",
            repo_id=args.repo,
            repo_type="dataset",
            token=token,
        )
        # Upload progress
        progress_file = os.path.join(args.input_dir, "progress.json")
        if os.path.exists(progress_file):
            api.upload_file(
                path_or_fileobj=progress_file,
                path_in_repo="progress.json",
                repo_id=args.repo,
                repo_type="dataset",
                token=token,
            )
        print(f"Uploaded to {args.repo}")


if __name__ == "__main__":
    main()
