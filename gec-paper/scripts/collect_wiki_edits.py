#!/usr/bin/env python3
"""Download and extract GEC pairs from Kazakh Wikipedia edit history."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gecpaper.data.organic_wiki import extract_edit_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_revisions", type=int, default=5000)
    parser.add_argument("--max_edit_ratio", type=float, default=0.20)
    parser.add_argument("--output", default="data/organic_wiki.jsonl")
    args = parser.parse_args()

    extract_edit_pairs(
        max_revisions=args.max_revisions,
        max_edit_ratio=args.max_edit_ratio,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
