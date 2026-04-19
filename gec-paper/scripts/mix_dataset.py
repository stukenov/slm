#!/usr/bin/env python3
"""Merge all GEC data sources, deduplicate, add identity, split."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gecpaper.data.mixer import mix_datasets, save_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default="data/synthetic.jsonl")
    parser.add_argument("--wiki", default="data/organic_wiki.jsonl")
    parser.add_argument("--social", default="data/organic_social.jsonl")
    parser.add_argument("--output_dir", default="data/mixed")
    parser.add_argument("--identity_ratio", type=float, default=0.05)
    args = parser.parse_args()

    sources = {}
    for name, path in [("synthetic", args.synthetic), ("wiki", args.wiki), ("social", args.social)]:
        p = Path(path)
        if p.exists():
            sources[name] = p

    if not sources:
        print("No data sources found. Generate data first.")
        return

    splits = mix_datasets(sources, identity_ratio=args.identity_ratio)
    save_splits(splits, Path(args.output_dir))


if __name__ == "__main__":
    main()
