#!/usr/bin/env python3
"""Generate taxonomy-tagged synthetic GEC pairs via GPT-4o."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from openai import OpenAI
from datasets import load_dataset

from gecpaper.data.synthetic import generate_balanced_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_per_l2", type=int, default=600,
                        help="Target pairs per L2 category (18 categories x N = total)")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--output", default="data/synthetic.jsonl")
    parser.add_argument("--seed_dataset", default="kz-transformers/multidomain-kazakh-dataset")
    parser.add_argument("--max_seeds", type=int, default=20000)
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading seed texts from %s...", args.seed_dataset)
    ds = load_dataset(args.seed_dataset, split="train", streaming=True)
    seeds = []
    for row in ds:
        text = row.get("text", "").strip()
        if 20 < len(text) < 300 and text.count(" ") >= 3:
            seeds.append(text)
            if len(seeds) >= args.max_seeds:
                break
    logger.info("Collected %d seed texts", len(seeds))

    results = generate_balanced_dataset(
        client, seeds,
        target_per_l2=args.target_per_l2,
        model=args.model,
        output_path=output_path,
    )
    logger.info("Done. %d pairs written to %s", len(results), output_path)


if __name__ == "__main__":
    main()
