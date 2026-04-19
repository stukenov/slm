#!/usr/bin/env python3
"""Generate multi-reference GEC benchmark via GPT-4o (5 correction strategies)."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from openai import OpenAI

from gecpaper.data.multi_ref import generate_multi_ref_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True, help="JSONL test file")
    parser.add_argument("--output", default="data/multi_ref_benchmark.jsonl")
    parser.add_argument("--max_sentences", type=int, default=700)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    test_data = []
    with open(args.test_data) as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    generate_multi_ref_benchmark(
        client, test_data,
        max_sentences=args.max_sentences,
        model=args.model,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
