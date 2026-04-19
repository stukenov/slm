#!/usr/bin/env python3
"""Correct social media Kazakh texts via GPT-4o to create organic GEC pairs."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from openai import OpenAI

from gecpaper.data.organic_social import correct_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Text file, one sentence per line")
    parser.add_argument("--output", default="data/organic_social.jsonl")
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(args.input) as f:
        texts = [line.strip() for line in f if line.strip()]

    correct_texts(client, texts, model=args.model, output_path=Path(args.output))


if __name__ == "__main__":
    main()
