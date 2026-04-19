#!/usr/bin/env python3
"""Train Qwen-distilled char-level morpheme segmenter."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_wordforms(data_path: str, max_words: int = 50000) -> list[str]:
    words = set()
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for field in ("input", "target"):
                text = item.get(field, "")
                for w in text.split():
                    w = w.strip(".,!?;:\"'()[]")
                    if len(w) >= 3:
                        words.add(w)
            if len(words) >= max_words:
                break
    return list(words)[:max_words]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "train"])
    parser.add_argument("--data_path", default="data/mixed/train.jsonl")
    parser.add_argument("--segmentation_data", default="data/morph_segmentations.jsonl")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_words", type=int, default=50000)
    parser.add_argument("--output_model", default="outputs/char_segmenter")
    args = parser.parse_args()

    if args.mode == "generate":
        from gecpaper.morph.segmenter import generate_segmentation_data

        words = extract_wordforms(args.data_path, args.max_words)
        logger.info("Extracted %d unique wordforms", len(words))

        generate_segmentation_data(
            words,
            model_name=args.model_name,
            output_path=Path(args.segmentation_data),
        )

    elif args.mode == "train":
        logger.info("Char-level segmenter training requires GPU.")
        logger.info("Segmentation data: %s", args.segmentation_data)
        logger.info("Output model: %s", args.output_model)
        logger.info("Run on server with PyTorch installed.")


if __name__ == "__main__":
    main()
