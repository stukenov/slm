#!/usr/bin/env python3
"""Train XLM-RoBERTa edit tagger for GEC."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from gecpaper.models.edit_tagger import build_tag_vocab, extract_edit_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_pairs(path: str) -> list[tuple[list[str], list[str]]]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            src = item.get("input", "").strip()
            tgt = item.get("target", "").strip()
            if src and tgt and src != tgt:
                pairs.append((src.split(), tgt.split()))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round2_tagger.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--build_vocab_only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_path = args.data_path or config.get("data_path", "data/mixed/train.jsonl")
    top_k = config.get("top_k_tags", 2000)

    logger.info("Loading data from %s", data_path)
    pairs = load_pairs(data_path)
    logger.info("Loaded %d pairs", len(pairs))

    logger.info("Building tag vocab (top_k=%d)...", top_k)
    vocab = build_tag_vocab(pairs, top_k=top_k)
    logger.info("Tag vocab size: %d", len(vocab))

    vocab_path = Path(config.get("output_dir", "outputs/round2_tagger")) / "tag_vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info("Tag vocab saved to %s", vocab_path)

    if args.build_vocab_only:
        return

    logger.info("Full XLM-R training requires GPU. Run on server with transformers installed.")
    logger.info("Config: model=%s, epochs=%d, batch=%d, lr=%s",
                config.get("model_name", "xlm-roberta-base"),
                config.get("num_train_epochs", 5),
                config.get("batch_size", 32),
                config.get("learning_rate", 5e-5))


if __name__ == "__main__":
    main()
