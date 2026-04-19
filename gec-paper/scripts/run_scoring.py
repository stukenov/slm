#!/usr/bin/env python3
"""Run GEC model assessment on a test set."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from gecpaper.models.nllb_gec import generate_correction
from gecpaper.scoring.benchmark import print_metrics, run_assessment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or HF model name")
    parser.add_argument("--test_data", required=True, help="JSONL test file")
    parser.add_argument("--model_type", default="nllb", choices=["nllb"])
    parser.add_argument("--multi_ref", action="store_true")
    parser.add_argument("--output", default=None, help="Save results JSON to path")
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with torch.inference_mode():
        def predict_fn(text: str) -> str:
            return generate_correction(model, tokenizer, text, num_beams=args.num_beams)

        logger.info("Loading test data: %s", args.test_data)
        test_data = load_test_data(args.test_data)
        logger.info("Running assessment on %d examples...", len(test_data))

        output = run_assessment(predict_fn, test_data, multi_ref=args.multi_ref)

    print_metrics(output["metrics"], model_name=args.model)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output["metrics"], f, indent=2, ensure_ascii=False)
        logger.info("Metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
