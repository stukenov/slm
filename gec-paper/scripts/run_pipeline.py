#!/usr/bin/env python3
"""Run dual-model GEC inference pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nllb_model", default=None, help="NLLB model path")
    parser.add_argument("--tagger_model", default=None, help="XLM-R tagger path")
    parser.add_argument("--mode", default="cascade", choices=["tagger_only", "seq2seq_only", "cascade"])
    parser.add_argument("--input_file", default=None, help="JSONL file for batch mode")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    seq2seq_fn = None
    tagger_fn = None

    if args.nllb_model:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from gecpaper.models.nllb_gec import generate_correction

        tokenizer = AutoTokenizer.from_pretrained(args.nllb_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.nllb_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        def seq2seq_fn(text):
            return generate_correction(model, tokenizer, text)

    from gecpaper.pipeline import DualPipeline
    pipe = DualPipeline(
        tagger_fn=tagger_fn,
        seq2seq_fn=seq2seq_fn,
        mode=args.mode,
    )

    if args.input_file:
        results = []
        with open(args.input_file) as f:
            for line in f:
                item = json.loads(line.strip())
                source = item["input"]
                corrected = pipe.correct(source)
                results.append({"input": source, "prediction": corrected})

        if args.output_file:
            with open(args.output_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            for r in results:
                print(json.dumps(r, ensure_ascii=False))
    else:
        print("Interactive mode. Type a Kazakh sentence (Ctrl+D to quit):")
        for line in sys.stdin:
            line = line.strip()
            if line:
                result = pipe.correct(line)
                print(f"  -> {result}")


if __name__ == "__main__":
    main()
