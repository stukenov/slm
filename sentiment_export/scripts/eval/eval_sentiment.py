"""Evaluate model on sentiment classification using KazSAnDRA dataset.

Scoring: full answer text likelihood via score_choices() from eval_mc_bench.
Labels: positive/negative/neutral mapped to Kazakh words.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm

# Allow imports from scripts/eval/ when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import MODEL_REGISTRY, load_model, get_model_short_name
from eval_mc_bench import score_choices, score_choices_api

SENTIMENT_LABELS = {
    "positive": "\u043f\u043e\u0437\u0438\u0442\u0438\u0432\u0442\u0456",
    "negative": "\u043d\u0435\u0433\u0430\u0442\u0438\u0432\u0442\u0456",
    "neutral": "\u0431\u0435\u0439\u0442\u0430\u0440\u0430\u043f",
}


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment evaluation on KazSAnDRA dataset"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model registry key (e.g. sozkz-50m) or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: paper/results/sentiment/{model_short}.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of rows (0=all)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Force 4-bit quantization (overrides registry default)",
    )
    args = parser.parse_args()

    # Resolve model
    if args.model in MODEL_REGISTRY:
        model_key = args.model
        model_short = get_model_short_name(model_key)
        entry = MODEL_REGISTRY[model_key]

        if entry["type"] == "api":
            model, tokenizer = None, None
            api_url = entry["api_url"]
            is_api = True
        else:
            model, tokenizer = load_model(model_key)
            is_api = False
            device = next(model.parameters()).device
    else:
        # Direct HuggingFace ID
        model_short = args.model.split("/")[-1]
        model_key = model_short
        is_api = False
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": device_str}
        if args.quantize:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            del load_kwargs["device_map"]

        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = next(model.parameters()).device

    # Output path
    output_path = args.output or f"paper/results/sentiment/{model_short}.json"

    # Load dataset
    try:
        ds = load_dataset("issai/kazsandra", split="test")
    except ValueError:
        # No test split -- use last 20% of train
        full = load_dataset("issai/kazsandra", split="train")
        n = len(full)
        start = int(n * 0.8)
        ds = full.select(range(start, n))

    # Validate columns
    # KazSAnDRA uses "label" column instead of "sentiment"
    if "label" in ds.column_names and "sentiment" not in ds.column_names:
        ds = ds.rename_column("label", "sentiment")
    required_cols = {"text", "sentiment"}
    actual_cols = set(ds.column_names)
    if not required_cols.issubset(actual_cols):
        print(f"ERROR: Expected columns {required_cols}, got {actual_cols}")
        sys.exit(1)

    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct = 0
    total = 0
    per_label = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in tqdm(ds, desc=f"Sentiment ({model_short})"):
        prompt = f'\u041c\u04d9\u0442\u0456\u043d: "{row["text"]}"\n\n\u0411\u04b1\u043b \u043c\u04d9\u0442\u0456\u043d\u043d\u0456\u04a3 \u0442\u043e\u043d\u0430\u043b\u044c\u0434\u0456\u043b\u0456\u0433\u0456: '
        choices = SENTIMENT_LABELS

        if is_api:
            pred, _ = score_choices_api(prompt, choices, api_url)
        else:
            pred, _ = score_choices(
                model, tokenizer, prompt, choices, device=str(device)
            )

        gold = row["sentiment"]
        hit = pred == gold
        correct += int(hit)
        total += 1
        per_label[gold]["total"] += 1
        per_label[gold]["correct"] += int(hit)

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall: {correct}/{total} = {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Random baseline: 33.3%\n")

    label_results = {}
    for label in sorted(per_label):
        c = per_label[label]["correct"]
        t = per_label[label]["total"]
        acc = c / t if t > 0 else 0
        label_results[label] = {"correct": c, "total": t, "accuracy": round(acc, 4)}
        print(f"  {label}: {c}/{t} = {acc * 100:.1f}%")

    results = {
        "model": args.model,
        "model_short": model_short,
        "task": "sentiment",
        "dataset": "issai/kazsandra",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "total": total,
            "correct": correct,
        },
        "per_label": label_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
