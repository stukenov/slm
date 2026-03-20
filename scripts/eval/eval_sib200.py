"""Evaluate model on SIB-200 topic classification (kaz_Cyrl).

Scoring: full answer text likelihood via score_choices() from eval_mc_bench.
7-category topic classification with Kazakh label words.
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

TOPIC_LABELS = {
    "science/technology": "\u0493\u044b\u043b\u044b\u043c \u0436\u04d9\u043d\u0435 \u0442\u0435\u0445\u043d\u043e\u043b\u043e\u0433\u0438\u044f",
    "travel": "\u0441\u0430\u044f\u0445\u0430\u0442",
    "politics": "\u0441\u0430\u044f\u0441\u0430\u0442",
    "sports": "\u0441\u043f\u043e\u0440\u0442",
    "health": "\u0434\u0435\u043d\u0441\u0430\u0443\u043b\u044b\u049b",
    "entertainment": "\u043e\u0439\u044b\u043d-\u0441\u0430\u0443\u044b\u049b",
    "geography": "\u0433\u0435\u043e\u0433\u0440\u0430\u0444\u0438\u044f",
}


def main():
    parser = argparse.ArgumentParser(
        description="SIB-200 topic classification evaluation (kaz_Cyrl)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model registry key (e.g. sozkz-50m) or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: paper/results/sib200/{model_short}.json)",
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
    output_path = args.output or f"paper/results/sib200/{model_short}.json"

    # Load dataset
    try:
        ds = load_dataset("Davlan/sib200", "kaz_Cyrl", split="test")
    except ValueError:
        ds = load_dataset("Davlan/sib200", "kaz_Cyrl", split="validation")

    # Validate columns
    required_cols = {"text", "category"}
    actual_cols = set(ds.column_names)
    if not required_cols.issubset(actual_cols):
        print(f"ERROR: Expected columns {required_cols}, got {actual_cols}")
        sys.exit(1)

    # Check if actual category values match our mapping
    unique_cats = set(ds["category"])
    known_cats = set(TOPIC_LABELS.keys())
    unknown = unique_cats - known_cats
    if unknown:
        print(f"WARNING: Unknown categories in dataset: {unknown}")
        print(f"Known categories: {known_cats}")
        print("Results may be incomplete for unknown categories.")

    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct = 0
    total = 0
    per_topic = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in tqdm(ds, desc=f"SIB-200 ({model_short})"):
        gold = row["category"]

        # Skip samples with unknown categories
        if gold not in TOPIC_LABELS:
            continue

        prompt = (
            f'\u041c\u04d9\u0442\u0456\u043d: "{row["text"]}"\n\n'
            f'\u0411\u04b1\u043b \u043c\u04d9\u0442\u0456\u043d\u043d\u0456\u04a3 \u0442\u0430\u049b\u044b\u0440\u044b\u0431\u044b: '
        )
        choices = TOPIC_LABELS

        if is_api:
            pred, _ = score_choices_api(prompt, choices, api_url)
        else:
            pred, _ = score_choices(
                model, tokenizer, prompt, choices, device=str(device)
            )

        hit = pred == gold
        correct += int(hit)
        total += 1
        per_topic[gold]["total"] += 1
        per_topic[gold]["correct"] += int(hit)

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall: {correct}/{total} = {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Random baseline: {100 / len(TOPIC_LABELS):.1f}%\n")

    topic_results = {}
    for cat in sorted(per_topic):
        c = per_topic[cat]["correct"]
        t = per_topic[cat]["total"]
        acc = c / t if t > 0 else 0
        topic_results[cat] = {"correct": c, "total": t, "accuracy": round(acc, 4)}
        print(f"  {cat}: {c}/{t} = {acc * 100:.1f}%")

    results = {
        "model": args.model,
        "model_short": model_short,
        "task": "sib200",
        "dataset": "Davlan/sib200",
        "config": "kaz_Cyrl",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "total": total,
            "correct": correct,
        },
        "per_topic": topic_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
