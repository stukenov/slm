"""Evaluate model on Belebele reading comprehension (kaz_Cyrl).

Scoring: full answer text likelihood via score_choices() from eval_mc_bench.
4-choice reading comprehension with passage, question, and answer options.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm

# Allow imports from scripts/eval/ when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import MODEL_REGISTRY, load_model, get_model_short_name
from eval_mc_bench import score_choices, score_choices_api


def main():
    parser = argparse.ArgumentParser(
        description="Belebele reading comprehension evaluation (kaz_Cyrl)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model registry key (e.g. sozkz-50m) or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: paper/results/belebele/{model_short}.json)",
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
    output_path = args.output or f"paper/results/belebele/{model_short}.json"

    # Load dataset
    try:
        ds = load_dataset("facebook/belebele", "kaz_Cyrl", split="test")
    except ValueError:
        ds = load_dataset("facebook/belebele", "kaz_Cyrl", split="train")

    # Validate columns
    required_cols = {
        "flores_passage",
        "question",
        "mc_answer1",
        "mc_answer2",
        "mc_answer3",
        "mc_answer4",
        "correct_answer_num",
    }
    actual_cols = set(ds.column_names)
    if not required_cols.issubset(actual_cols):
        print(f"ERROR: Expected columns {required_cols}, got {actual_cols}")
        sys.exit(1)

    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct = 0
    total = 0

    for row in tqdm(ds, desc=f"Belebele ({model_short})"):
        prompt = (
            f'{row["flores_passage"]}\n\n'
            f'\u0421\u04b1\u0440\u0430\u049b: {row["question"]}\n\n'
            f'\u0416\u0430\u0443\u0430\u0431\u044b: '
        )
        choices = {
            "1": row["mc_answer1"],
            "2": row["mc_answer2"],
            "3": row["mc_answer3"],
            "4": row["mc_answer4"],
        }
        gold = str(row["correct_answer_num"])

        if is_api:
            pred, _ = score_choices_api(prompt, choices, api_url)
        else:
            pred, _ = score_choices(
                model, tokenizer, prompt, choices, device=str(device)
            )

        hit = pred == gold
        correct += int(hit)
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall: {correct}/{total} = {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Random baseline: 25.0%\n")

    results = {
        "model": args.model,
        "model_short": model_short,
        "task": "belebele",
        "dataset": "facebook/belebele",
        "config": "kaz_Cyrl",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "total": total,
            "correct": correct,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
