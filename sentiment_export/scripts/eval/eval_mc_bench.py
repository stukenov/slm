"""Evaluate model on kz-transformers/kk-socio-cultural-bench-mc (multiple choice).

Scoring: full answer text likelihood (sum of log-probs over choice tokens,
length-normalized). Fixes the single-token logit comparison bug.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from collections import defaultdict

import requests
import torch
from datasets import load_dataset
from tqdm import tqdm

# Allow imports from scripts/eval/ when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import MODEL_REGISTRY, load_model, get_model_short_name

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

CHOICES = ["A", "B", "C", "D"]


def score_choices(
    model,
    tokenizer,
    prompt: str,
    choices: dict[str, str],
    device: str = "cuda",
) -> tuple[str | None, dict[str, float]]:
    """Score each choice by full answer text likelihood.

    For each choice, concatenates prompt + choice_text, runs a forward pass,
    and sums log-probs of ONLY the choice tokens (length-normalized).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: The question/prompt text.
        choices: Dict like {"A": "Kazakhstan", "B": "Russia", ...}.
        device: Device string.

    Returns:
        (predicted_label, scores_dict) where predicted_label has highest score.
    """
    scores: dict[str, float] = {}
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    for label, choice_text in choices.items():
        full_text = prompt + choice_text
        input_ids = tokenizer.encode(full_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=device)

        with torch.no_grad():
            logits = model(input_tensor).logits  # [1, seq_len, vocab]

        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

        # Sum log-probs of choice tokens only (positions prompt_len to end)
        n_choice_tokens = len(input_ids) - prompt_len
        if n_choice_tokens == 0:
            scores[label] = float("-inf")
            continue

        total_lp = 0.0
        for i in range(prompt_len, len(input_ids)):
            # log_probs[i-1] gives the prediction distribution for position i
            total_lp += log_probs[i - 1, input_ids[i]].item()

        # Length-normalize
        scores[label] = total_lp / n_choice_tokens

    predicted = max(scores, key=scores.get) if scores else None
    return predicted, scores


def score_choices_api(
    prompt: str,
    choices: dict[str, str],
    api_url: str = "http://localhost:8080",
) -> tuple[str | None, dict[str, float]]:
    """Score choices via GPT-OSS-120B API (llama.cpp compatible).

    Posts prompt+choice to the /completion endpoint with logprobs enabled.
    Uses the total prompt log-probability for scoring.

    Args:
        prompt: The question/prompt text.
        choices: Dict like {"A": "Kazakhstan", ...}.
        api_url: Base URL of the API server.

    Returns:
        (predicted_label, scores_dict). Returns (None, {}) on failure.
    """
    scores: dict[str, float] = {}

    for label, choice_text in choices.items():
        full_text = prompt + choice_text
        try:
            resp = requests.post(
                f"{api_url}/completion",
                json={"prompt": full_text, "n_predict": 0, "logprobs": True},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            # Extract total prompt log-probability
            prompt_logprob = data.get("prompt_logprob", None)
            if prompt_logprob is not None:
                num_bytes = len(choice_text.encode("utf-8"))
                scores[label] = prompt_logprob / max(num_bytes, 1)
            else:
                logger.warning(
                    "API did not return prompt_logprob for choice %s", label
                )
                scores[label] = float("-inf")
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning("API call failed for choice %s: %s", label, e)
            return None, {}

    predicted = max(scores, key=scores.get) if scores else None
    return predicted, scores


def main():
    parser = argparse.ArgumentParser(
        description="MC QA evaluation on kk-socio-cultural-bench-mc"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model registry key (e.g. sozkz-50m) or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: paper/results/mc_qa/{model_short}.json)",
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
            # Determine device from model
            device = next(model.parameters()).device
    else:
        # Direct HuggingFace ID
        model_short = args.model.split("/")[-1]
        model_key = model_short
        is_api = False
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device_str = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
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
    output_path = args.output or f"paper/results/mc_qa/{model_short}.json"

    # Load dataset
    ds = load_dataset("kz-transformers/kk-socio-cultural-bench-mc", split="train")
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in tqdm(ds, desc=f"MC QA ({model_short})"):
        # Build plain-text prompt (no chat template -- base model evaluation)
        options = "\n".join(f"{c}) {row[c]}" for c in CHOICES)
        prompt = row["question"] + "\n" + options + "\n\n" + "\u0416\u0430\u0443\u0430\u0431\u044b: "

        choices_dict = {c: row[c] for c in CHOICES}

        if is_api:
            pred, _ = score_choices_api(prompt, choices_dict, api_url)
        else:
            pred, _ = score_choices(model, tokenizer, prompt, choices_dict, device=str(device))

        gold = row["answer"]
        cat = row.get("category", "unknown")
        hit = pred == gold
        correct += int(hit)
        total += 1
        by_category[cat]["total"] += 1
        by_category[cat]["correct"] += int(hit)

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall: {correct}/{total} = {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Random baseline: 25.0%\n")

    cat_results = {}
    for cat in sorted(by_category):
        c = by_category[cat]["correct"]
        t = by_category[cat]["total"]
        acc = c / t if t > 0 else 0
        cat_results[cat] = {"correct": c, "total": t, "accuracy": round(acc, 4)}
        print(f"  {cat}: {c}/{t} = {acc * 100:.1f}%")

    results = {
        "model": args.model,
        "model_short": model_short,
        "task": "mc_qa",
        "dataset": "kz-transformers/kk-socio-cultural-bench-mc",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "total": total,
            "correct": correct,
        },
        "per_category": cat_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
