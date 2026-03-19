"""Evaluate model on kz-transformers/kk-socio-cultural-bench-mc (multiple choice).

Scoring: compare logits of A/B/C/D tokens at the last position of the prompt.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


CHOICES = ["A", "B", "C", "D"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stukenov/sozkz-core-llama-150m-kk-instruct-v2")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer (default: same as model)")
    parser.add_argument("--prompt-format", default="auto", choices=["alpaca", "chatml", "auto"],
                        help="Prompt format for evaluation")
    parser.add_argument("--output", default="results/eval_mc_bench.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0=all)")
    args = parser.parse_args()

    tok_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    # Detect prompt format
    fmt = args.prompt_format
    if fmt == "auto":
        vocab = set(tokenizer.get_vocab().keys())
        fmt = "chatml" if "<|user|>" in vocab else "alpaca"
    print(f"Using prompt format: {fmt}")

    # Get token IDs for A, B, C, D
    choice_token_ids = {}
    for c in CHOICES:
        ids = tokenizer.encode(c, add_special_tokens=False)
        choice_token_ids[c] = ids[0]

    ds = load_dataset("kz-transformers/kk-socio-cultural-bench-mc", split="train")
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in tqdm(ds, desc="Evaluating"):
        options = "\n".join(f"{c}) {row[c]}" for c in CHOICES)
        if fmt == "chatml":
            prompt = (
                "<|user|>\n"
                + row['question'] + "\n\n"
                + options + "\n\n"
                + "Дұрыс жауапты таңдаңыз. Тек A, B, C немесе D әрпін жазыңыз.\n"
                + "<|end|>\n<|assistant|>\n"
            )
        else:
            prompt = "### Нұсқаулық:\n" + row['question'] + "\n\n" + options + "\n\n### Жауап:\n"
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]

        # Pick choice with highest logit
        scores = {c: logits[choice_token_ids[c]].item() for c in CHOICES}
        pred = max(scores, key=scores.get)
        gold = row["answer"]
        cat = row["category"]
        hit = pred == gold
        correct += hit
        total += 1
        by_category[cat]["total"] += 1
        by_category[cat]["correct"] += hit

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Random baseline: 25.0%\n")

    cat_results = {}
    for cat in sorted(by_category):
        c = by_category[cat]["correct"]
        t = by_category[cat]["total"]
        acc = c / t if t > 0 else 0
        cat_results[cat] = {"correct": c, "total": t, "accuracy": round(acc, 4)}
        print(f"  {cat}: {c}/{t} = {acc*100:.1f}%")

    results = {
        "model": args.model,
        "benchmark": "kz-transformers/kk-socio-cultural-bench-mc",
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "categories": cat_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
