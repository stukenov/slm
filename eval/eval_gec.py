"""Evaluation script for Kazakh GEC model.

Metrics:
- Exact match accuracy (% of examples where output == target)
- Word-level Precision / Recall / F0.5 (standard GEC metric)
- Identity preservation (% of correct inputs left unchanged)
- Character Error Rate (CER)

Usage:
    python eval/eval_gec.py --model saken-tukenov/sozkz-fix-mt5-50m-kk-gec-v1
    python eval/eval_gec.py --model saken-tukenov/sozkz-fix-mt5-50m-kk-gec-v1 --max_examples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slm.data_gec import SEP, SRC, TASK_FIX, load_gec_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_correction(model, tokenizer, text: str, max_new_tokens: int = 256) -> str:
    prompt = f"{TASK_FIX}{SRC}{text}{SEP}"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def word_level_edits(source: str, text: str) -> set[tuple[str, str]]:
    """Extract word-level edits between source and text (simple diff)."""
    src_words = source.split()
    txt_words = text.split()
    edits = set()

    # Use simple LCS-based alignment
    m, n = len(src_words), len(txt_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_words[i - 1] == txt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find aligned pairs
    i, j = m, n
    aligned = []
    while i > 0 and j > 0:
        if src_words[i - 1] == txt_words[j - 1]:
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            edits.add(("del", i - 1, src_words[i - 1]))
            i -= 1
        else:
            edits.add(("ins", j - 1, txt_words[j - 1]))
            j -= 1
    while i > 0:
        edits.add(("del", i - 1, src_words[i - 1]))
        i -= 1
    while j > 0:
        edits.add(("ins", j - 1, txt_words[j - 1]))
        j -= 1
    return edits


def char_error_rate(prediction: str, reference: str) -> float:
    """Compute CER using edit distance."""
    if not reference:
        return 0.0 if not prediction else 1.0
    m, n = len(prediction), len(reference)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if prediction[i - 1] == reference[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / n


def evaluate(model, tokenizer, dataset, max_examples: int | None = None) -> dict:
    examples = dataset
    if max_examples and max_examples < len(dataset):
        examples = dataset.select(range(max_examples))

    exact_match = 0
    identity_total = 0
    identity_preserved = 0
    total_cer = 0.0

    # For word-level P/R/F
    tp = 0  # edits in both gold and predicted
    fp = 0  # edits in predicted but not gold
    fn = 0  # edits in gold but not predicted

    results = []

    for ex in tqdm(examples, desc="Evaluating"):
        source = ex["input"]
        target = ex["target"]
        prediction = generate_correction(model, tokenizer, source)

        is_identity = source.strip() == target.strip()
        if is_identity:
            identity_total += 1
            if prediction.strip() == source.strip():
                identity_preserved += 1

        if prediction.strip() == target.strip():
            exact_match += 1

        total_cer += char_error_rate(prediction, target)

        # Word-level edits
        gold_edits = word_level_edits(source, target)
        pred_edits = word_level_edits(source, prediction)
        tp += len(gold_edits & pred_edits)
        fp += len(pred_edits - gold_edits)
        fn += len(gold_edits - pred_edits)

        results.append({
            "input": source,
            "target": target,
            "prediction": prediction,
            "exact_match": prediction.strip() == target.strip(),
        })

    n = len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta = 0.5
    f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "n_examples": n,
        "exact_match": exact_match / n,
        "cer": total_cer / n,
        "word_precision": precision,
        "word_recall": recall,
        "word_f0.5": f05,
        "identity_total": identity_total,
        "identity_preserved": identity_preserved / identity_total if identity_total > 0 else None,
    }
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GEC model")
    parser.add_argument("--model", type=str, default="saken-tukenov/sozkz-fix-mt5-50m-kk-gec-v1")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    logger.info("Loading dataset (split=%s)...", args.split)
    ds = load_gec_dataset(identity_ratio=0.0)
    if args.split not in ds:
        logger.error("Split '%s' not found. Available: %s", args.split, list(ds.keys()))
        sys.exit(1)

    metrics, results = evaluate(model, tokenizer, ds[args.split], args.max_examples)

    print("\n" + "=" * 50)
    print("GEC Evaluation Results")
    print("=" * 50)
    print(f"  Model:              {args.model}")
    print(f"  Split:              {args.split}")
    print(f"  Examples:           {metrics['n_examples']}")
    print(f"  Exact Match:        {metrics['exact_match']:.1%}")
    print(f"  CER:                {metrics['cer']:.4f}")
    print(f"  Word Precision:     {metrics['word_precision']:.3f}")
    print(f"  Word Recall:        {metrics['word_recall']:.3f}")
    print(f"  Word F0.5:          {metrics['word_f0.5']:.3f}")
    if metrics["identity_preserved"] is not None:
        print(f"  Identity preserved: {metrics['identity_preserved']:.1%} ({metrics['identity_total']} examples)")
    print("=" * 50)

    # Show some error examples
    errors = [r for r in results if not r["exact_match"]][:10]
    if errors:
        print("\nSample errors:")
        for r in errors:
            print(f"  Input:      {r['input']}")
            print(f"  Target:     {r['target']}")
            print(f"  Prediction: {r['prediction']}")
            print()

    output_path = args.output or f"eval/gec_eval_{args.split}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
