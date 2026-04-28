#!/usr/bin/env python3
"""exp030: Evaluate GEC 1B models — LoRA or full fine-tune.

Usage:
    python3 exp030_eval.py --model_dir /root/exp030_030a/final --exp_id 030a
    python3 exp030_eval.py --model_dir /root/exp030_030d/final --exp_id 030d --method full
    python3 exp030_eval.py --hf_model stukenov/sozkz-core-llama-1b-kk-gec-v1 --exp_id final

Metrics: Exact Match, CER, Word F0.5, Identity Preservation
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download


HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ── Data loading (test split) ───────────────────────────────────────────────

def word_edit_distance(a, b):
    wa, wb = a.split(), b.split()
    if len(wa) < len(wb):
        return word_edit_distance(b, a)
    if not wb:
        return len(wa)
    prev = list(range(len(wb) + 1))
    for i, ca in enumerate(wa):
        curr = [i + 1]
        for j, cb in enumerate(wb):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def load_test_data(max_examples=500):
    """Load test examples: 50% with errors, 50% clean (identity)."""
    REPO = "stukenov/sozkz-corpus-synthetic-kk-gec-v1"
    FILES = [
        "data/grammar_balanced_v2/train.jsonl",
        "data/grammar_v2/train.jsonl",
    ]

    errors = []
    for fname in FILES:
        try:
            local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset", token=HF_TOKEN)
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    inp, tgt = ex.get("input", ""), ex.get("target", "")
                    et = ex.get("error_type", "unknown")
                    if not inp or not tgt or et in ("unknown", "clean"):
                        continue
                    if word_edit_distance(inp, tgt) > 2:
                        continue
                    if inp.strip() != tgt.strip():
                        errors.append({"input": inp, "target": tgt, "is_clean": False})
        except Exception:
            pass

    import random
    random.seed(12345)  # Different seed from training
    random.shuffle(errors)

    n_err = min(max_examples // 2, len(errors))
    test = errors[:n_err]

    # Add clean (identity) examples
    for e in errors[n_err : n_err + n_err]:
        test.append({"input": e["target"], "target": e["target"], "is_clean": True})

    random.shuffle(test)
    print(f"Test set: {len(test)} examples ({n_err} errors, {len(test) - n_err} clean)")
    return test


# ── Metrics ──────────────────────────────────────────────────────────────────

def char_error_rate(pred, ref):
    if not ref:
        return 0.0 if not pred else 1.0
    m, n = len(pred), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / n


def word_edits(source, text):
    """LCS-based word-level edit extraction."""
    src, tgt = source.split(), text.split()
    m, n = len(src), len(tgt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src[i - 1] == tgt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    edits = set()
    i, j = m, n
    while i > 0 and j > 0:
        if src[i - 1] == tgt[j - 1]:
            i -= 1; j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            edits.add(("del", i - 1, src[i - 1])); i -= 1
        else:
            edits.add(("ins", j - 1, tgt[j - 1])); j -= 1
    while i > 0:
        edits.add(("del", i - 1, src[i - 1])); i -= 1
    while j > 0:
        edits.add(("ins", j - 1, tgt[j - 1])); j -= 1
    return edits


# ── Generation + Evaluation ─────────────────────────────────────────────────

def run_evaluation(model, tokenizer, test_data, device):
    exact_match = 0
    total_cer = 0.0
    identity_total = identity_ok = 0
    tp = fp = fn = 0
    results = []

    for i, ex in enumerate(test_data):
        prompt = ex["input"] + "\n"
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        pred = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
        target = ex["target"].strip()
        source = ex["input"].strip()

        is_exact = pred == target
        if is_exact:
            exact_match += 1

        total_cer += char_error_rate(pred, target)

        if ex.get("is_clean", False):
            identity_total += 1
            if pred == source:
                identity_ok += 1

        gold_edits = word_edits(source, target)
        pred_edits = word_edits(source, pred)
        tp += len(gold_edits & pred_edits)
        fp += len(pred_edits - gold_edits)
        fn += len(gold_edits - pred_edits)

        results.append({
            "input": source, "target": target, "prediction": pred,
            "exact": is_exact, "is_clean": ex.get("is_clean", False),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(test_data)}] EM so far: {exact_match}/{i + 1}")

    n = len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta = 0.5
    f05 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)
           if (precision + recall) > 0 else 0.0)

    metrics = {
        "n_examples": n,
        "exact_match": round(exact_match / n * 100, 1),
        "cer": round(total_cer / n, 4),
        "word_precision": round(precision, 3),
        "word_recall": round(recall, 3),
        "word_f05": round(f05, 3),
        "identity_total": identity_total,
        "identity_preserved": round(identity_ok / identity_total * 100, 1) if identity_total > 0 else None,
    }
    return metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Local checkpoint directory")
    parser.add_argument("--hf_model", type=str, help="HF model ID (alternative to model_dir)")
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "lora", "full"])
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="/root/exp030_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.hf_model or args.model_dir

    # Detect method
    method = args.method
    if method == "auto":
        adapter_config = os.path.join(args.model_dir, "adapter_config.json") if args.model_dir else None
        if adapter_config and os.path.exists(adapter_config):
            method = "lora"
        else:
            method = "full"

    print(f"Loading model: {model_path} (method={method})")

    if method == "lora":
        from peft import PeftModel
        # Load base model
        results_file = os.path.join(args.model_dir, "results.json")
        base = "stukenov/sozkz-core-llama-1b-kk-base-v1"
        if os.path.exists(results_file):
            with open(results_file) as f:
                res = json.load(f)
            base = res.get("base_model", base)

        tokenizer = AutoTokenizer.from_pretrained(base, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, token=HF_TOKEN)
        model = PeftModel.from_pretrained(model, args.model_dir)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, token=HF_TOKEN)

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resize embeddings if needed
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Load test data
    test_data = load_test_data(args.max_examples)

    # Run
    t0 = time.time()
    metrics, results = run_evaluation(model, tokenizer, test_data, device)
    eval_time = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.exp_id}")
    print(f"{'='*60}")
    print(f"  Exact Match:     {metrics['exact_match']}%")
    print(f"  CER:             {metrics['cer']}")
    print(f"  Word Precision:  {metrics['word_precision']}")
    print(f"  Word Recall:     {metrics['word_recall']}")
    print(f"  Word F0.5:       {metrics['word_f05']}")
    if metrics['identity_preserved'] is not None:
        print(f"  Identity Pres:   {metrics['identity_preserved']}%")
    print(f"  Time:            {eval_time:.0f}s")
    print(f"{'='*60}")

    # Show errors
    errs = [r for r in results if not r["exact"]][:5]
    if errs:
        print("\nSample errors:")
        for r in errs:
            print(f"  IN:   {r['input']}")
            print(f"  TGT:  {r['target']}")
            print(f"  PRED: {r['prediction']}")
            print()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"eval_{args.exp_id}.json")
    with open(out_file, "w") as f:
        json.dump({"exp_id": args.exp_id, "metrics": metrics, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
