#!/usr/bin/env python3
"""Score dataset texts by perplexity using a pretrained model.

Uses the 150M Kazakh model to compute PPL for each text.
High PPL = garbage/foreign/broken text. Low PPL = clean Kazakh.

Usage:
    # Score and show distribution (dry run)
    python scripts/data/ppl_score_dataset.py --dataset kk --sample 10000

    # Full scoring + filter + push
    python scripts/data/ppl_score_dataset.py --dataset kk --run \
        --percentile 90 --output stukenov/sozkz-corpus-clean-kk-text-v4
"""
from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "stukenov/sozkz-core-llama-150m-kk-base-v1"
MAX_LENGTH = 512
BATCH_SIZE = 128


@torch.no_grad()
def compute_ppl_batch(texts, model, tokenizer, device, max_length=MAX_LENGTH):
    """Compute perplexity for a batch of texts."""
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    ppls = []
    for i in range(len(texts)):
        mask = attention_mask[i]
        length = mask.sum().item()
        if length <= 1:
            ppls.append(float("inf"))
            continue

        shift_logits = logits[i, :length - 1, :]
        shift_labels = input_ids[i, 1:length]

        loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="mean"
        )
        ppl = math.exp(min(loss.item(), 20))
        ppls.append(ppl)

    return ppls


def load_model(device="cuda"):
    logger.info("Loading model: %s", MODEL_NAME)
    # Use PreTrainedTokenizerFast directly (AutoTokenizer broken in transformers 4.57+)
    from huggingface_hub import hf_hub_download
    tok_file = hf_hub_download(MODEL_NAME, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    # Use token 1 as pad (same as model config pad_token_id=1)
    tokenizer.pad_token_id = 1
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    model = torch.compile(model)
    # Warmup compile
    dummy = torch.zeros(1, 16, dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(dummy)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded + compiled: %.1fM params on %s", params_m, device)
    return model, tokenizer


def sample_and_score(dataset_name, text_column, n_samples, device="cuda"):
    """Score a sample of texts and show PPL distribution."""
    model, tokenizer = load_model(device)

    logger.info("Loading dataset: %s (streaming)", dataset_name)
    ds = load_dataset(dataset_name, split="train", streaming=True)

    texts = []
    for row in ds:
        if len(texts) >= n_samples:
            break
        text = row.get(text_column, "")
        if text and len(text.strip()) > 20:
            texts.append(text)

    logger.info("Scoring %d texts...", len(texts))
    all_ppls = []
    t0 = time.time()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        ppls = compute_ppl_batch(batch, model, tokenizer, device)
        all_ppls.extend(ppls)
        if (i // BATCH_SIZE + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = len(all_ppls) / elapsed
            logger.info("  %d/%d scored (%.0f texts/s)", len(all_ppls), len(texts), rate)

    elapsed = time.time() - t0
    logger.info("Scored %d texts in %.0fs (%.0f texts/s)", len(all_ppls), elapsed, len(all_ppls) / elapsed)

    ppls_arr = np.array(all_ppls)
    ppls_finite = ppls_arr[np.isfinite(ppls_arr)]

    print()
    sep = "=" * 70
    print(sep)
    print("PPL DISTRIBUTION: %s (%d texts)" % (dataset_name, len(ppls_finite)))
    print(sep)
    print("  Mean:   %.1f" % np.mean(ppls_finite))
    print("  Median: %.1f" % np.median(ppls_finite))
    print("  Std:    %.1f" % np.std(ppls_finite))
    print()
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(ppls_finite, p)
        print("  P%02d:    %.1f" % (p, val))

    print()
    print("PPL buckets:")
    for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, float("inf"))]:
        count = int(np.sum((ppls_finite >= lo) & (ppls_finite < hi)))
        pct = 100 * count / len(ppls_finite)
        hi_str = str(hi) if hi != float("inf") else "inf"
        print("  %10s: %6d (%.1f%%)" % ("%d-%s" % (lo, hi_str), count, pct))

    sorted_indices = np.argsort(ppls_arr)

    print()
    print(sep)
    print("BEST (lowest PPL) - cleanest texts:")
    print(sep)
    for idx in sorted_indices[:5]:
        snip = texts[idx][:200].replace("\n", "\\n")
        print("  [PPL=%.1f] %s" % (ppls_arr[idx], snip))
        print()

    print(sep)
    print("WORST (highest PPL) - dirtiest texts:")
    print(sep)
    for idx in sorted_indices[-5:]:
        snip = texts[idx][:200].replace("\n", "\\n")
        print("  [PPL=%.1f] %s" % (ppls_arr[idx], snip))
        print()

    print(sep)
    print("AROUND P90 - borderline texts:")
    print(sep)
    p90_val = np.percentile(ppls_finite, 90)
    near_p90 = [(i, ppls_arr[i]) for i in range(len(ppls_arr))
                if abs(ppls_arr[i] - p90_val) < p90_val * 0.05]
    for idx, ppl in near_p90[:5]:
        snip = texts[idx][:200].replace("\n", "\\n")
        print("  [PPL=%.1f] %s" % (ppl, snip))
        print()


def run_full_filter(dataset_name, text_column, output_repo, percentile, device="cuda"):
    """Score all texts, filter by PPL percentile, push to HF."""
    model, tokenizer = load_model(device)

    logger.info("Loading full dataset: %s", dataset_name)
    ds = load_dataset(dataset_name, verification_mode="no_checks")
    if "train" in ds:
        data = ds["train"]
    else:
        data = next(iter(ds.values()))

    logger.info("Dataset size: %d", len(data))
    logger.info("Scoring all texts (batch_size=%d)...", BATCH_SIZE)
    all_ppls = []
    t0 = time.time()

    for i in range(0, len(data), BATCH_SIZE):
        batch_texts = [data[j][text_column] or "" for j in range(i, min(i + BATCH_SIZE, len(data)))]
        ppls = compute_ppl_batch(batch_texts, model, tokenizer, device)
        all_ppls.extend(ppls)
        if (i // BATCH_SIZE + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = len(all_ppls) / elapsed
            eta = (len(data) - len(all_ppls)) / rate / 3600
            logger.info("  %d/%d (%.0f/s, ETA %.1fh)", len(all_ppls), len(data), rate, eta)

    elapsed = time.time() - t0
    logger.info("Scoring done: %d texts in %.0fs", len(all_ppls), elapsed)

    ppls_arr = np.array(all_ppls)
    threshold = float(np.percentile(ppls_arr[np.isfinite(ppls_arr)], percentile))
    logger.info("PPL threshold (P%d): %.1f", percentile, threshold)

    keep_mask = ppls_arr <= threshold
    keep_count = int(keep_mask.sum())
    logger.info("Keeping %d / %d texts (%.1f%%)", keep_count, len(data), 100 * keep_count / len(data))

    keep_indices = np.where(keep_mask)[0].tolist()
    filtered = data.select(keep_indices)

    logger.info("Pushing to %s...", output_repo)
    filtered.push_to_hub(output_repo, private=False)
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kk", "enkk"], required=True)
    parser.add_argument("--dataset-name", default=None, help="Override HF dataset name (use filtered version)")
    parser.add_argument("--sample", type=int, default=0, help="Score N texts and show distribution")
    parser.add_argument("--run", action="store_true", help="Full score + filter + push")
    parser.add_argument("--percentile", type=int, default=90, help="Keep texts below this PPL percentile")
    parser.add_argument("--output", default=None, help="Output HF repo")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    KK = ("saken-tukenov/sozkz-corpus-clean-kk-text-v2", "text")
    ENKK = ("saken-tukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1", "text_kk")

    ds_name, text_col = KK if args.dataset == "kk" else ENKK
    if args.dataset_name:
        ds_name = args.dataset_name

    if args.sample > 0:
        sample_and_score(ds_name, text_col, args.sample, args.device)

    if args.run:
        if not args.output:
            print("--output required with --run")
            return
        run_full_filter(ds_name, text_col, args.output, args.percentile, args.device)


if __name__ == "__main__":
    main()
