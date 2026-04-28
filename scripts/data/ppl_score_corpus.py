#!/usr/bin/env python3
"""Score sozkz-corpus-clean-v3 texts by perplexity using Qwen 500M.

Computes per-text PPL, adds as column, pushes scored dataset to HF.
Saves intermediate parquet shards for crash recovery.

Usage:
    # Sample mode: score 10K texts, show distribution
    python scripts/data/ppl_score_corpus.py --sample 10000

    # Full scoring: score all 13.7M texts, push to HF
    python scripts/data/ppl_score_corpus.py --run \
        --output stukenov/sozkz-corpus-scored-kk-v1

    # Resume from last shard (after crash)
    python scripts/data/ppl_score_corpus.py --run --resume \
        --output stukenov/sozkz-corpus-scored-kk-v1
"""
from __future__ import annotations

import argparse
import logging
import math
import queue as Q
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_REPO = "stukenov/sozkz-core-qwen-500m-kk-base-v1"
TOKENIZER_REPO = "stukenov/sozkz-morphbpe-100k-kk-v1"
MAX_LENGTH = 256          # 256 tokens covers ~80% of texts; 4x less HBM traffic than 512
BATCH_SIZE = 256          # RTX 5090 32GB; with MAX_LENGTH=256 logits=[256,255,100K]=12GB
SHARD_SIZE = 50_000
SHARD_DIR = "ppl_shards"


def load_model(device="cuda"):
    """Load Qwen 500M model and morphbpe-100k tokenizer."""
    logger.info("Loading tokenizer: %s", TOKENIZER_REPO)
    tok_file = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)

    logger.info("Loading model: %s", MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded: %.1fM params on %s", params_m, device)
    return model, tokenizer


@torch.no_grad()
def compute_ppl_batch(texts, model, tokenizer, device, max_length=MAX_LENGTH):
    """Compute per-text perplexity. Processes texts one-by-one through model
    but in a single tokenizer call. Avoids OOM from large logits tensors."""
    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_length, padding=True,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # With MAX_LENGTH=256: logits=[BATCH,255,100K]=12GB, safe on 32GB RTX 5090
    # MICRO=BATCH_SIZE → single forward pass per call, no inner loop overhead
    MICRO = len(texts)
    ppls = []
    for start in range(0, len(texts), MICRO):
        end = min(start + MICRO, len(texts))
        mb_ids = input_ids[start:end]
        mb_mask = attention_mask[start:end]

        outputs = model(mb_ids, attention_mask=mb_mask, use_cache=False)
        # Slice before del outputs to avoid holding two full logit tensors at once
        # (logits [MICRO,512,100K] peak = 12GB; freeing outputs first halves that)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        del outputs
        shift_labels = mb_ids[:, 1:].contiguous()
        shift_mask = mb_mask[:, 1:].contiguous().float()

        loss_per_token = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())

        masked_loss = (loss_per_token * shift_mask).sum(dim=1)
        lengths = shift_mask.sum(dim=1)
        mean_loss = masked_loss / lengths.clamp(min=1)
        mean_loss = mean_loss.clamp(max=20)

        # Single GPU->CPU transfer instead of 128+ individual .item() syncs
        ppl_vals = mean_loss.exp().cpu().tolist()
        len_vals = lengths.cpu().tolist()
        ppls.extend(float("inf") if ln < 1 else pv for pv, ln in zip(ppl_vals, len_vals))

        del shift_logits, loss_per_token
    return ppls


@torch.no_grad()
def score_tokens(input_ids, attention_mask, model, device):
    """GPU scoring from pre-tokenized CPU tensors. Tokenization happens in prefetch thread."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    del outputs
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())
    del shift_logits

    masked_loss = (loss_per_token * shift_mask).sum(dim=1)
    lengths = shift_mask.sum(dim=1)
    mean_loss = masked_loss / lengths.clamp(min=1)
    mean_loss = mean_loss.clamp(max=20)

    ppl_vals = mean_loss.exp().cpu().tolist()
    len_vals = lengths.cpu().tolist()
    del loss_per_token
    return [float("inf") if ln < 1 else pv for pv, ln in zip(ppl_vals, len_vals)]


def get_completed_shards(shard_dir):
    """Find which shard indices are already written."""
    p = Path(shard_dir)
    if not p.exists():
        return set()
    completed = set()
    for f in p.glob("shard_*.parquet"):
        try:
            idx = int(f.stem.split("_")[1])
            completed.add(idx)
        except (IndexError, ValueError):
            pass
    return completed


def save_shard(texts, sources, ppls, shard_idx, shard_dir):
    """Save a scored shard as parquet."""
    Path(shard_dir).mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "text": pa.array(texts, type=pa.string()),
        "source": pa.array(sources, type=pa.string()),
        "ppl": pa.array(ppls, type=pa.float32()),
    })
    out_path = Path(shard_dir) / ("shard_%05d.parquet" % shard_idx)
    pq.write_table(table, out_path)
    logger.info("Saved shard %d: %d rows -> %s", shard_idx, len(texts), out_path)


def run_sample(n_samples, device="cuda"):
    """Score a sample and print distribution."""
    model, tokenizer = load_model(device)

    logger.info("Loading dataset (streaming)...")
    ds = load_dataset("saken-tukenov/sozkz-corpus-clean-v3", split="train", streaming=True)

    texts, sources = [], []
    for row in ds:
        if len(texts) >= n_samples:
            break
        t = row.get("text", "")
        if t and len(t.strip()) > 20:
            texts.append(t)
            sources.append(row.get("source", "unknown"))

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
    ppls_arr = np.array(all_ppls)
    ppls_finite = ppls_arr[np.isfinite(ppls_arr)]

    print()
    sep = "=" * 70
    print(sep)
    print("PPL DISTRIBUTION (Qwen 500M scorer, %d texts, %.0fs)" % (len(ppls_finite), elapsed))
    print(sep)
    print("  Mean:   %.1f" % np.mean(ppls_finite))
    print("  Median: %.1f" % np.median(ppls_finite))
    print("  Std:    %.1f" % np.std(ppls_finite))
    print()
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print("  P%02d:    %.1f" % (p, np.percentile(ppls_finite, p)))

    print()
    print("PPL buckets:")
    for lo, hi in [(0, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, float("inf"))]:
        count = int(np.sum((ppls_finite >= lo) & (ppls_finite < hi)))
        pct = 100 * count / len(ppls_finite)
        hi_s = str(hi) if hi != float("inf") else "inf"
        print("  %d-%5s: %6d (%.1f%%)" % (lo, hi_s, count, pct))

    sorted_idx = np.argsort(ppls_arr)
    for label, indices in [("BEST (lowest PPL)", sorted_idx[:5]),
                           ("WORST (highest PPL)", sorted_idx[-5:]),
                           ("AROUND P90", sorted_idx[int(len(sorted_idx) * 0.89):int(len(sorted_idx) * 0.91)][:5])]:
        print()
        print("--- %s ---" % label)
        for idx in indices:
            snip = texts[idx][:200].replace("\n", " ")
            print("  [PPL=%.1f] [%s] %s" % (ppls_arr[idx], sources[idx], snip))
            print()


def _score_worker(gpu_id, start_idx, end_idx, ds, shard_dir):
    """Worker function: score a slice of the dataset on a specific GPU.

    Uses a prefetch thread to tokenize the next batch while GPU computes the current one,
    hiding tokenization latency (morphBPE 100K vocab is the main CPU bottleneck).
    """
    device = "cuda:%d" % gpu_id
    model, tokenizer = load_model(device)
    worker_tag = "[GPU%d]" % gpu_id

    # Pre-load slice once into Python lists to avoid repeated slow Arrow random-access
    logger.info("  %s Pre-loading %d rows into memory...", worker_tag, end_idx - start_idx)
    t_load = time.time()
    worker_slice = ds[start_idx:end_idx]
    all_texts = worker_slice["text"]
    all_sources = worker_slice["source"]
    logger.info("  %s Pre-load done in %.0fs", worker_tag, time.time() - t_load)

    total_worker = end_idx - start_idx
    batch_starts = list(range(0, total_worker, BATCH_SIZE))

    # 4 parallel tokenizer threads, each owns every 4th batch (round-robin).
    # morphBPE 100K tokenizer is 4x slower than GPU → 4 threads fully hide latency.
    # Each thread writes to its own queue; consumer cycles queues in order → preserves batch order.
    NUM_TOK = 4
    tok_qs = [Q.Queue(maxsize=4) for _ in range(NUM_TOK)]

    def make_tok_thread(tid):
        def _run():
            for idx in range(tid, len(batch_starts), NUM_TOK):
                s = batch_starts[idx]
                e = min(s + BATCH_SIZE, total_worker)
                texts = [t if t and len(t.strip()) > 0 else " " for t in all_texts[s:e]]
                enc = tokenizer(texts, return_tensors="pt", truncation=True,
                                max_length=MAX_LENGTH, padding=True)
                tok_qs[tid].put((s, enc["input_ids"], enc["attention_mask"]))
            tok_qs[tid].put(None)  # sentinel
        return _run

    for tid in range(NUM_TOK):
        threading.Thread(target=make_tok_thread(tid), daemon=True).start()

    shard_texts, shard_sources, shard_ppls = [], [], []
    scored = 0
    t0 = time.time()
    q_idx = 0
    done = [False] * NUM_TOK

    while not all(done):
        if done[q_idx]:
            q_idx = (q_idx + 1) % NUM_TOK
            continue
        item = tok_qs[q_idx].get()
        q_idx = (q_idx + 1) % NUM_TOK
        if item is None:
            done[q_idx - 1] = True
            continue
        s, input_ids, attention_mask = item
        e = min(s + BATCH_SIZE, total_worker)
        batch_texts = all_texts[s:e]
        batch_sources = all_sources[s:e]

        ppls = score_tokens(input_ids, attention_mask, model, device)
        shard_texts.extend(batch_texts)
        shard_sources.extend(batch_sources)
        shard_ppls.extend(ppls)
        scored += len(ppls)

        if len(shard_texts) >= SHARD_SIZE:
            shard_idx = (start_idx + s) // SHARD_SIZE
            save_shard(shard_texts[:SHARD_SIZE], shard_sources[:SHARD_SIZE],
                       shard_ppls[:SHARD_SIZE], shard_idx, shard_dir)
            shard_texts = shard_texts[SHARD_SIZE:]
            shard_sources = shard_sources[SHARD_SIZE:]
            shard_ppls = shard_ppls[SHARD_SIZE:]

        if scored % 10000 < BATCH_SIZE:
            elapsed = time.time() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            eta_h = (total_worker - scored) / rate / 3600 if rate > 0 else 0
            logger.info("  %s %d/%d (%.0f/s, ETA %.1fh)",
                        worker_tag, scored, total_worker, rate, eta_h)

    # Save remaining
    if shard_texts:
        shard_idx = end_idx // SHARD_SIZE + gpu_id
        save_shard(shard_texts, shard_sources, shard_ppls, shard_idx, shard_dir)

    elapsed = time.time() - t0
    logger.info("  %s DONE: %d texts in %.0fs (%.0f/s)",
                worker_tag, scored, elapsed, scored / elapsed if elapsed > 0 else 0)


def run_full(output_repo, resume=False, device="cuda"):
    """Score all texts using all available GPUs, save shards, push to HF."""
    import multiprocessing as mp

    num_gpus = torch.cuda.device_count()
    logger.info("GPUs available: %d", num_gpus)

    logger.info("Loading full dataset...")
    ds = load_dataset("saken-tukenov/sozkz-corpus-clean-v3", split="train", num_proc=8)
    total = len(ds)
    logger.info("Dataset size: %d, splitting across %d GPUs", total, num_gpus)

    if resume:
        completed = get_completed_shards(SHARD_DIR)
        if completed:
            logger.info("Resume: %d shards already done", len(completed))

    # Split dataset evenly across GPUs
    chunk_size = (total + num_gpus - 1) // num_gpus
    processes = []
    for gpu_id in range(num_gpus):
        start = gpu_id * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            break
        logger.info("  GPU %d: rows %d-%d (%d texts)", gpu_id, start, end, end - start)
        p = mp.Process(target=_score_worker, args=(gpu_id, start, end, ds, SHARD_DIR))
        processes.append(p)

    t0 = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    elapsed = time.time() - t0
    logger.info("All GPUs done in %.0fs (%.1fh)", elapsed, elapsed / 3600)

    # Merge shards to get PPL scores in order
    logger.info("Merging shards...")
    tables = []
    for f in sorted(Path(SHARD_DIR).glob("shard_*.parquet")):
        tables.append(pq.read_table(f))
    merged = pa.concat_tables(tables)
    ppl_values = merged.column("ppl").to_pylist()
    logger.info("Merged %d PPL scores", len(ppl_values))

    # Add ppl column to original dataset and push back
    logger.info("Loading original dataset to add ppl column...")
    original = load_dataset("saken-tukenov/sozkz-corpus-clean-v3", split="train", num_proc=8)
    if len(ppl_values) != len(original):
        logger.warning("PPL count (%d) != dataset size (%d). Using min.",
                        len(ppl_values), len(original))
        ppl_values = ppl_values[:len(original)]
    original = original.add_column("ppl", ppl_values)
    logger.info("Pushing updated dataset with ppl column to %s...", output_repo)
    original.push_to_hub(output_repo, private=False)
    logger.info("Done! %d texts with ppl column uploaded.", len(original))


def main():
    parser = argparse.ArgumentParser(description="Score corpus by PPL using Qwen 500M")
    parser.add_argument("--sample", type=int, default=0, help="Score N texts and show distribution")
    parser.add_argument("--run", action="store_true", help="Full scoring + push to HF")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved shard")
    parser.add_argument("--output", default="stukenov/sozkz-corpus-scored-kk-v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.sample > 0:
        run_sample(args.sample, args.device)
    if args.run:
        run_full(args.output, args.resume, args.device)
    if not args.sample and not args.run:
        parser.print_help()


if __name__ == "__main__":
    main()
