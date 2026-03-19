#!/usr/bin/env python3
"""Translate IlyaGusev/gazeta (ru) to Kazakh using deepvk/kazRush-ru-kk.

Supports:
  - Chunking long texts (T5 max ~512 tokens)
  - Multi-GPU via --split-id / --num-splits
  - bf16 inference
  - Batched translation
  - Periodic checkpointing (parquet)
  - Benchmark mode (--benchmark)

Usage:
  # Benchmark on 100 samples
  python translate_gazeta.py --benchmark --num-samples 100 --gpu-id 0

  # Full translation on 2 GPUs
  python translate_gazeta.py --gpu-id 0 --split-id 0 --num-splits 2 --bf16 --batch-size 16 &
  python translate_gazeta.py --gpu-id 1 --split-id 1 --num-splits 2 --bf16 --batch-size 16 &

  # Merge results
  python translate_gazeta.py --merge --output-dir ./gazeta_kk

  # Upload to HuggingFace
  python translate_gazeta.py --upload --output-dir ./gazeta_kk --hf-repo saken-tukenov/sozkz-corpus-raw-kk-gazeta-v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def chunk_text(text: str, tokenizer, max_tokens: int = 400) -> list[str]:
    """Split text into chunks that fit within max_tokens.

    Strategy: accumulate sentences until adding the next would exceed the limit.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current_chunk: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
        if sent_len > max_tokens:
            # Sentence itself is too long — force-split by tokens
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            # Tokenize and split into sub-chunks
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            for i in range(0, len(tokens), max_tokens):
                sub = tokenizer.decode(tokens[i:i + max_tokens], skip_special_tokens=True)
                chunks.append(sub)
            continue

        if current_len + sent_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# Translation (optimized: batched chunks, torch.compile, SDPA)
# ---------------------------------------------------------------------------

def translate_batch(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 512,
    num_beams: int = 1,
) -> list[str]:
    """Translate a batch of texts."""
    if not texts:
        return []
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_batch_fields(
    rows: list[dict],
    model,
    tokenizer,
    device: torch.device,
    num_beams: int = 1,
    chunk_batch_size: int = 32,
) -> list[dict]:
    """Translate title, summary, text for a batch of rows.

    All fields are batched — text chunks are collected across rows
    and translated in large batches for maximum GPU utilization.
    """
    # Batch translate titles
    titles = [r.get("title", "") or "" for r in rows]
    titles_kk = translate_batch(
        titles, model, tokenizer, device, num_beams=num_beams,
    )

    # Batch translate summaries
    summaries = [r.get("summary", "") or "" for r in rows]
    summaries_kk = translate_batch(
        summaries, model, tokenizer, device, num_beams=num_beams,
    )

    # Collect ALL text chunks across rows, then batch-translate
    chunk_map = []  # (row_idx, chunk_idx, chunk_text)
    for i, r in enumerate(rows):
        txt = r.get("text", "") or ""
        if not txt.strip():
            chunk_map.append((i, 0, ""))
            continue
        chunks = chunk_text(txt, tokenizer, max_tokens=400)
        for j, c in enumerate(chunks):
            chunk_map.append((i, j, c))

    # Translate chunks in large batches
    all_chunk_texts = [c[2] for c in chunk_map]
    all_translated = []
    for b_start in range(0, len(all_chunk_texts), chunk_batch_size):
        batch = all_chunk_texts[b_start:b_start + chunk_batch_size]
        # Skip empty-only batches
        non_empty = [t for t in batch if t.strip()]
        if not non_empty:
            all_translated.extend([""] * len(batch))
            continue
        translated = translate_batch(
            batch, model, tokenizer, device, num_beams=num_beams,
        )
        all_translated.extend(translated)

    # Reassemble translated texts per row
    texts_kk_parts: dict[int, list[str]] = {}
    for idx, (row_idx, chunk_idx, _) in enumerate(chunk_map):
        if row_idx not in texts_kk_parts:
            texts_kk_parts[row_idx] = []
        texts_kk_parts[row_idx].append(all_translated[idx])

    results = []
    for i, r in enumerate(rows):
        out = dict(r)
        out["title_kk"] = titles_kk[i]
        out["summary_kk"] = summaries_kk[i]
        parts = texts_kk_parts.get(i, [""])
        out["text_kk"] = " ".join(p for p in parts if p)
        results.append(out)

    return results


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(records: list[dict], output_dir: Path, split: str, split_id: int):
    """Save translated records to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}_part{split_id}.parquet"
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"  Checkpoint saved: {path} ({len(records)} rows)")
    return path


def load_checkpoint(output_dir: Path, split: str, split_id: int) -> list[dict]:
    """Load existing checkpoint if available."""
    path = output_dir / f"{split}_part{split_id}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        print(f"  Resuming from checkpoint: {path} ({len(df)} rows)")
        return df.to_dict("records")
    return []


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args):
    """Run speed benchmark with different configurations."""
    print("=" * 60)
    print("BENCHMARK MODE")
    print("=" * 60)

    ds = load_dataset("IlyaGusev/gazeta", split=f"train[:{args.num_samples}]")
    print(f"Loaded {len(ds)} samples")

    configs = []

    if args.benchmark_variant == "all" or args.benchmark_variant == "baseline":
        configs.append(("A. Baseline (bs=1, fp32, beams=4)", 1, False, 4))
    if args.benchmark_variant == "all" or args.benchmark_variant == "batched":
        configs.append(("B. Batched (bs=16, fp32, beams=4)", 16, False, 4))
    if args.benchmark_variant == "all" or args.benchmark_variant == "bf16":
        configs.append(("C. bf16 (bs=16, bf16, beams=4)", 16, True, 4))
    if args.benchmark_variant == "all" or args.benchmark_variant == "greedy":
        configs.append(("D. Greedy (bs=16, bf16, beams=1)", 16, True, 1))

    device = torch.device(f"cuda:{args.gpu_id}")
    model_name = args.model_name

    results = []

    for name, batch_size, use_bf16, num_beams in configs:
        print(f"\n--- {name} ---")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device)
        model.eval()

        # Benchmark: translate title + summary (short fields)
        t0 = time.time()
        short_count = 0
        for i in range(0, len(ds), batch_size):
            batch_rows = ds[i:i + batch_size]
            # ds[i:j] returns dict of lists, convert to list of dicts
            n = len(batch_rows["title"])
            titles = batch_rows["title"]
            summaries = batch_rows["summary"]
            translate_batch(titles, model, tokenizer, device, num_beams=num_beams)
            translate_batch(summaries, model, tokenizer, device, num_beams=num_beams)
            short_count += n
        t_short = time.time() - t0

        # Benchmark: translate text (long, chunked)
        t0 = time.time()
        long_count = 0
        for i in range(len(ds)):
            text = ds[i]["text"] or ""
            translate_field(
                text, model, tokenizer, device,
                need_chunking=True, num_beams=num_beams,
            )
            long_count += 1
        t_long = time.time() - t0

        row = {
            "config": name,
            "short_samples": short_count,
            "short_time_s": round(t_short, 1),
            "short_samples_per_sec": round(short_count / t_short, 2),
            "long_samples": long_count,
            "long_time_s": round(t_long, 1),
            "long_samples_per_sec": round(long_count / t_long, 2),
        }
        results.append(row)
        print(f"  Short (title+summary): {row['short_samples_per_sec']} samples/sec")
        print(f"  Long (text, chunked):  {row['long_samples_per_sec']} samples/sec")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Save results
    out = Path(args.output_dir) / "benchmark_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


# ---------------------------------------------------------------------------
# Main translation loop
# ---------------------------------------------------------------------------

def load_optimized_model(model_name: str, device: torch.device, use_bf16: bool = True, use_compile: bool = True):
    """Load model with optimizations: bf16, BetterTransformer, torch.compile."""
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, dtype=dtype,
    ).to(device)
    model.eval()

    if use_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("  torch.compile enabled (max-autotune)", flush=True)
        except Exception as e:
            print(f"  torch.compile failed ({e}), continuing without it", flush=True)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded: {total_params:.0f}M params, dtype={dtype}, device={device}")
    return model, tokenizer


def run_translate(args):
    """Run translation on a dataset split."""
    device = torch.device(f"cuda:{args.gpu_id}")
    output_dir = Path(args.output_dir)

    print(f"GPU: {args.gpu_id}, Split: {args.split_id}/{args.num_splits}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}, bf16: {args.bf16}, beams: {args.num_beams}")
    print(f"Chunk batch size: {args.chunk_batch_size}")

    # Load model with optimizations
    model, tokenizer = load_optimized_model(
        args.model_name, device, use_bf16=args.bf16, use_compile=not args.no_compile,
    )
    print("Model ready")

    # Process each split
    for split in args.dataset_splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")

        ds = load_dataset("IlyaGusev/gazeta", split=split)
        total = len(ds)

        if args.num_samples > 0:
            total = min(args.num_samples, total)
            ds = ds.select(range(total))

        # Compute this worker's slice
        chunk_size = total // args.num_splits
        start = args.split_id * chunk_size
        end = total if args.split_id == args.num_splits - 1 else start + chunk_size
        ds = ds.select(range(start, end))
        print(f"  This worker: rows {start}–{end} ({len(ds)} samples)")

        # Resume from checkpoint
        existing = load_checkpoint(output_dir, split, args.split_id)
        start_idx = len(existing)
        if start_idx > 0:
            print(f"  Resuming from row {start_idx}")

        translated = existing
        t0 = time.time()
        batch = []

        for i in range(start_idx, len(ds)):
            row = ds[i]
            # Convert to plain dict
            row_dict = {k: row[k] for k in row.keys() if k in [
                "text", "summary", "title", "date", "url",
            ]}
            batch.append(row_dict)

            if len(batch) >= args.batch_size:
                results = translate_batch_fields(
                    batch, model, tokenizer, device,
                    num_beams=args.num_beams,
                    chunk_batch_size=args.chunk_batch_size,
                )
                translated.extend(results)
                batch = []

                done = len(translated)
                elapsed = time.time() - t0
                speed = (done - start_idx) / elapsed if elapsed > 0 else 0
                eta = (len(ds) - done) / speed if speed > 0 else 0
                print(
                    f"  [{split}] {done}/{len(ds)} "
                    f"({100*done/len(ds):.1f}%) "
                    f"| {speed:.2f} samples/sec "
                    f"| ETA: {eta/60:.0f} min"
                )

                # Checkpoint every 1000 samples
                if done % 1000 < args.batch_size:
                    save_checkpoint(translated, output_dir, split, args.split_id)

        # Final batch
        if batch:
            results = translate_batch_fields(
                batch, model, tokenizer, device,
                num_beams=args.num_beams,
                chunk_batch_size=args.chunk_batch_size,
            )
            translated.extend(results)

        # Final save
        save_checkpoint(translated, output_dir, split, args.split_id)

        elapsed = time.time() - t0
        actual_translated = len(translated) - start_idx
        speed = actual_translated / elapsed if elapsed > 0 else 0
        print(f"\n  Split {split} done: {len(translated)} rows in {elapsed/60:.1f} min ({speed:.2f} samples/sec)")

    print("\nTranslation complete!")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def run_merge(args):
    """Merge parquet parts from multiple workers."""
    output_dir = Path(args.output_dir)
    print("Merging results...")

    for split in args.dataset_splits:
        parts = sorted(output_dir.glob(f"{split}_part*.parquet"))
        if not parts:
            print(f"  No parts found for split '{split}', skipping")
            continue

        dfs = [pd.read_parquet(p) for p in parts]
        merged = pd.concat(dfs, ignore_index=True)
        out_path = output_dir / f"{split}.parquet"
        merged.to_parquet(out_path, index=False)
        print(f"  {split}: {len(merged)} rows -> {out_path}")

        # Verify
        original = load_dataset("IlyaGusev/gazeta", split=split)
        if len(merged) == len(original):
            print(f"  ✓ Row count matches original ({len(original)})")
        else:
            print(f"  ✗ Row count mismatch: {len(merged)} vs {len(original)} original")

    print("Merge complete!")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def run_upload(args):
    """Upload merged parquets to HuggingFace."""
    from huggingface_hub import HfApi

    output_dir = Path(args.output_dir)
    repo_id = args.hf_repo

    print(f"Uploading to {repo_id}...")

    # Load all splits
    splits = {}
    for split in args.dataset_splits:
        path = output_dir / f"{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            splits[split] = Dataset.from_pandas(df)
            print(f"  {split}: {len(df)} rows")

    if not splits:
        print("No merged parquets found. Run --merge first.")
        return

    ds_dict = DatasetDict(splits)

    # Push
    ds_dict.push_to_hub(repo_id, private=False)
    print(f"\nUploaded to https://huggingface.co/datasets/{repo_id}")

    # Upload README from pre-written dataset card
    api = HfApi()
    readme_path = Path(__file__).parent.parent / "dataset_card_gazeta_kazakh.md"
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"README uploaded from {readme_path}")
    else:
        print(f"WARNING: {readme_path} not found, skipping README upload")


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def run_preview(args):
    """Show a few translation examples for quality check."""
    output_dir = Path(args.output_dir)
    path = output_dir / "train_part0.parquet"
    if not path.exists():
        path = output_dir / "train.parquet"
    if not path.exists():
        print("No translated data found. Run translation first.")
        return

    df = pd.read_parquet(path)
    n = min(5, len(df))
    print(f"Showing {n} examples:\n")

    for i in range(n):
        row = df.iloc[i]
        print(f"--- Example {i+1} ---")
        print(f"Title (ru):      {row.get('title', '')[:100]}")
        print(f"Title (kk):      {row.get('title_kk', '')[:100]}")
        print(f"Summary (ru):    {row.get('summary', '')[:200]}")
        print(f"Summary (kk):    {row.get('summary_kk', '')[:200]}")
        print(f"Text (ru):       {str(row.get('text', ''))[:200]}...")
        print(f"Text (kk):       {str(row.get('text_kk', ''))[:200]}...")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Translate gazeta dataset ru→kk")
    parser.add_argument("--model-name", default="deepvk/kazRush-ru-kk")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=0,
                        help="0 = all samples")
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--output-dir", default="./gazeta_kk")
    parser.add_argument("--dataset-splits", nargs="+",
                        default=["train", "validation", "test"])
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--chunk-batch-size", type=int, default=32,
                        help="Batch size for text chunk translation")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")

    # Mode flags
    parser.add_argument("--benchmark", action="store_true",
                        help="Run speed benchmark")
    parser.add_argument("--benchmark-variant", default="all",
                        choices=["all", "baseline", "batched", "bf16", "greedy"])
    parser.add_argument("--merge", action="store_true",
                        help="Merge parquet parts")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace")
    parser.add_argument("--hf-repo", default="saken-tukenov/sozkz-corpus-raw-kk-gazeta-v1")
    parser.add_argument("--preview", action="store_true",
                        help="Show translation examples")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args)
    elif args.merge:
        run_merge(args)
    elif args.upload:
        run_upload(args)
    elif args.preview:
        run_preview(args)
    else:
        run_translate(args)


if __name__ == "__main__":
    main()
