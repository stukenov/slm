#!/usr/bin/env python3
"""Translate IlyaGusev/gazeta (ru) to Kazakh using google/madlad400-3b-mt.

MADLAD400 uses language tags: prepend "<2kk> " to source text.

Usage (2 GPUs in parallel):
  python scripts/translate_gazeta_madlad.py --gpu-id 0 --split-id 0 --num-splits 2 &
  python scripts/translate_gazeta_madlad.py --gpu-id 1 --split-id 1 --num-splits 2 &

  # Merge + upload
  python scripts/translate_gazeta_madlad.py --merge
  python scripts/translate_gazeta_madlad.py --upload --hf-repo saken-tukenov/sozkz-corpus-raw-kk-gazeta-v1
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/madlad400-3b-mt"
LANG_TAG = "<2kk> "
OUTPUT_DIR = "./gazeta_kk_madlad"


def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def chunk_text(text: str, tokenizer, max_tokens: int = 400) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks, current, current_len = [], [], 0
    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
        if sent_len > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            for i in range(0, len(tokens), max_tokens):
                sub = tokenizer.decode(tokens[i:i + max_tokens], skip_special_tokens=True)
                chunks.append(sub)
            continue
        if current_len + sent_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def translate_batch(
    texts: list[str], model, tokenizer, device,
    max_new_tokens: int = 512, num_beams: int = 1,
) -> list[str]:
    if not texts:
        return []
    # Prepend language tag for MADLAD400
    tagged = [LANG_TAG + t for t in texts]
    inputs = tokenizer(
        tagged, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    ).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_rows(
    rows: list[dict], model, tokenizer, device,
    num_beams: int = 1, chunk_batch_size: int = 32,
) -> list[dict]:
    # Titles
    titles = [r.get("title", "") or "" for r in rows]
    titles_kk = translate_batch(titles, model, tokenizer, device, num_beams=num_beams)

    # Summaries
    summaries = [r.get("summary", "") or "" for r in rows]
    summaries_kk = translate_batch(summaries, model, tokenizer, device, num_beams=num_beams)

    # Text chunks
    chunk_map = []
    for i, r in enumerate(rows):
        txt = r.get("text", "") or ""
        if not txt.strip():
            chunk_map.append((i, 0, ""))
            continue
        for j, c in enumerate(chunk_text(txt, tokenizer, max_tokens=400)):
            chunk_map.append((i, j, c))

    all_chunks = [c[2] for c in chunk_map]
    all_translated = []
    for b in range(0, len(all_chunks), chunk_batch_size):
        batch = all_chunks[b:b + chunk_batch_size]
        non_empty = [t for t in batch if t.strip()]
        if not non_empty:
            all_translated.extend([""] * len(batch))
            continue
        all_translated.extend(translate_batch(
            batch, model, tokenizer, device, num_beams=num_beams,
        ))

    texts_kk: dict[int, list[str]] = {}
    for idx, (ri, ci, _) in enumerate(chunk_map):
        texts_kk.setdefault(ri, []).append(all_translated[idx])

    results = []
    for i, r in enumerate(rows):
        out = dict(r)
        out["title_kk"] = titles_kk[i]
        out["summary_kk"] = summaries_kk[i]
        out["text_kk"] = " ".join(p for p in texts_kk.get(i, [""]) if p)
        results.append(out)
    return results


def save_checkpoint(records, output_dir, split, split_id):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}_part{split_id}.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    print(f"  Checkpoint: {path} ({len(records)} rows)", flush=True)


def load_checkpoint(output_dir, split, split_id):
    path = output_dir / f"{split}_part{split_id}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        print(f"  Resuming: {path} ({len(df)} rows)", flush=True)
        return df.to_dict("records")
    return []


def run_translate(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    output_dir = Path(args.output_dir)

    print(f"GPU: {args.gpu_id}, Split: {args.split_id}/{args.num_splits}", flush=True)
    print(f"Model: {MODEL_NAME}, batch={args.batch_size}, beams={args.num_beams}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}", flush=True)

    for split in args.dataset_splits:
        print(f"\n{'='*60}\nSplit: {split}\n{'='*60}", flush=True)
        ds = load_dataset("IlyaGusev/gazeta", split=split)
        total = len(ds)

        chunk_size = total // args.num_splits
        start = args.split_id * chunk_size
        end = total if args.split_id == args.num_splits - 1 else start + chunk_size
        ds = ds.select(range(start, end))
        print(f"  Rows {start}-{end} ({len(ds)} samples)", flush=True)

        existing = load_checkpoint(output_dir, split, args.split_id)
        start_idx = len(existing)
        translated = existing
        t0 = time.time()
        batch = []

        for i in range(start_idx, len(ds)):
            row = ds[i]
            row_dict = {k: row[k] for k in ["text", "summary", "title", "date", "url"] if k in row}
            batch.append(row_dict)

            if len(batch) >= args.batch_size:
                results = translate_rows(
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
                    f"| {speed:.2f} s/sec "
                    f"| ETA: {eta/60:.0f} min",
                    flush=True,
                )

                if done % 500 < args.batch_size:
                    save_checkpoint(translated, output_dir, split, args.split_id)

        if batch:
            translated.extend(translate_rows(
                batch, model, tokenizer, device,
                num_beams=args.num_beams,
                chunk_batch_size=args.chunk_batch_size,
            ))

        save_checkpoint(translated, output_dir, split, args.split_id)
        elapsed = time.time() - t0
        actual = len(translated) - start_idx
        print(f"\n  {split} done: {len(translated)} rows, {elapsed/60:.1f} min, {actual/(elapsed or 1):.2f} s/sec", flush=True)

    print("\nTranslation complete!", flush=True)


def run_merge(args):
    output_dir = Path(args.output_dir)
    for split in args.dataset_splits:
        parts = sorted(output_dir.glob(f"{split}_part*.parquet"))
        if not parts:
            continue
        dfs = [pd.read_parquet(p) for p in parts]
        merged = pd.concat(dfs, ignore_index=True)
        out = output_dir / f"{split}.parquet"
        merged.to_parquet(out, index=False)
        print(f"  {split}: {len(merged)} rows -> {out}")
    print("Merge complete!")


def run_upload(args):
    output_dir = Path(args.output_dir)
    splits = {}
    for split in args.dataset_splits:
        path = output_dir / f"{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            splits[split] = Dataset.from_pandas(df)
            print(f"  {split}: {len(df)} rows")
    if not splits:
        print("No merged data. Run --merge first.")
        return
    DatasetDict(splits).push_to_hub(args.hf_repo, private=False)
    print(f"Uploaded to https://huggingface.co/datasets/{args.hf_repo}")


def main():
    parser = argparse.ArgumentParser(description="Translate gazeta ru→kk (MADLAD400)")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--dataset-splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--chunk-batch-size", type=int, default=16)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo", default="saken-tukenov/sozkz-corpus-raw-kk-gazeta-v1")
    args = parser.parse_args()

    if args.merge:
        run_merge(args)
    elif args.upload:
        run_upload(args)
    else:
        run_translate(args)


if __name__ == "__main__":
    main()
