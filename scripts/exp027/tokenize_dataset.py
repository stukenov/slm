#!/usr/bin/env python3
"""
Tokenize the annotated EkiTil dataset with the trained BPE tokenizer.

Loads annotated dataset from HF, filters by language (kk + ru only, no junk),
tokenizes with the new BPE 64K tokenizer, packs into blocks, uploads.

Usage:
    python3 tokenize_dataset.py
    python3 tokenize_dataset.py --block-size 2048
    python3 tokenize_dataset.py --kk-ratio 0.5 --ru-ratio 0.5

Output: stukenov/ekitil-corpus-tokenized-kkru-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_ANNOTATED = "stukenov/ekitil-corpus-annotated-kk-v1"
HF_PARALLEL = "stukenov/ekitil-corpus-parallel-kkru-v1"
HF_TOKENIZER = "stukenov/ekitil-vocab-bpe-64k-kkru-v1"
HF_OUTPUT = "stukenov/ekitil-corpus-tokenized-kkru-v1"
WORK_DIR = Path("/root/slm/exp027")
CACHE_DIR = str(WORK_DIR / "cache")

TG_BOT_TOKEN = "REDACTED_TG_BOT_TOKEN"
TG_CHAT_ID = "47474471"


def tg_send(text: str):
    try:
        url = (
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage?"
            f"chat_id={TG_CHAT_ID}&text={urllib.parse.quote(text)}"
        )
        urllib.request.urlopen(url, timeout=10)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--kk-ratio", type=float, default=0.50, help="Fraction of kk tokens")
    parser.add_argument("--ru-ratio", type=float, default=0.45, help="Fraction of ru tokens")
    parser.add_argument("--parallel-ratio", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=0, help="Max total tokens (0=all)")
    args = parser.parse_args()

    from datasets import load_dataset, Dataset
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load tokenizer ----
    logger.info(f"Loading tokenizer from {HF_TOKENIZER}...")
    tok_path = hf_hub_download(HF_TOKENIZER, "tokenizer.json", token=HF_TOKEN,
                               cache_dir=CACHE_DIR)
    tokenizer = Tokenizer.from_file(tok_path)
    logger.info(f"  Vocab size: {tokenizer.get_vocab_size()}")

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    kk_tag = tokenizer.token_to_id("<|kk|>")
    ru_tag = tokenizer.token_to_id("<|ru|>")
    translate_tag = tokenizer.token_to_id("<|translate|>")
    logger.info(f"  Special tokens: eos={eos_id}, kk={kk_tag}, ru={ru_tag}, translate={translate_tag}")

    tg_send(f"🔤 Tokenization started (block_size={args.block_size})")

    # ---- Load annotated dataset (streaming to save RAM) ----
    logger.info(f"Loading {HF_ANNOTATED} (streaming)...")
    ds = load_dataset(HF_ANNOTATED, split="train", streaming=True)

    # ---- Tokenize in streaming fashion, write to disk in chunks ----
    # We'll collect token IDs and write parquet chunks periodically
    all_ids_kk = []
    all_ids_ru = []
    kk_count = 0
    ru_count = 0
    other_count = 0

    logger.info("Tokenizing sentences (streaming)...")
    for i, row in enumerate(tqdm(ds, desc="Tokenizing")):
        lang = row["detected_lang"]
        text = row["text"]
        conf = row["lang_confidence"]

        if lang == "kk" and conf >= 0.5:
            ids = tokenizer.encode(text).ids
            all_ids_kk.append(kk_tag)
            all_ids_kk.extend(ids)
            all_ids_kk.append(eos_id)
            kk_count += 1
        elif lang == "ru" and conf >= 0.5:
            ids = tokenizer.encode(text).ids
            all_ids_ru.append(ru_tag)
            all_ids_ru.extend(ids)
            all_ids_ru.append(eos_id)
            ru_count += 1
        else:
            other_count += 1

        # Log every 5M sentences
        if (i + 1) % 5_000_000 == 0:
            logger.info(
                f"  {i+1:,} sents | kk={kk_count:,} ({len(all_ids_kk):,} tok) | "
                f"ru={ru_count:,} ({len(all_ids_ru):,} tok) | skip={other_count:,}"
            )

        # Check max tokens
        if args.max_tokens > 0 and (len(all_ids_kk) + len(all_ids_ru)) >= args.max_tokens:
            logger.info(f"  Reached max_tokens={args.max_tokens:,}, stopping")
            break

    logger.info(f"Tokenization done:")
    logger.info(f"  kk: {kk_count:,} sents -> {len(all_ids_kk):,} tokens")
    logger.info(f"  ru: {ru_count:,} sents -> {len(all_ids_ru):,} tokens")
    logger.info(f"  skipped: {other_count:,}")

    # ---- Tokenize parallel data ----
    logger.info(f"Loading parallel data from {HF_PARALLEL}...")
    all_ids_parallel = []
    try:
        ds_par = load_dataset(HF_PARALLEL, split="train", cache_dir=CACHE_DIR)
        # Group by doc_id: sent_idx=0 is kk, sent_idx=1 is ru
        pairs = {}
        for row in ds_par:
            did = row["doc_id"]
            if did not in pairs:
                pairs[did] = {}
            pairs[did][row["sent_idx"]] = row["text"]

        par_count = 0
        for did, p in pairs.items():
            if 0 in p and 1 in p:
                kk_ids = tokenizer.encode(p[0]).ids
                ru_ids = tokenizer.encode(p[1]).ids
                # kk -> ru
                all_ids_parallel.extend([kk_tag] + kk_ids + [translate_tag] + [ru_tag] + ru_ids + [eos_id])
                # ru -> kk
                all_ids_parallel.extend([ru_tag] + ru_ids + [translate_tag] + [kk_tag] + kk_ids + [eos_id])
                par_count += 1

        logger.info(f"  parallel: {par_count:,} pairs -> {len(all_ids_parallel):,} tokens")
        del ds_par, pairs
    except Exception as e:
        logger.warning(f"  Parallel failed: {e}")

    # ---- Mix by ratio ----
    total_kk = len(all_ids_kk)
    total_ru = len(all_ids_ru)
    total_par = len(all_ids_parallel)
    total_all = total_kk + total_ru + total_par

    logger.info(f"Raw totals: kk={total_kk:,}, ru={total_ru:,}, parallel={total_par:,}")
    logger.info(f"Total: {total_all:,} tokens ({total_all/1e9:.2f}B)")

    # Trim to desired ratios if needed
    target_total = total_all
    target_kk = int(target_total * args.kk_ratio)
    target_ru = int(target_total * args.ru_ratio)

    if total_kk > target_kk:
        all_ids_kk = all_ids_kk[:target_kk]
        logger.info(f"  Trimmed kk to {target_kk:,}")
    if total_ru > target_ru:
        all_ids_ru = all_ids_ru[:target_ru]
        logger.info(f"  Trimmed ru to {target_ru:,}")

    # Interleave: shuffle blocks of ~10K tokens
    logger.info("Interleaving kk + ru + parallel...")
    CHUNK = 10_000
    chunks = []
    for arr, label in [(all_ids_kk, "kk"), (all_ids_ru, "ru"), (all_ids_parallel, "par")]:
        for start in range(0, len(arr), CHUNK):
            chunks.append(arr[start:start+CHUNK])

    del all_ids_kk, all_ids_ru, all_ids_parallel

    rng = np.random.default_rng(42)
    rng.shuffle(chunks)

    # Flatten
    all_ids = []
    for chunk in chunks:
        all_ids.extend(chunk)
    del chunks

    logger.info(f"  Final: {len(all_ids):,} tokens ({len(all_ids)/1e9:.2f}B)")

    # ---- Pack into blocks ----
    logger.info(f"Packing into {args.block_size}-token blocks...")
    all_ids_np = np.array(all_ids, dtype=np.int32)
    del all_ids

    n_blocks = len(all_ids_np) // args.block_size
    all_ids_np = all_ids_np[:n_blocks * args.block_size]
    blocks = all_ids_np.reshape(n_blocks, args.block_size)
    logger.info(f"  {n_blocks:,} blocks of {args.block_size} tokens")

    # Shuffle blocks
    rng.shuffle(blocks)

    # ---- Upload ----
    logger.info(f"Uploading to {HF_OUTPUT}...")
    ds_out = Dataset.from_dict({"input_ids": blocks.tolist()})
    ds_out.push_to_hub(HF_OUTPUT, token=HF_TOKEN, private=False)
    logger.info(f"DONE: {HF_OUTPUT}")

    tg_send(
        f"✅ Tokenization DONE!\n"
        f"Blocks: {n_blocks:,} × {args.block_size}\n"
        f"Total: {n_blocks * args.block_size / 1e9:.2f}B tokens\n"
        f"Dataset: {HF_OUTPUT}"
    )


if __name__ == "__main__":
    main()
