#!/usr/bin/env python3
"""
Document-level tokenization for EkiTil. Clean, simple, no OOM.

Single-pass streaming: read original dataset → filter → tokenize → write to disk.
Never holds more than 1 document in RAM at a time.

Usage:
    python3 tokenize_documents.py
    python3 tokenize_documents.py --block-size 2048 --kk-threshold 0.7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import struct
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_ANNOTATED = "stukenov/ekitil-corpus-annotated-kk-v1"
HF_PARALLEL = "stukenov/ekitil-corpus-parallel-kkru-v1"
HF_ORIGINAL = "kz-transformers/multidomain-kazakh-dataset"
HF_TOKENIZER = "stukenov/ekitil-vocab-bpe-64k-kkru-v1"
HF_OUTPUT = "stukenov/ekitil-corpus-tokenized-kkru-v1"
WORK_DIR = Path(os.environ.get("WORK_DIR", "/root/exp027"))

TG_BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = "47474471"

RE_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')
RE_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def tg_send(text: str):
    try:
        url = (f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage?"
               f"chat_id={TG_CHAT_ID}&text={urllib.parse.quote(text)}")
        urllib.request.urlopen(url, timeout=10)
    except Exception:
        pass


def make_doc_id(text: str) -> str:
    """Same doc_id as annotate script: split → rejoin → md5."""
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text).strip()
    parts = RE_SENT_SPLIT.split(text)
    sents = []
    for part in parts:
        for sub in part.split("\n"):
            sub = sub.strip()
            if len(sub) >= 10:
                sents.append(sub)
    if not sents:
        return ""
    return hashlib.md5("\n".join(sents).encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--kk-threshold", type=float, default=0.7)
    parser.add_argument("--ru-threshold", type=float, default=0.7)
    args = parser.parse_args()

    import glob
    from huggingface_hub import hf_hub_download, snapshot_download
    from tokenizers import Tokenizer

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = str(WORK_DIR / "cache")
    tokens_file = str(WORK_DIR / "tokens.bin")

    # ---- Load tokenizer ----
    tok_path = hf_hub_download(HF_TOKENIZER, "tokenizer.json", token=HF_TOKEN, cache_dir=cache)
    tokenizer = Tokenizer.from_file(tok_path)
    eos = tokenizer.token_to_id("<|endoftext|>")
    kk_tag = tokenizer.token_to_id("<|kk|>")
    ru_tag = tokenizer.token_to_id("<|ru|>")
    tr_tag = tokenizer.token_to_id("<|translate|>")
    logger.info(f"Tokenizer: {tokenizer.get_vocab_size()} vocab, eos={eos} kk={kk_tag} ru={ru_tag}")

    tg_send("🚀 Doc-level tokenization v5 (clean rewrite)")

    # ---- Pass 1: Build doc_id → decision from annotations (~180MB RAM) ----
    logger.info("Pass 1: scanning annotations...")
    anno_dir = snapshot_download(HF_ANNOTATED, repo_type="dataset", token=HF_TOKEN, cache_dir=cache)
    pq_files = sorted(glob.glob(f"{anno_dir}/**/*.parquet", recursive=True))

    doc_stats = {}  # doc_id -> [total, kk, ru, en]
    for pf_path in pq_files:
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=1_000_000, columns=["doc_id", "detected_lang"]):
            for did, lang in zip(batch.column("doc_id").to_pylist(),
                                  batch.column("detected_lang").to_pylist()):
                if did not in doc_stats:
                    doc_stats[did] = [0, 0, 0, 0]
                doc_stats[did][0] += 1
                if lang == "kk":
                    doc_stats[did][1] += 1
                elif lang == "ru":
                    doc_stats[did][2] += 1
                elif lang == "en":
                    doc_stats[did][3] += 1

    # Decide: include everything except English-dominant docs
    decisions = {}
    for did, (n, nk, nr, ne) in doc_stats.items():
        if n < 2:
            continue
        # Skip English-dominant documents
        if ne / n >= 0.5:
            continue
        # Tag by majority language
        if nk >= nr:
            decisions[did] = "kk"
        else:
            decisions[did] = "ru"
    del doc_stats

    n_kk = sum(1 for v in decisions.values() if v == "kk")
    n_ru = sum(1 for v in decisions.values() if v == "ru")
    logger.info(f"  Decisions: {n_kk:,} kk, {n_ru:,} ru (no en filter), dict ~{len(decisions)*20//1e6:.0f}MB")

    tg_send(f"📊 Pass 1: kk={n_kk:,}, ru={n_ru:,} (no en filter)")

    # ---- Pass 2: Stream original dataset → tokenize → write binary ----
    logger.info("Pass 2: streaming original dataset, tokenizing to disk...")
    from datasets import load_dataset
    ds = load_dataset(HF_ORIGINAL, split="train", cache_dir=cache)
    logger.info(f"  {len(ds):,} documents")

    n_tokens = 0
    n_matched = 0
    n_docs = len(ds)
    int32 = struct.Struct("<i")  # little-endian int32

    with open(tokens_file, "wb") as fout:
        for i, row in enumerate(ds):
            text = row.get("text", "")
            did = make_doc_id(text)
            if not did:
                continue

            lang = decisions.get(did)
            if not lang:
                continue

            # Tokenize full document
            tag = kk_tag if lang == "kk" else ru_tag
            ids = tokenizer.encode(text).ids
            # Write: tag + ids + eos
            fout.write(int32.pack(tag))
            for t in ids:
                fout.write(int32.pack(t))
            fout.write(int32.pack(eos))
            n_tokens += len(ids) + 2
            n_matched += 1

            if (i + 1) % 1_000_000 == 0:
                logger.info(f"  {i+1:,}/{n_docs:,} scanned, {n_matched:,} matched, {n_tokens:,} tokens")

    del ds, decisions
    logger.info(f"  Documents: {n_matched:,}, tokens: {n_tokens:,} ({n_tokens/1e9:.2f}B)")

    # ---- Parallel pairs ----
    logger.info("Adding parallel pairs...")
    n_par = 0
    try:
        ds_par = load_dataset(HF_PARALLEL, split="train", cache_dir=cache)
        pairs = {}
        for row in ds_par:
            did = row["doc_id"]
            if did not in pairs:
                pairs[did] = {}
            pairs[did][row["sent_idx"]] = row["text"]

        with open(tokens_file, "ab") as fout:
            for p in pairs.values():
                if 0 not in p or 1 not in p:
                    continue
                kk_ids = tokenizer.encode(p[0]).ids
                ru_ids = tokenizer.encode(p[1]).ids
                # kk→ru
                for t in [kk_tag] + kk_ids + [tr_tag, ru_tag] + ru_ids + [eos]:
                    fout.write(int32.pack(t))
                # ru→kk
                for t in [ru_tag] + ru_ids + [tr_tag, kk_tag] + kk_ids + [eos]:
                    fout.write(int32.pack(t))
                n_par += 1
                n_tokens += (len(kk_ids) + len(ru_ids) + 4) * 2

        del ds_par, pairs
        logger.info(f"  Parallel: {n_par:,} pairs, total tokens: {n_tokens:,}")
    except Exception as e:
        logger.warning(f"  Parallel failed: {e}")

    tg_send(f"📊 Tokenized: {n_tokens:,} ({n_tokens/1e9:.2f}B). Packing blocks...")

    # ---- Pack into blocks ----
    logger.info(f"Packing {args.block_size}-token blocks...")
    file_bytes = os.path.getsize(tokens_file)
    total_ints = file_bytes // 4
    n_blocks = total_ints // args.block_size
    keep = n_blocks * args.block_size
    logger.info(f"  {total_ints:,} ints -> {n_blocks:,} blocks (dropping {total_ints - keep:,} tail)")

    # Shuffle block indices (not data) then write parquet in chunks
    rng = np.random.default_rng(42)
    block_order = rng.permutation(n_blocks)

    import pyarrow as pa
    import pyarrow.parquet as pq_write

    parquet_out = str(WORK_DIR / "train.parquet")
    mm = np.memmap(tokens_file, dtype=np.int32, mode="r", shape=(keep,))
    blocks_view = mm.reshape(n_blocks, args.block_size)

    logger.info(f"  Writing {n_blocks:,} shuffled blocks to parquet...")
    WRITE_CHUNK = 50_000
    writer = None
    for start in range(0, n_blocks, WRITE_CHUNK):
        idx = block_order[start:start + WRITE_CHUNK]
        chunk_data = [blocks_view[i].tolist() for i in idx]
        table = pa.table({"input_ids": chunk_data})
        if writer is None:
            writer = pq_write.ParquetWriter(parquet_out, table.schema)
        writer.write_table(table)
        del chunk_data, table
        if start > 0 and start % 200_000 == 0:
            logger.info(f"    {start:,}/{n_blocks:,} blocks written")
    if writer:
        writer.close()
    del mm, blocks_view

    logger.info(f"  Uploading {parquet_out} to {HF_OUTPUT}...")
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_OUTPUT, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=parquet_out,
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=HF_OUTPUT, repo_type="dataset", token=HF_TOKEN,
    )
    logger.info(f"DONE: {HF_OUTPUT}")

    os.remove(tokens_file)
    os.remove(parquet_out)

    chinchilla = n_blocks * args.block_size / 600_000_000
    tg_send(
        f"✅ Tokenization DONE!\n"
        f"Blocks: {n_blocks:,} × {args.block_size}\n"
        f"Total: {n_blocks * args.block_size / 1e9:.2f}B tokens\n"
        f"Chinchilla (600M): {chinchilla:.1f}:1\n"
        f"Dataset: {HF_OUTPUT}"
    )

    stats = {"tokens": n_blocks * args.block_size, "blocks": n_blocks,
             "block_size": args.block_size, "chinchilla_600m": round(chinchilla, 1),
             "kk_docs": n_kk, "ru_docs": n_ru, "parallel_pairs": n_par}
    logger.info(f"Stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
