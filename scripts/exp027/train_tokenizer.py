#!/usr/bin/env python3
"""
Train BPE 64K tokenizer from scratch on balanced kk+ru corpus.

Uses annotated dataset from Phase 1 + parallel data from Phase 2.
Samples balanced corpus: 50% kk, 45% ru, 5% parallel.

Usage:
    python3 train_tokenizer.py
    python3 train_tokenizer.py --vocab-size 64000 --sample-size 5000000

Output: stukenov/ekitil-vocab-bpe-64k-kkru-v1
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
        pass  # kaznu may not have internet to telegram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=64_000)
    parser.add_argument("--sample-size", type=int, default=5_000_000,
                        help="Total sentences to sample for training")
    args = parser.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tok_dir = WORK_DIR / "tokenizer"
    tok_dir.mkdir(exist_ok=True)

    # ---- Sample balanced corpus ----
    logger.info("Loading annotated dataset...")
    ds = load_dataset(HF_ANNOTATED, split="train", cache_dir=CACHE_DIR)
    logger.info(f"  Loaded {len(ds):,} sentences")

    # Reservoir sampling — single pass, no index lists in memory
    n_kk = int(args.sample_size * 0.50)
    n_ru = int(args.sample_size * 0.45)

    rng = np.random.default_rng(42)

    logger.info(f"Reservoir sampling: {n_kk:,} kk, {n_ru:,} ru (single pass, low memory)...")
    kk_texts = []
    ru_texts = []
    kk_seen = 0
    ru_seen = 0

    for row in tqdm(ds, desc="Sampling", total=len(ds)):
        lang = row["detected_lang"]
        text = row["text"]

        if lang == "kk":
            kk_seen += 1
            if len(kk_texts) < n_kk:
                kk_texts.append(text)
            else:
                j = rng.integers(0, kk_seen)
                if j < n_kk:
                    kk_texts[j] = text
        elif lang == "ru":
            ru_seen += 1
            if len(ru_texts) < n_ru:
                ru_texts.append(text)
            else:
                j = rng.integers(0, ru_seen)
                if j < n_ru:
                    ru_texts[j] = text

    logger.info(f"  Sampled kk: {len(kk_texts):,} (from {kk_seen:,}), ru: {len(ru_texts):,} (from {ru_seen:,})")

    # Free dataset
    del ds

    # Load parallel
    n_par = int(args.sample_size * 0.05)
    logger.info("Loading parallel data...")
    par_sample = []
    try:
        ds_par = load_dataset(HF_PARALLEL, split="train", cache_dir=CACHE_DIR)
        par_idx = rng.choice(len(ds_par), min(n_par, len(ds_par)), replace=False)
        par_sample = [ds_par[int(i)]["text"] for i in par_idx]
        logger.info(f"  Parallel: {len(par_sample):,} sentences")
        del ds_par
    except Exception as e:
        logger.warning(f"  Parallel load failed: {e}")

    all_texts = kk_texts + ru_texts + par_sample
    kk_texts = ru_texts = par_sample = None  # free refs
    rng.shuffle(all_texts)
    logger.info(f"  Total training sentences: {len(all_texts):,}")

    # ---- Write to temp file ----
    train_file = str(WORK_DIR / "tokenizer_train.txt")
    logger.info(f"Writing to {train_file}...")
    with open(train_file, "w") as f:
        for text in tqdm(all_texts, desc="Writing"):
            f.write(text + "\n")

    file_size_gb = os.path.getsize(train_file) / (1024**3)
    logger.info(f"  File size: {file_size_gb:.1f} GB")

    # Free memory
    del all_texts

    # ---- Train BPE ----
    SPECIAL_TOKENS = [
        "<|endoftext|>",   # 0
        "<|padding|>",     # 1
        "<|startoftext|>", # 2
        "<|kk|>",          # 3
        "<|ru|>",          # 4
        "<|translate|>",   # 5
    ]

    logger.info(f"Training ByteLevel BPE, vocab_size={args.vocab_size}...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=50,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([train_file], trainer=trainer)
    logger.info(f"  Final vocab size: {tokenizer.get_vocab_size()}")

    # Save
    tokenizer.save(str(tok_dir / "tokenizer.json"))
    logger.info(f"  Saved to {tok_dir}")

    # ---- Fertility analysis (use saved train file, no reload) ----
    logger.info("Fertility analysis...")
    # Read first 2000 lines from train file as test
    with open(train_file, "r") as f:
        test_lines = [f.readline().strip() for _ in range(2000)]

    for lang_name, test_texts in [("Mixed", test_lines)]:
        total_words = 0
        total_tokens = 0
        for text in test_texts:
            if len(text) < 10:
                continue
            words = text.split()
            total_words += len(words)
            encoded = tokenizer.encode(text)
            total_tokens += len(encoded.ids)

        fertility = total_tokens / max(total_words, 1)
        logger.info(f"  {lang_name}: {fertility:.2f} tokens/word "
                     f"({total_tokens:,} tokens / {total_words:,} words)")

    # ---- Upload to HF ----
    logger.info(f"Uploading tokenizer to {HF_TOKENIZER}...")
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_TOKENIZER, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(
        folder_path=str(tok_dir),
        repo_id=HF_TOKENIZER,
        token=HF_TOKEN,
    )
    logger.info(f"DONE: {HF_TOKENIZER}")

    tg_send(
        f"✅ Tokenizer trained!\n"
        f"Vocab: {tokenizer.get_vocab_size():,}\n"
        f"Dataset: {HF_TOKENIZER}"
    )

    # Cleanup
    os.remove(train_file)
    logger.info("Cleaned up temp file")


if __name__ == "__main__":
    main()
