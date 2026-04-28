#!/usr/bin/env python3
"""
exp027: Prepare bilingual kk+ru dataset for 600M Qwen3-arch model.

Pipeline (all runs on a single cheap RunPod with fast network):
  1. Download & filter Kazakh corpus → ~6B tokens
  2. Download & filter Russian corpus → ~5B tokens
  3. Download kk↔ru parallel data → ~1B tokens
  4. Train BPE tokenizer (64K vocab) on balanced sample
  5. Tokenize everything + upload to HF Hub

Usage:
    python prepare_bilingual_data.py --step all          # run everything
    python prepare_bilingual_data.py --step filter-kk    # only Kazakh filtering
    python prepare_bilingual_data.py --step filter-ru    # only Russian
    python prepare_bilingual_data.py --step parallel     # only parallel data
    python prepare_bilingual_data.py --step tokenizer    # only train tokenizer
    python prepare_bilingual_data.py --step tokenize     # only tokenize & upload

Requires:
    pip install datasets tokenizers huggingface_hub fasttext numpy tqdm datasketch
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Config
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")
WORK_DIR = Path("/workspace/exp027")
CACHE_DIR = str(WORK_DIR / "cache")

# Output repos
HF_CORPUS_KK = "stukenov/ekitil-corpus-filtered-kk-v1"
HF_CORPUS_RU = "stukenov/ekitil-corpus-ru-v1"
HF_CORPUS_PARALLEL = "stukenov/ekitil-corpus-parallel-kkru-v1"
HF_TOKENIZER = "stukenov/ekitil-vocab-bpe-64k-kkru-v1"
HF_TOKENIZED = "stukenov/ekitil-corpus-tokenized-kkru-v1"

VOCAB_SIZE = 64_000
BLOCK_SIZE = 2048

# Source datasets
KK_DATASET = "kz-transformers/multidomain-kazakh-dataset"

# Filtering thresholds
MIN_CHARS = 50
MAX_CHARS = 500_000
MIN_CYRILLIC_RATIO = 0.6
LANGID_THRESHOLD = 0.7
DEDUP_THRESHOLD = 0.8
GZIP_RATIO_MAX = 0.5  # repetitive text filter

# Regex patterns
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")
RE_KAZAKH_SPECIFIC = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")
RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


# ============================================================================
# Step 1: Filter Kazakh corpus
# ============================================================================

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def clean_junk(text: str) -> str:
    text = RE_URL.sub("", text)
    text = RE_EMAIL.sub("", text)
    text = RE_HTML_TAG.sub("", text)
    return text.strip()


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return 0.0
    cyr = len(RE_CYRILLIC.findall(text))
    return cyr / alpha


def gzip_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if len(raw) < 100:
        return 0.0
    compressed = gzip.compress(raw, compresslevel=6)
    return len(compressed) / len(raw)


def is_repetitive(text: str, threshold: float = GZIP_RATIO_MAX) -> bool:
    return gzip_ratio(text) < threshold  # very low ratio = repetitive


def load_fasttext_model():
    import fasttext
    model_path = "/tmp/lid.176.bin"
    if not os.path.exists(model_path):
        logger.info("Downloading fasttext langid model...")
        subprocess.run(
            ["wget", "-q",
             "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
             "-O", model_path],
            check=True,
        )
    # suppress fasttext warnings
    return fasttext.load_model(model_path)


def langid_filter(text: str, model, target_lang: str = "kk", threshold: float = LANGID_THRESHOLD) -> bool:
    """Returns True if text is in target language with confidence >= threshold."""
    text_clean = text.replace("\n", " ")[:500]
    predictions = model.predict(text_clean, k=1)
    lang = predictions[0][0].replace("__label__", "")
    conf = predictions[1][0]
    return lang == target_lang and conf >= threshold


def filter_kazakh_corpus():
    """Download, filter, and upload clean Kazakh corpus."""
    from datasets import load_dataset, Dataset

    logger.info("=== Step 1: Filtering Kazakh corpus ===")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {KK_DATASET}...")
    ds = load_dataset(KK_DATASET, split="train", cache_dir=CACHE_DIR)
    logger.info(f"  Raw: {len(ds)} samples")

    ft_model = load_fasttext_model()

    # Phase 1: normalize + basic filters
    logger.info("Phase 1: normalize + length + cyrillic filter...")
    clean_texts = []
    stats = Counter()

    for row in tqdm(ds, desc="Filtering kk"):
        text = row.get("text", "")
        stats["total"] += 1

        text = normalize_text(text)
        text = clean_junk(text)

        if len(text) < MIN_CHARS:
            stats["too_short"] += 1
            continue
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]

        if cyrillic_ratio(text) < MIN_CYRILLIC_RATIO:
            stats["low_cyrillic"] += 1
            continue

        if not langid_filter(text, ft_model, "kk"):
            stats["langid_fail"] += 1
            continue

        if is_repetitive(text):
            stats["repetitive"] += 1
            continue

        clean_texts.append(text)
        stats["kept"] += 1

    logger.info(f"  Stats: {dict(stats)}")

    # Phase 2: exact dedup (MD5)
    logger.info("Phase 2: exact dedup...")
    seen_hashes = set()
    deduped = []
    for text in tqdm(clean_texts, desc="Dedup"):
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(text)

    logger.info(f"  After exact dedup: {len(deduped)} (removed {len(clean_texts) - len(deduped)})")

    # Phase 3: MinHash near-dedup (optional, memory-intensive)
    try:
        from datasketch import MinHash, MinHashLSH
        logger.info("Phase 3: MinHash near-dedup...")
        lsh = MinHashLSH(threshold=DEDUP_THRESHOLD, num_perm=128)
        final_texts = []
        for i, text in enumerate(tqdm(deduped, desc="MinHash")):
            mh = MinHash(num_perm=128)
            for w in text.lower().split():
                mh.update(w.encode("utf-8"))
            if not lsh.query(mh):
                lsh.insert(f"doc_{i}", mh)
                final_texts.append(text)
        logger.info(f"  After near-dedup: {len(final_texts)} (removed {len(deduped) - len(final_texts)})")
    except ImportError:
        logger.warning("datasketch not installed, skipping near-dedup")
        final_texts = deduped

    # Upload
    logger.info(f"Uploading {len(final_texts)} texts to {HF_CORPUS_KK}...")
    ds_clean = Dataset.from_dict({"text": final_texts})
    ds_clean.push_to_hub(HF_CORPUS_KK, token=HF_TOKEN, private=False)
    logger.info(f"  DONE: {HF_CORPUS_KK}")

    return final_texts


# ============================================================================
# Step 2: Collect Russian corpus
# ============================================================================

def filter_russian_corpus():
    """Download and filter Russian corpus from multiple sources."""
    from datasets import load_dataset, Dataset

    logger.info("=== Step 2: Collecting Russian corpus ===")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    all_texts = []

    # Source 1: Russian Wikipedia
    logger.info("Loading Russian Wikipedia...")
    try:
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train",
                               cache_dir=CACHE_DIR, trust_remote_code=True)
        wiki_texts = []
        for row in tqdm(ds_wiki, desc="Wiki ru"):
            text = normalize_text(row.get("text", ""))
            if len(text) >= MIN_CHARS:
                wiki_texts.append(text)
        logger.info(f"  Wikipedia: {len(wiki_texts)} articles")
        all_texts.extend(wiki_texts)
    except Exception as e:
        logger.error(f"  Wikipedia failed: {e}")

    # Source 2: CulturaX Russian subset (streaming, take ~3M docs)
    logger.info("Loading CulturaX (ru)...")
    try:
        ds_cx = load_dataset("uonlp/CulturaX", "ru", split="train",
                             streaming=True, cache_dir=CACHE_DIR, trust_remote_code=True)
        cx_texts = []
        target_cx = 3_000_000  # ~3M docs, aim for ~2-3B tokens
        for i, row in enumerate(tqdm(ds_cx, desc="CulturaX ru", total=target_cx)):
            if i >= target_cx:
                break
            text = normalize_text(row.get("text", ""))
            if len(text) < MIN_CHARS:
                continue
            if cyrillic_ratio(text) < MIN_CYRILLIC_RATIO:
                continue
            cx_texts.append(text)
        logger.info(f"  CulturaX: {len(cx_texts)} docs")
        all_texts.extend(cx_texts)
    except Exception as e:
        logger.error(f"  CulturaX failed: {e}")

    # Exact dedup
    logger.info(f"Total raw: {len(all_texts)}, deduplicating...")
    seen = set()
    deduped = []
    for text in tqdm(all_texts, desc="Dedup ru"):
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(text)

    logger.info(f"  After dedup: {len(deduped)}")

    # Upload
    logger.info(f"Uploading {len(deduped)} texts to {HF_CORPUS_RU}...")
    ds_clean = Dataset.from_dict({"text": deduped})
    ds_clean.push_to_hub(HF_CORPUS_RU, token=HF_TOKEN, private=False)
    logger.info(f"  DONE: {HF_CORPUS_RU}")

    return deduped


# ============================================================================
# Step 3: Collect parallel kk↔ru data
# ============================================================================

def collect_parallel_data():
    """Collect Kazakh-Russian parallel sentences from OPUS and other sources."""
    from datasets import load_dataset, Dataset

    logger.info("=== Step 3: Collecting kk↔ru parallel data ===")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    pairs_kk = []
    pairs_ru = []

    # OPUS-100 (kk-ru)
    logger.info("Loading OPUS-100 kk-ru...")
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "kk-ru", split="train",
                          cache_dir=CACHE_DIR, trust_remote_code=True)
        for row in tqdm(ds, desc="OPUS-100"):
            tr = row.get("translation", {})
            kk = tr.get("kk", "").strip()
            ru = tr.get("ru", "").strip()
            if len(kk) >= 10 and len(ru) >= 10:
                pairs_kk.append(kk)
                pairs_ru.append(ru)
        logger.info(f"  OPUS-100: {len(pairs_kk)} pairs")
    except Exception as e:
        logger.error(f"  OPUS-100 failed: {e}")

    # CCAligned kk-ru
    logger.info("Loading CCAligned kk-ru...")
    try:
        ds = load_dataset("Helsinki-NLP/ccaligned_multilingual",
                          language_pair="kk-ru", split="train",
                          streaming=True, cache_dir=CACHE_DIR, trust_remote_code=True)
        count_before = len(pairs_kk)
        target = 2_000_000
        for i, row in enumerate(tqdm(ds, desc="CCAligned", total=target)):
            if i >= target:
                break
            src = row.get("translation", {})
            kk = src.get("kk", "").strip()
            ru = src.get("ru", "").strip()
            if len(kk) >= 10 and len(ru) >= 10:
                pairs_kk.append(kk)
                pairs_ru.append(ru)
        logger.info(f"  CCAligned: {len(pairs_kk) - count_before} pairs")
    except Exception as e:
        logger.error(f"  CCAligned failed: {e}")

    # Tatoeba kk-ru
    logger.info("Loading Tatoeba kk-ru...")
    try:
        ds = load_dataset("Helsinki-NLP/tatoeba_mt", "kaz-rus", split="test",
                          cache_dir=CACHE_DIR, trust_remote_code=True)
        count_before = len(pairs_kk)
        for row in tqdm(ds, desc="Tatoeba"):
            kk = row.get("sourceString", "").strip()
            ru = row.get("targetString", "").strip()
            if len(kk) >= 5 and len(ru) >= 5:
                pairs_kk.append(kk)
                pairs_ru.append(ru)
        logger.info(f"  Tatoeba: {len(pairs_kk) - count_before} pairs")
    except Exception as e:
        logger.error(f"  Tatoeba failed: {e}")

    logger.info(f"Total parallel pairs: {len(pairs_kk)}")

    # Dedup pairs
    seen = set()
    unique_kk, unique_ru = [], []
    for kk, ru in zip(pairs_kk, pairs_ru):
        key = hashlib.md5(f"{kk}|||{ru}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique_kk.append(kk)
            unique_ru.append(ru)

    logger.info(f"After dedup: {len(unique_kk)} pairs")

    # Upload
    ds_parallel = Dataset.from_dict({"kk": unique_kk, "ru": unique_ru})
    ds_parallel.push_to_hub(HF_CORPUS_PARALLEL, token=HF_TOKEN, private=False)
    logger.info(f"  DONE: {HF_CORPUS_PARALLEL}")

    return unique_kk, unique_ru


# ============================================================================
# Step 4: Train BPE tokenizer (64K vocab, from scratch)
# ============================================================================

def train_tokenizer(kk_texts=None, ru_texts=None, parallel_kk=None, parallel_ru=None):
    """Train BPE 64K tokenizer on balanced kk+ru sample."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    from datasets import load_dataset

    logger.info("=== Step 4: Training BPE tokenizer ===")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tok_dir = WORK_DIR / "tokenizer"
    tok_dir.mkdir(exist_ok=True)

    # Load data if not passed
    if kk_texts is None:
        logger.info("Loading filtered Kazakh corpus from HF...")
        ds = load_dataset(HF_CORPUS_KK, split="train", cache_dir=CACHE_DIR)
        kk_texts = ds["text"]
    if ru_texts is None:
        logger.info("Loading Russian corpus from HF...")
        ds = load_dataset(HF_CORPUS_RU, split="train", cache_dir=CACHE_DIR)
        ru_texts = ds["text"]

    # Build balanced training sample
    # 50% kk, 45% ru, 5% parallel (interleaved)
    sample_size = 5_000_000  # 5M docs total
    n_kk = int(sample_size * 0.50)
    n_ru = int(sample_size * 0.45)
    n_par = int(sample_size * 0.05)

    logger.info(f"Sampling: {n_kk} kk, {n_ru} ru, {n_par} parallel")

    rng = np.random.default_rng(42)

    sample_kk = [kk_texts[i] for i in rng.choice(len(kk_texts), min(n_kk, len(kk_texts)), replace=False)]
    sample_ru = [ru_texts[i] for i in rng.choice(len(ru_texts), min(n_ru, len(ru_texts)), replace=False)]

    # Parallel: interleave kk and ru sentences
    sample_par = []
    if parallel_kk and parallel_ru:
        idx = rng.choice(len(parallel_kk), min(n_par, len(parallel_kk)), replace=False)
        for i in idx:
            sample_par.append(f"{parallel_kk[i]}\n{parallel_ru[i]}")

    all_texts = sample_kk + sample_ru + sample_par
    rng.shuffle(all_texts)
    logger.info(f"  Total training texts: {len(all_texts)}")

    # Write temp file for tokenizer training
    train_file = str(WORK_DIR / "tokenizer_train.txt")
    logger.info(f"Writing training corpus to {train_file}...")
    with open(train_file, "w") as f:
        for text in tqdm(all_texts, desc="Writing"):
            f.write(text + "\n")

    # Train BPE
    SPECIAL_TOKENS = [
        "<|endoftext|>",   # 0
        "<|padding|>",     # 1
        "<|startoftext|>", # 2
        "<|kk|>",          # 3 - Kazakh language tag
        "<|ru|>",          # 4 - Russian language tag
        "<|translate|>",   # 5 - translation task tag
    ]

    logger.info(f"Training ByteLevel BPE, vocab_size={VOCAB_SIZE}...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=50,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([train_file], trainer=trainer)
    logger.info(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # Save locally
    tokenizer.save(str(tok_dir / "tokenizer.json"))
    logger.info(f"  Saved to {tok_dir}")

    # Fertility analysis
    logger.info("Fertility analysis:")
    for lang, texts, n_sample in [("kk", sample_kk, 1000), ("ru", sample_ru, 1000)]:
        total_words = 0
        total_tokens = 0
        for text in texts[:n_sample]:
            words = text.split()
            total_words += len(words)
            encoded = tokenizer.encode(text)
            total_tokens += len(encoded.ids)
        fertility = total_tokens / max(total_words, 1)
        logger.info(f"  {lang}: {fertility:.2f} tokens/word ({total_tokens} tokens / {total_words} words)")

    # Upload to HF
    logger.info(f"Uploading tokenizer to {HF_TOKENIZER}...")
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_TOKENIZER, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(
        folder_path=str(tok_dir),
        repo_id=HF_TOKENIZER,
        token=HF_TOKEN,
    )
    logger.info(f"  DONE: {HF_TOKENIZER}")

    # Cleanup temp file
    os.remove(train_file)

    return tokenizer


# ============================================================================
# Step 5: Tokenize everything and upload
# ============================================================================

def tokenize_and_upload(tokenizer=None):
    """Tokenize all corpora and pack into blocks for training."""
    from datasets import load_dataset, Dataset

    logger.info("=== Step 5: Tokenize and upload ===")

    if tokenizer is None:
        from tokenizers import Tokenizer
        tok_path = str(WORK_DIR / "tokenizer" / "tokenizer.json")
        if not os.path.exists(tok_path):
            logger.info("Loading tokenizer from HF...")
            from huggingface_hub import hf_hub_download
            tok_path = hf_hub_download(HF_TOKENIZER, "tokenizer.json", token=HF_TOKEN)
        tokenizer = Tokenizer.from_file(tok_path)

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    kk_tag = tokenizer.token_to_id("<|kk|>")
    ru_tag = tokenizer.token_to_id("<|ru|>")
    translate_tag = tokenizer.token_to_id("<|translate|>")

    all_token_ids = []

    # Tokenize Kazakh
    logger.info("Tokenizing Kazakh corpus...")
    ds_kk = load_dataset(HF_CORPUS_KK, split="train", cache_dir=CACHE_DIR)
    for row in tqdm(ds_kk, desc="Tokenize kk"):
        ids = tokenizer.encode(row["text"]).ids
        all_token_ids.append(kk_tag)
        all_token_ids.extend(ids)
        all_token_ids.append(eos_id)

    n_kk = len(all_token_ids)
    logger.info(f"  Kazakh tokens: {n_kk:,}")

    # Tokenize Russian
    logger.info("Tokenizing Russian corpus...")
    ds_ru = load_dataset(HF_CORPUS_RU, split="train", cache_dir=CACHE_DIR)
    for row in tqdm(ds_ru, desc="Tokenize ru"):
        ids = tokenizer.encode(row["text"]).ids
        all_token_ids.append(ru_tag)
        all_token_ids.extend(ids)
        all_token_ids.append(eos_id)

    n_ru = len(all_token_ids) - n_kk
    logger.info(f"  Russian tokens: {n_ru:,}")

    # Tokenize parallel (both directions)
    logger.info("Tokenizing parallel corpus...")
    ds_par = load_dataset(HF_CORPUS_PARALLEL, split="train", cache_dir=CACHE_DIR)
    for row in tqdm(ds_par, desc="Tokenize parallel"):
        # kk → ru direction
        kk_ids = tokenizer.encode(row["kk"]).ids
        ru_ids = tokenizer.encode(row["ru"]).ids
        all_token_ids.extend([kk_tag] + kk_ids + [translate_tag] + [ru_tag] + ru_ids + [eos_id])
        # ru → kk direction
        all_token_ids.extend([ru_tag] + ru_ids + [translate_tag] + [kk_tag] + kk_ids + [eos_id])

    n_par = len(all_token_ids) - n_kk - n_ru
    logger.info(f"  Parallel tokens: {n_par:,}")
    logger.info(f"  TOTAL tokens: {len(all_token_ids):,}")

    # Pack into blocks
    logger.info(f"Packing into {BLOCK_SIZE}-token blocks...")
    all_ids = np.array(all_token_ids, dtype=np.int32)
    n_blocks = len(all_ids) // BLOCK_SIZE
    all_ids = all_ids[: n_blocks * BLOCK_SIZE]
    blocks = all_ids.reshape(n_blocks, BLOCK_SIZE)
    logger.info(f"  {n_blocks:,} blocks of {BLOCK_SIZE} tokens")

    # Upload
    logger.info(f"Uploading to {HF_TOKENIZED}...")
    ds_tokenized = Dataset.from_dict({"input_ids": blocks.tolist()})
    ds_tokenized.push_to_hub(HF_TOKENIZED, token=HF_TOKEN, private=False)
    logger.info(f"  DONE: {HF_TOKENIZED}")

    return n_blocks


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="exp027: bilingual data preparation")
    parser.add_argument("--step", default="all",
                        choices=["all", "filter-kk", "filter-ru", "parallel", "tokenizer", "tokenize"])
    args = parser.parse_args()

    kk_texts = ru_texts = par_kk = par_ru = tokenizer_obj = None

    if args.step in ("all", "filter-kk"):
        kk_texts = filter_kazakh_corpus()

    if args.step in ("all", "filter-ru"):
        ru_texts = filter_russian_corpus()

    if args.step in ("all", "parallel"):
        par_kk, par_ru = collect_parallel_data()

    if args.step in ("all", "tokenizer"):
        tokenizer_obj = train_tokenizer(kk_texts, ru_texts, par_kk, par_ru)

    if args.step in ("all", "tokenize"):
        tokenize_and_upload(tokenizer_obj)

    logger.info("ALL DONE!")


if __name__ == "__main__":
    main()
