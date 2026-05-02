#!/usr/bin/env python3
"""
Phase 2: Add Russian corpus + parallel kk-ru data to the annotated dataset.

Reads phase1_stats.json to know how much Russian already exists in the kk dataset,
then adds enough Russian from CulturaX (cleaned web) + Wikipedia to match kk volume.
Also adds kk-ru parallel pairs from OPUS-100 and KazNU corpus.

Usage:
    python3 add_russian_and_parallel.py
    python3 add_russian_and_parallel.py --max-ru-docs 10000 --max-parallel 1000

Output: stukenov/ekitil-corpus-annotated-v1 (merged kk + ru + parallel)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import unicodedata
from collections import Counter

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
TG_BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = "47474471"
HF_KK_ANNOTATED = "stukenov/ekitil-corpus-annotated-kk-v1"
HF_OUTPUT = "stukenov/ekitil-corpus-annotated-v1"
CACHE_DIR = "/workspace/exp027/cache"
STATS_PATH = "/workspace/exp027/phase1_stats.json"

RE_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')
RE_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
MIN_SENT_CHARS = 10
RU_CONF_THRESHOLD = 0.7


def tg_send(text: str):
    try:
        import urllib.request, urllib.parse
        url = (
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage?"
            f"chat_id={TG_CHAT_ID}&text={urllib.parse.quote(text)}"
        )
        urllib.request.urlopen(url, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


def load_fasttext():
    import fasttext
    model_path = "/tmp/lid.176.bin"
    if not os.path.exists(model_path):
        subprocess.run(
            ["wget", "-q",
             "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
             "-O", model_path],
            check=True,
        )
    fasttext.FastText.eprint = lambda x: None
    return fasttext.load_model(model_path)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = RE_SENT_SPLIT.split(text)
    expanded = []
    for part in parts:
        for sub in part.split("\n"):
            sub = sub.strip()
            if len(sub) >= MIN_SENT_CHARS:
                expanded.append(sub)
    return expanded


def make_doc_id(sentences: list[str]) -> str:
    reassembled = "\n".join(sentences)
    return hashlib.md5(reassembled.encode("utf-8")).hexdigest()[:16]


def detect_lang(text: str, ft_model) -> tuple[str, float]:
    clean = text.replace("\n", " ")[:500]
    if len(clean) < 5:
        return "unk", 0.0
    preds = ft_model.predict(clean, k=1)
    lang = preds[0][0].replace("__label__", "")
    conf = round(float(preds[1][0]), 3)
    return lang, conf


def collect_russian(ft_model, target_sents: int, max_docs: int = 0) -> dict:
    """Collect Russian sentences from CulturaX + Wikipedia."""
    from datasets import load_dataset

    rows = {
        "doc_id": [], "sent_idx": [], "text": [], "source": [],
        "domain": [], "detected_lang": [], "lang_confidence": [],
        "num_chars": [], "is_kk": [],
    }

    total_sents = 0

    # --- CulturaX (ru) — primary, already cleaned/deduped ---
    logger.info(f"Loading CulturaX (ru), target ~{target_sents:,} sents...")
    try:
        ds_cx = load_dataset("uonlp/CulturaX", "ru", split="train",
                             streaming=True, cache_dir=CACHE_DIR, trust_remote_code=False)
        doc_count = 0
        for row in tqdm(ds_cx, desc="CulturaX ru"):
            if total_sents >= target_sents:
                break
            if max_docs > 0 and doc_count >= max_docs:
                break

            text = normalize(row.get("text", ""))
            if len(text) < 50:
                continue

            # Quick langcheck on first 200 chars
            lang, conf = detect_lang(text[:200], ft_model)
            if lang != "ru" or conf < RU_CONF_THRESHOLD:
                continue

            sentences = split_sentences(text)
            if not sentences:
                continue

            doc_id = make_doc_id(sentences)
            doc_count += 1

            for si, sent in enumerate(sentences):
                slang, sconf = detect_lang(sent, ft_model)
                rows["doc_id"].append(doc_id)
                rows["sent_idx"].append(si)
                rows["text"].append(sent)
                rows["source"].append("culturax-ru")
                rows["domain"].append("web")
                rows["detected_lang"].append(slang)
                rows["lang_confidence"].append(sconf)
                rows["num_chars"].append(len(sent))
                rows["is_kk"].append(False)
                total_sents += 1

            if doc_count % 100_000 == 0:
                logger.info(f"  CulturaX: {doc_count:,} docs, {total_sents:,} sents")

        logger.info(f"  CulturaX done: {doc_count:,} docs, {total_sents:,} sents")
    except Exception as e:
        logger.error(f"  CulturaX failed: {e}")

    # --- Wikipedia (ru) — supplement if CulturaX not enough ---
    if total_sents < target_sents:
        remaining = target_sents - total_sents
        logger.info(f"Adding Wikipedia (ru) for {remaining:,} more sents...")
        try:
            ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train",
                                   cache_dir=CACHE_DIR, trust_remote_code=False)
            # Shuffle to get random sample
            indices = np.random.default_rng(42).permutation(len(ds_wiki))
            doc_count_wiki = 0

            for idx in tqdm(indices, desc="Wiki ru"):
                if total_sents >= target_sents:
                    break

                text = normalize(ds_wiki[int(idx)].get("text", ""))
                if len(text) < 100:  # skip stubs
                    continue

                sentences = split_sentences(text)
                if len(sentences) < 2:  # skip very short articles
                    continue

                doc_id = make_doc_id(sentences)
                doc_count_wiki += 1

                for si, sent in enumerate(sentences):
                    slang, sconf = detect_lang(sent, ft_model)
                    rows["doc_id"].append(doc_id)
                    rows["sent_idx"].append(si)
                    rows["text"].append(sent)
                    rows["source"].append("wikipedia-ru")
                    rows["domain"].append("wiki")
                    rows["detected_lang"].append(slang)
                    rows["lang_confidence"].append(sconf)
                    rows["num_chars"].append(len(sent))
                    rows["is_kk"].append(False)
                    total_sents += 1

            logger.info(f"  Wikipedia done: {doc_count_wiki:,} docs, +{total_sents - (total_sents - remaining):,} sents")
        except Exception as e:
            logger.error(f"  Wikipedia failed: {e}")

    logger.info(f"Total Russian sentences: {total_sents:,}")
    return rows


def collect_parallel(ft_model, max_pairs: int = 0) -> dict:
    """Collect kk-ru parallel pairs from OPUS-100 and KazNU."""
    from datasets import load_dataset

    rows = {
        "doc_id": [], "sent_idx": [], "text": [], "source": [],
        "domain": [], "detected_lang": [], "lang_confidence": [],
        "num_chars": [], "is_kk": [],
    }

    pair_count = 0

    # --- OPUS-100 kk-ru ---
    logger.info("Loading OPUS-100 kk-ru...")
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "kk-ru", split="train",
                          cache_dir=CACHE_DIR, trust_remote_code=False)
        for row in tqdm(ds, desc="OPUS-100"):
            if max_pairs > 0 and pair_count >= max_pairs:
                break

            tr = row.get("translation", {})
            kk = tr.get("kk", "").strip()
            ru = tr.get("ru", "").strip()
            if len(kk) < 10 or len(ru) < 10:
                continue

            doc_id = hashlib.md5(f"{kk}|||{ru}".encode()).hexdigest()[:16]

            kk_lang, kk_conf = detect_lang(kk, ft_model)
            ru_lang, ru_conf = detect_lang(ru, ft_model)

            # kk side
            rows["doc_id"].append(doc_id)
            rows["sent_idx"].append(0)
            rows["text"].append(kk)
            rows["source"].append("opus-100-kkru")
            rows["domain"].append("parallel")
            rows["detected_lang"].append(kk_lang)
            rows["lang_confidence"].append(kk_conf)
            rows["num_chars"].append(len(kk))
            rows["is_kk"].append(kk_lang == "kk")

            # ru side
            rows["doc_id"].append(doc_id)
            rows["sent_idx"].append(1)
            rows["text"].append(ru)
            rows["source"].append("opus-100-kkru")
            rows["domain"].append("parallel")
            rows["detected_lang"].append(ru_lang)
            rows["lang_confidence"].append(ru_conf)
            rows["num_chars"].append(len(ru))
            rows["is_kk"].append(False)

            pair_count += 1

        logger.info(f"  OPUS-100: {pair_count:,} pairs")
    except Exception as e:
        logger.error(f"  OPUS-100 failed: {e}")

    # --- KazNU parallel corpus ---
    logger.info("Loading KazNU parallel corpus...")
    try:
        ds = load_dataset("Dauren-Nur/kaz_rus_parallel_corpora_KAZNU", split="train",
                          cache_dir=CACHE_DIR, trust_remote_code=False)
        kaznu_count = 0
        for row in tqdm(ds, desc="KazNU"):
            if max_pairs > 0 and pair_count >= max_pairs:
                break

            # Try common column names
            kk = (row.get("kk", "") or row.get("kaz", "") or
                  row.get("kazakh", "") or row.get("source", "")).strip()
            ru = (row.get("ru", "") or row.get("rus", "") or
                  row.get("russian", "") or row.get("target", "")).strip()

            if len(kk) < 10 or len(ru) < 10:
                continue

            doc_id = hashlib.md5(f"{kk}|||{ru}".encode()).hexdigest()[:16]

            kk_lang, kk_conf = detect_lang(kk, ft_model)
            ru_lang, ru_conf = detect_lang(ru, ft_model)

            rows["doc_id"].append(doc_id)
            rows["sent_idx"].append(0)
            rows["text"].append(kk)
            rows["source"].append("kaznu-parallel")
            rows["domain"].append("parallel")
            rows["detected_lang"].append(kk_lang)
            rows["lang_confidence"].append(kk_conf)
            rows["num_chars"].append(len(kk))
            rows["is_kk"].append(kk_lang == "kk")

            rows["doc_id"].append(doc_id)
            rows["sent_idx"].append(1)
            rows["text"].append(ru)
            rows["source"].append("kaznu-parallel")
            rows["domain"].append("parallel")
            rows["detected_lang"].append(ru_lang)
            rows["lang_confidence"].append(ru_conf)
            rows["num_chars"].append(len(ru))
            rows["is_kk"].append(False)

            pair_count += 1
            kaznu_count += 1

        logger.info(f"  KazNU: {kaznu_count:,} pairs")
    except Exception as e:
        logger.error(f"  KazNU failed: {e}")

    logger.info(f"Total parallel pairs: {pair_count:,} ({len(rows['text']):,} rows)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ru-docs", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset, Dataset, concatenate_datasets

    ft_model = load_fasttext()

    # ---- Load Phase 1 stats ----
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            stats = json.load(f)
        kk_sents = stats["kk_sents"]
        ru_already = stats["ru_sents"]
        logger.info(f"Phase 1 stats: {kk_sents:,} kk sents, {ru_already:,} ru sents already in kk dataset")
    else:
        logger.warning("No phase1_stats.json found, assuming kk_sents=50M, ru_already=0")
        kk_sents = 50_000_000
        ru_already = 0

    # Target: match kk volume with Russian (minus what's already there)
    target_ru = max(0, kk_sents - ru_already)
    logger.info(f"Target additional Russian sentences: {target_ru:,}")

    tg_send(f"🚀 Phase 2 started: collecting parallel data (extra ru needed: {target_ru:,})")

    # ---- Collect Russian (skip if enough already) ----
    if target_ru > 5_000_000:  # only collect if we need 5M+ more
        ru_rows = collect_russian(ft_model, target_ru, max_docs=args.max_ru_docs)
    else:
        logger.info(f"  Skipping Russian collection — already have enough ({ru_already:,} vs {kk_sents:,} kk)")
        ru_rows = {k: [] for k in ["doc_id", "sent_idx", "text", "source", "domain",
                                     "detected_lang", "lang_confidence", "num_chars", "is_kk"]}

    # ---- Collect Parallel ----
    par_rows = collect_parallel(ft_model, max_pairs=args.max_parallel)

    # ---- Merge: parallel only (don't reload Phase 1, just upload parallel separately) ----
    logger.info("Building parallel + extra ru dataset...")
    from datasets import Dataset
    all_rows = {k: ru_rows[k] + par_rows[k] for k in ru_rows}
    ds_new = Dataset.from_dict(all_rows)
    logger.info(f"  New rows: {len(ds_new):,}")

    # ---- Stats ----
    n = len(ds_new)
    if n > 0:
        source_counter = Counter(ds_new["source"])
        lang_counter = Counter(ds_new["detected_lang"])

        logger.info("=" * 70)
        logger.info(f"PHASE 2 RESULT: {n:,} new sentences (parallel + extra ru)")
        logger.info(f"  BY SOURCE:")
        for src, cnt in source_counter.most_common():
            logger.info(f"    {src:25s}: {cnt:>10,}")
        logger.info(f"  BY LANGUAGE:")
        for lang, cnt in lang_counter.most_common(5):
            logger.info(f"    {lang:5s}: {cnt:>10,}")
        logger.info("=" * 70)

        # Upload parallel data as separate dataset
        HF_PARALLEL = "stukenov/ekitil-corpus-parallel-kkru-v1"
        logger.info(f"Uploading parallel data to {HF_PARALLEL}...")
        ds_new.push_to_hub(HF_PARALLEL, token=HF_TOKEN, private=False)
        logger.info(f"DONE: {HF_PARALLEL}")

        tg_send(
            f"✅ Phase 2 DONE!\n"
            f"Parallel pairs + extra ru: {n:,} rows\n"
            f"Sources: {dict(source_counter)}\n"
            f"Dataset: {HF_PARALLEL}"
        )
    else:
        logger.info("No new data collected")
        tg_send("⚠️ Phase 2: no parallel data found")


if __name__ == "__main__":
    main()
