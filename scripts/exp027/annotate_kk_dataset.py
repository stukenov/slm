#!/usr/bin/env python3
"""
Phase 1: Annotate kz-transformers/multidomain-kazakh-dataset at sentence level.

Features:
  - Splits docs into sentences, fasttext langdetect each
  - Saves checkpoints every N docs (resumable)
  - Detailed logging: progress, speed, ETA
  - Uploads to HF Hub at the end

Usage:
    python3 annotate_kk_dataset.py                        # full run
    python3 annotate_kk_dataset.py --max-docs 5000        # test
    python3 annotate_kk_dataset.py --resume                # resume from checkpoint
    python3 annotate_kk_dataset.py --checkpoint-every 50000

Output: stukenov/ekitil-corpus-annotated-kk-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/workspace/exp027/phase1_detailed.log"),
    ],
)
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
KK_DATASET = "kz-transformers/multidomain-kazakh-dataset"
HF_OUTPUT = "stukenov/ekitil-corpus-annotated-kk-v1"
CACHE_DIR = "/workspace/exp027/cache"
CHECKPOINT_DIR = Path("/workspace/exp027/checkpoints")

RE_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')
RE_CYRILLIC = re.compile(r'[\u0400-\u04FF]')
RE_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

MIN_SENT_CHARS = 10
KK_CONF_THRESHOLD = 0.5
CHECKPOINT_EVERY = 50_000  # save checkpoint every N docs

# Telegram notifications
TG_BOT_TOKEN = "REDACTED_TG_BOT_TOKEN"
TG_CHAT_ID = "47474471"
TG_NOTIFY_EVERY = 500_000  # notify every N docs


def tg_send(text: str):
    """Send a Telegram notification."""
    try:
        import urllib.request
        import urllib.parse
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
        logger.info("Downloading fasttext langid model...")
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


def save_checkpoint(data: dict, doc_idx: int, stats: dict):
    """Save current progress to disk."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Save data as parquet chunks
    from datasets import Dataset
    ds = Dataset.from_dict(data)
    chunk_path = CHECKPOINT_DIR / f"chunk_{doc_idx:010d}.parquet"
    ds.to_parquet(str(chunk_path))

    # Save metadata
    meta = {
        "last_doc_idx": doc_idx,
        "total_sents": len(data["text"]),
        "stats": stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CHECKPOINT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  CHECKPOINT saved: doc_idx={doc_idx:,}, sents={len(data['text']):,} -> {chunk_path.name}")


def load_checkpoint() -> tuple[int, dict]:
    """Load last checkpoint. Returns (start_doc_idx, accumulated_stats)."""
    meta_path = CHECKPOINT_DIR / "meta.json"
    if not meta_path.exists():
        return 0, {}

    with open(meta_path) as f:
        meta = json.load(f)

    logger.info(f"  Resuming from checkpoint: doc_idx={meta['last_doc_idx']:,}, "
                f"sents={meta['total_sents']:,}, saved at {meta['timestamp']}")
    return meta["last_doc_idx"] + 1, meta.get("stats", {})


def merge_and_upload_checkpoints(hf_repo: str, token: str):
    """Merge all parquet chunks and upload directly without loading into Python dicts."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = sorted(CHECKPOINT_DIR.glob("chunk_*.parquet"))

    if not chunks:
        return None

    logger.info(f"  Merging {len(chunks)} checkpoint chunks via pyarrow (low memory)...")

    # Merge to a single parquet file on disk, not in RAM
    merged_path = str(CHECKPOINT_DIR / "merged.parquet")
    writer = None
    total_rows = 0

    for chunk in chunks:
        table = pq.read_table(str(chunk))
        if writer is None:
            writer = pq.ParquetWriter(merged_path, table.schema)
        writer.write_table(table)
        total_rows += len(table)
        del table  # free memory

    if writer:
        writer.close()

    logger.info(f"  Merged {total_rows:,} rows -> {merged_path}")

    # Compute stats from parquet without loading full data
    table = pq.read_table(merged_path, columns=["detected_lang", "is_kk", "domain"])
    langs = table.column("detected_lang").to_pylist()
    is_kk_list = table.column("is_kk").to_pylist()
    domains = table.column("domain").to_pylist()
    del table

    stats = {
        "total_rows": total_rows,
        "lang_counts": dict(Counter(langs)),
        "domain_counts": dict(Counter(domains)),
        "n_kk": sum(is_kk_list),
        "n_ru": sum(1 for l in langs if l == "ru"),
        "n_en": sum(1 for l in langs if l == "en"),
    }

    del langs, is_kk_list, domains

    # Upload parquet directly via HF API
    logger.info(f"  Uploading {merged_path} to {hf_repo}...")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(hf_repo, repo_type="dataset", exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=merged_path,
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=hf_repo,
        repo_type="dataset",
        token=token,
    )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    from datasets import load_dataset, Dataset

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Resume logic ----
    start_idx = 0
    prev_stats = {}
    if args.resume:
        start_idx, prev_stats = load_checkpoint()
        if start_idx > 0:
            logger.info(f"Resuming from doc {start_idx:,}")

    # ---- Load dataset ----
    logger.info(f"Loading {KK_DATASET}...")
    ds = load_dataset(KK_DATASET, split="train", cache_dir=CACHE_DIR)
    total = len(ds)
    logger.info(f"  Total docs: {total:,}")
    logger.info(f"  Features: {list(ds.features.keys())}")

    if args.max_docs > 0:
        total = min(args.max_docs, total)
        logger.info(f"  Limited to {total:,} docs")

    ft_model = load_fasttext()

    tg_send(f"🚀 Phase 1 started: {total:,} docs to annotate (from idx {start_idx:,})")

    # ---- Process ----
    data = {
        "doc_id": [], "sent_idx": [], "text": [], "source": [],
        "domain": [], "detected_lang": [], "lang_confidence": [],
        "num_chars": [], "is_kk": [],
    }

    # Running stats
    lang_counts = Counter(prev_stats.get("languages", {}))
    domain_counts = Counter(prev_stats.get("domains", {}))
    doc_count = start_idx
    empty_docs = prev_stats.get("empty_docs", 0)
    total_sents = prev_stats.get("total_sents", 0)

    t_start = time.time()
    last_log_time = t_start

    for doc_idx, row in enumerate(tqdm(ds, desc="Annotating", total=total)):
        if doc_idx < start_idx:
            continue
        if args.max_docs > 0 and doc_idx >= args.max_docs:
            break
        text = normalize(row.get("text", ""))

        # Try to get domain from row
        domain = row.get("source", row.get("domain", "unknown"))
        if domain == "unknown":
            domain = "unknown"

        sentences = split_sentences(text)
        if not sentences:
            empty_docs += 1
            continue

        doc_id = make_doc_id(sentences)
        doc_count += 1

        for si, sent in enumerate(sentences):
            lang, conf = detect_lang(sent, ft_model)
            is_kk = (lang == "kk" and conf >= KK_CONF_THRESHOLD)

            data["doc_id"].append(doc_id)
            data["sent_idx"].append(si)
            data["text"].append(sent)
            data["source"].append("multidomain-kk")
            data["domain"].append(domain)
            data["detected_lang"].append(lang)
            data["lang_confidence"].append(conf)
            data["num_chars"].append(len(sent))
            data["is_kk"].append(is_kk)

            lang_counts[lang] += 1
            domain_counts[domain] += 1
            total_sents += 1

        # ---- Periodic detailed logging ----
        now = time.time()
        if now - last_log_time >= 60:  # every 60 seconds
            elapsed = now - t_start
            docs_done = doc_idx - start_idx + 1
            docs_per_sec = docs_done / max(elapsed, 1)
            remaining = total - doc_idx - 1
            eta_secs = remaining / max(docs_per_sec, 0.01)
            eta_h = eta_secs / 3600

            n_kk = lang_counts.get("kk", 0)
            n_ru = lang_counts.get("ru", 0)
            n_en = lang_counts.get("en", 0)

            logger.info(
                f"PROGRESS: {doc_idx+1:,}/{total:,} docs ({100*(doc_idx+1)/total:.1f}%) | "
                f"{total_sents:,} sents | "
                f"{docs_per_sec:.0f} docs/s | "
                f"ETA: {eta_h:.1f}h | "
                f"kk={n_kk:,} ru={n_ru:,} en={n_en:,}"
            )
            last_log_time = now

        # ---- Telegram notify ----
        if (doc_idx + 1) % TG_NOTIFY_EVERY == 0:
            n_kk_now = lang_counts.get("kk", 0)
            n_ru_now = lang_counts.get("ru", 0)
            pct = 100 * (doc_idx + 1) / total
            elapsed_now = time.time() - t_start
            docs_done_now = doc_idx - start_idx + 1
            speed = docs_done_now / max(elapsed_now, 1)
            eta = (total - doc_idx - 1) / max(speed, 0.01) / 3600
            tg_send(
                f"📊 Phase 1: {doc_idx+1:,}/{total:,} ({pct:.0f}%)\n"
                f"Sents: {total_sents:,}\n"
                f"kk={n_kk_now:,} ru={n_ru_now:,}\n"
                f"Speed: {speed:.0f} docs/s\n"
                f"ETA: {eta:.1f}h"
            )

        # ---- Checkpoint ----
        if (doc_idx + 1) % args.checkpoint_every == 0 and len(data["text"]) > 0:
            current_stats = {
                "languages": dict(lang_counts),
                "domains": dict(domain_counts),
                "empty_docs": empty_docs,
                "total_sents": total_sents,
            }
            save_checkpoint(data, doc_idx, current_stats)
            # Clear current batch (data saved to parquet)
            for key in data:
                data[key] = []

    # ---- Save final chunk if anything left ----
    if len(data["text"]) > 0:
        current_stats = {
            "languages": dict(lang_counts),
            "domains": dict(domain_counts),
            "empty_docs": empty_docs,
            "total_sents": total_sents,
        }
        save_checkpoint(data, total - 1, current_stats)

    # ---- Merge and upload (low memory — pyarrow on disk) ----
    logger.info("Merging checkpoints and uploading...")
    stats = merge_and_upload_checkpoints(HF_OUTPUT, HF_TOKEN)
    if stats is None:
        logger.error("No data to upload!")
        return

    n_sents = stats["total_rows"]
    n_kk = stats["n_kk"]
    n_ru = stats["n_ru"]
    n_en = stats["n_en"]

    elapsed_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"PHASE 1 COMPLETE in {elapsed_total/3600:.1f} hours")
    logger.info(f"  Documents: {doc_count:,} processed, {empty_docs:,} empty skipped")
    logger.info(f"  Sentences: {n_sents:,}")
    logger.info(f"  kk: {n_kk:,} ({100*n_kk/n_sents:.1f}%)")
    logger.info(f"  ru: {n_ru:,} ({100*n_ru/n_sents:.1f}%) ← already in kk dataset!")
    logger.info(f"  en: {n_en:,} ({100*n_en/n_sents:.1f}%)")
    logger.info(f"  Additional Russian needed: {max(0, n_kk - n_ru):,}")
    logger.info(f"  Languages: {stats['lang_counts']}")
    logger.info(f"  Domains: {stats['domain_counts']}")
    logger.info("=" * 70)

    tg_send(
        f"✅ Phase 1 DONE!\n"
        f"Docs: {doc_count:,}\n"
        f"Sents: {n_sents:,}\n"
        f"kk={n_kk:,} ({100*n_kk/n_sents:.0f}%)\n"
        f"ru={n_ru:,} ({100*n_ru/n_sents:.0f}%)\n"
        f"Need extra ru: {max(0, n_kk-n_ru):,}\n"
        f"Time: {elapsed_total/3600:.1f}h\n"
        f"Dataset: {HF_OUTPUT}"
    )

    # Save stats for Phase 2
    phase1_stats = {
        "total_docs": doc_count,
        "total_sents": n_sents,
        "kk_sents": n_kk,
        "ru_sents": n_ru,
        "en_sents": n_en,
        "additional_ru_needed": max(0, n_kk - n_ru),
        "domains": stats["domain_counts"],
        "languages": stats["lang_counts"],
    }
    stats_path = "/workspace/exp027/phase1_stats.json"
    with open(stats_path, "w") as f:
        json.dump(phase1_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
