"""Collect wave2 Kazakh datasets — no dedup, just download/clean/save/push."""

import logging
import lzma
import os
import sys
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slm.collect.pipeline import clean_text, iter_source
from slm.collect.sources import WAVE2

from datasets import Dataset

OUTPUT_DIR = "/root/slm/data/collected_wave2"
HUB_REPO = "saken-tukenov/sozkz-corpus-dedup-kk-web-v2"
CLEAN_CFG = {
    "min_text_len": 20,  # lower threshold for short sentences
    "max_chars": 500_000,
    "max_url_density": 5.0,
    "max_html_tags": 5,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_texts = []
all_sources = []


def collect_cc100():
    """Download CC-100 Kazakh from statmt.org (deprecated HF loader)."""
    logger.info("=== Collecting cc100 (manual download) ===")
    url = "https://data.statmt.org/cc-100/kk.txt.xz"
    xz_path = os.path.join(OUTPUT_DIR, "kk.txt.xz")

    if not os.path.exists(xz_path):
        logger.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, xz_path)
        logger.info("Downloaded %.1f MB", os.path.getsize(xz_path) / 1e6)
    else:
        logger.info("Using cached %s", xz_path)

    texts = []
    count = 0
    filtered = 0
    current_doc = []

    logger.info("Decompressing and processing...")
    with lzma.open(xz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                # Empty line = document boundary in CC-100
                if current_doc:
                    text = "\n".join(current_doc)
                    count += 1
                    cleaned = clean_text(text, CLEAN_CFG)
                    if cleaned:
                        texts.append(cleaned)
                    else:
                        filtered += 1
                    current_doc = []
                    if count % 100_000 == 0:
                        logger.info("[cc100] Processed %d docs, kept %d", count, len(texts))
            else:
                current_doc.append(line)
        # Last doc
        if current_doc:
            text = "\n".join(current_doc)
            count += 1
            cleaned = clean_text(text, CLEAN_CFG)
            if cleaned:
                texts.append(cleaned)
            else:
                filtered += 1

    logger.info("[cc100] Done: %d raw -> %d kept (filtered=%d)", count, len(texts), filtered)

    if texts:
        ds = Dataset.from_dict({"text": texts, "source": ["cc100"] * len(texts)})
        out_path = os.path.join(OUTPUT_DIR, "cc100.parquet")
        ds.to_parquet(out_path)
        logger.info("[cc100] Saved %d texts to %s", len(texts), out_path)

    return texts


# Step 1: CC-100 manual download
cc100_texts = collect_cc100()
if cc100_texts:
    all_texts.extend(cc100_texts)
    all_sources.extend(["cc100"] * len(cc100_texts))

# Step 2: HF datasets from WAVE2
for src in WAVE2:
    logger.info("=== Collecting %s ===", src.name)
    texts = []
    count = 0
    filtered = 0

    try:
        for text in iter_source(src):
            count += 1
            if count % 100_000 == 0:
                logger.info("[%s] Processed %d, kept %d", src.name, count, len(texts))

            cleaned = clean_text(text, CLEAN_CFG)
            if cleaned is None:
                filtered += 1
                continue
            texts.append(cleaned)
    except Exception as e:
        logger.error("[%s] Error after %d rows: %s", src.name, count, e)

    logger.info("[%s] Done: %d raw -> %d kept (filtered=%d)", src.name, count, len(texts), filtered)

    if texts:
        ds = Dataset.from_dict({"text": texts, "source": [src.name] * len(texts)})
        out_path = os.path.join(OUTPUT_DIR, f"{src.name}.parquet")
        ds.to_parquet(out_path)
        logger.info("[%s] Saved %d texts to %s", src.name, len(texts), out_path)
        all_texts.extend(texts)
        all_sources.extend([src.name] * len(texts))

logger.info("")
logger.info("=== TOTAL: %d texts from %d sources ===", len(all_texts), len(set(all_sources)))

if all_texts:
    logger.info("Pushing %d texts to %s ...", len(all_texts), HUB_REPO)
    merged = Dataset.from_dict({"text": all_texts, "source": all_sources})
    merged.push_to_hub(HUB_REPO, private=False)
    logger.info("Done! https://huggingface.co/datasets/%s", HUB_REPO)

    from collections import Counter
    logger.info("")
    logger.info("=== PER-SOURCE BREAKDOWN ===")
    for src_name, cnt in Counter(all_sources).most_common():
        logger.info("  %-20s: %d", src_name, cnt)
