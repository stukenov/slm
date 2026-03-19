#!/usr/bin/env python3
"""Unified cleaner for all Kazakh corpus sources.

Processes:
  1. kz-transformers/multidomain-kazakh-dataset (5 CSVs via streaming)
  2. collected/*.parquet (wave1)
  3. collected_wave2/*.parquet (wave2)

Filters (fast → slow):
  1. Fix OSCAR dict wrapper
  2. NFC + control chars + whitespace normalization
  3. Min length (30 chars)
  4. Kazakh char check (≥1 specific char)
  5. Script profile (cyr≥60%, lat≤25%)
  6. Junk (URL density, HTML tags, special chars)
  7. Gzip repetition (ratio ≥ 0.20)
  8. FastText LID (kk≥0.5, gap≥0.1)
  9. Exact dedup (MD5 across ALL sources)

Usage:
    # Sample mode (for iterative testing):
    python scripts/clean_all.py --sample 20 --seed 42

    # Full run + push to HF:
    python scripts/clean_all.py --full --push-to-hub saken-tukenov/sozkz-corpus-clean-v3

    # Process only specific sources:
    python scripts/clean_all.py --sample 10 --only oscar,mc4,kazparc
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import logging
import multiprocessing as mp
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Compiled regexes
# ──────────────────────────────────────────────

RE_OSCAR_DICT = re.compile(r"^\s*\{\s*['\"]text['\"]\s*:\s*['\"](.+)['\"]\s*\}\s*$", re.DOTALL)
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")
RE_KAZAKH_CHAR = re.compile(
    r"[\u04D8\u04D9\u0492\u0493\u049A\u049B\u04A2\u04A3"
    r"\u04E8\u04E9\u04B0\u04B1\u04AE\u04AF\u04BA\u04BB\u0406\u0456]"
)

# ──────────────────────────────────────────────
# Filter config (tunable thresholds)
# ──────────────────────────────────────────────

THRESHOLDS = {
    "min_length": 50,
    "min_words": 10,
    "max_chars": 50_000,
    "min_cyrillic": 0.60,
    "max_latin": 0.25,
    "max_url_density": 5.0,  # per 1K chars
    "max_html_tags": 5,
    "max_special_char_ratio": 0.40,
    "min_gzip_ratio": 0.20,
    "lid_min_confidence": 0.50,
    "lid_min_gap": 0.10,
    "boilerplate_patterns": [
        "cookie", "javascript", "subscribe", "newsletter",
        "terms of service", "privacy policy", "sign in",
        "log in", "copyright ©",
    ],
}

# ──────────────────────────────────────────────
# Multidomain CSV mapping
# ──────────────────────────────────────────────

MULTIDOMAIN_REPO = "kz-transformers/multidomain-kazakh-dataset"
MULTIDOMAIN_FILES = {
    "oscar.csv": "md_oscar",
    "kazakhNews.csv": "md_kazakhNews",
    "kazakhBooks.csv": "md_kazakhBooks",
    "leipzig.csv": "md_leipzig",
    # cc100.csv not found on HF repo
}

# ──────────────────────────────────────────────
# FastText LID (lazy load)
# ──────────────────────────────────────────────

_ft_model = None
FT_MODEL_PATH = "/root/slm/models/lid.176.bin"


def get_ft_model():
    global _ft_model
    if _ft_model is None:
        # numpy 2.x compat for fasttext
        import numpy as _np
        _orig_array = _np.array
        def _safe_array(*a, **kw):
            kw.pop("copy", None)
            return _orig_array(*a, **kw)
        _np.array = _safe_array

        import fasttext
        fasttext.FastText.eprint = lambda x: None
        _ft_model = fasttext.load_model(FT_MODEL_PATH)
    return _ft_model


def ft_predict(text: str) -> dict[str, float]:
    model = get_ft_model()
    line = text.replace("\n", " ")[:5000]
    result = model.predict(line, k=3)
    labels = result[0]
    scores = list(result[1]) if hasattr(result[1], "__iter__") else [result[1]]
    return {l.replace("__label__", ""): float(s) for l, s in zip(labels, scores)}


# ──────────────────────────────────────────────
# Filter functions
# ──────────────────────────────────────────────


RE_OSCAR_MULTI_DICT = re.compile(
    r"\{\s*['\"]text['\"]\s*:\s*['\"](.+?)['\"](?:\s*,\s*['\"]text_idx['\"]\s*:\s*['\"][^'\"]*['\"])?\s*\}",
    re.DOTALL,
)


def fix_oscar_wrapper(text: str) -> str:
    """Fix OSCAR texts wrapped in Python dict literal(s): {'text': '...'}"""
    if "'text'" not in text[:100] and '"text"' not in text[:100]:
        return text
    # Try single dict parse first
    if text.lstrip().startswith("{"):
        try:
            parsed = ast.literal_eval(text.strip())
            if isinstance(parsed, dict) and "text" in parsed:
                return str(parsed["text"])
        except (ValueError, SyntaxError):
            pass
    # Try extracting multiple {'text': '...'} dicts (moscar format)
    matches = RE_OSCAR_MULTI_DICT.findall(text)
    if matches:
        return "\n".join(matches)
    return text


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()
    return text


def _split_by_separator(text: str, sep: str, max_chars: int) -> list[str]:
    """Split text into chunks of ~max_chars using given separator."""
    chunks = []
    parts = text.split(sep)
    current = []
    current_len = 0
    sep_len = len(sep)

    for part in parts:
        part_len = len(part) + sep_len
        if current_len + part_len > max_chars and current:
            chunks.append(sep.join(current).strip())
            current = [part]
            current_len = part_len
        else:
            current.append(part)
            current_len += part_len

    if current:
        chunks.append(sep.join(current).strip())

    return chunks


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of ~max_chars by paragraph/line/sentence boundaries.
    Returns list of chunks. Short texts return as single-element list.
    """
    if len(text) <= max_chars:
        return [text]

    # Try splitting by double newline first
    chunks = _split_by_separator(text, "\n\n", max_chars)

    # If any chunk still too long, split by single newline
    result = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            result.extend(_split_by_separator(chunk, "\n", max_chars))

    # If still too long, split by sentence (". ")
    final = []
    for chunk in result:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(_split_by_separator(chunk, ". ", max_chars))

    # Last resort: hard split
    truly_final = []
    for chunk in final:
        if len(chunk) <= max_chars:
            truly_final.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                truly_final.append(chunk[i:i + max_chars])

    return [c for c in truly_final if len(c) >= THRESHOLDS["min_length"]]


def has_kazakh_chars(text: str) -> bool:
    return bool(RE_KAZAKH_CHAR.search(text))


def script_profile_ok(text: str) -> bool:
    counts = Counter()
    for ch in text:
        if ch.isspace():
            continue
        if "\u0400" <= ch <= "\u04ff" or "\u0500" <= ch <= "\u052f":
            counts["cyr"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["lat"] += 1
        else:
            counts["oth"] += 1
    total = sum(counts.values()) or 1
    cyr_frac = counts["cyr"] / total
    lat_frac = counts["lat"] / total
    return cyr_frac >= THRESHOLDS["min_cyrillic"] and lat_frac <= THRESHOLDS["max_latin"]


def junk_ok(text: str) -> bool:
    text_len = len(text) or 1
    # URL density
    urls = RE_URL.findall(text)
    if len(urls) / (text_len / 1000) > THRESHOLDS["max_url_density"]:
        return False
    # HTML tags
    if len(RE_HTML_TAG.findall(text)) > THRESHOLDS["max_html_tags"]:
        return False
    # Special char ratio
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special / text_len > THRESHOLDS["max_special_char_ratio"]:
        return False
    # Boilerplate
    text_lower = text.lower()
    for pat in THRESHOLDS["boilerplate_patterns"]:
        if pat in text_lower:
            return False
    return True


def gzip_ok(text: str) -> bool:
    encoded = text.encode("utf-8")
    if len(encoded) == 0:
        return True
    compressed = gzip.compress(encoded, compresslevel=6)
    return len(compressed) / len(encoded) >= THRESHOLDS["min_gzip_ratio"]


def lid_ok(text: str) -> bool:
    lid = ft_predict(text)
    kk = lid.get("kk", 0.0)
    if kk < THRESHOLDS["lid_min_confidence"]:
        return False
    for rival in ("ru", "en", "ba", "tr", "ky"):
        if kk - lid.get(rival, 0.0) < THRESHOLDS["lid_min_gap"]:
            return False
    return True


# ──────────────────────────────────────────────
# Pipeline: apply all filters to one text
# ──────────────────────────────────────────────


def _filter_text(text: str, use_lid: bool) -> tuple[str | None, str]:
    """Apply filters 3-8 to a single text chunk.
    Returns (text, 'ok') or (None, reason).
    """
    # 3. Min length + min words
    if len(text) < THRESHOLDS["min_length"]:
        return None, "too_short"
    if len(text.split()) < THRESHOLDS["min_words"]:
        return None, "too_few_words"

    # 4. Kazakh chars
    if not has_kazakh_chars(text):
        return None, "no_kaz_chars"

    # 5. Script profile
    if not script_profile_ok(text):
        return None, "script_profile"

    # 6. Junk
    if not junk_ok(text):
        return None, "junk"

    # 7. Gzip repetition
    if not gzip_ok(text):
        return None, "gzip_repetition"

    # 8. FastText LID
    if use_lid:
        if not lid_ok(text):
            return None, "lid_rejected"

    return text, "ok"


def clean_one(text: str, source: str, use_lid: bool = True) -> list[tuple[str | None, str]]:
    """Clean a single text. Returns list of (cleaned_text, reason) tuples.
    Long texts are split into chunks, each filtered independently.
    """
    if not text:
        return [(None, "empty")]

    # 1. Fix OSCAR wrapper
    text = fix_oscar_wrapper(text)

    # 2. Normalize
    text = normalize(text)

    # Split long texts into chunks (books, etc.)
    chunks = split_long_text(text, THRESHOLDS["max_chars"])

    results = []
    for chunk in chunks:
        results.append(_filter_text(chunk, use_lid))

    return results if results else [(None, "too_short")]


# ──────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────


def _download_multidomain_csvs(cache_dir: str = "/root/slm/data/multidomain_cache"):
    """Download multidomain CSVs to local disk if not already cached."""
    from huggingface_hub import hf_hub_download
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for filename, source in MULTIDOMAIN_FILES.items():
        local = os.path.join(cache_dir, filename)
        if os.path.exists(local):
            logger.info("[%s] Using cached %s", source, local)
        else:
            logger.info("[%s] Downloading %s/%s ...", source, MULTIDOMAIN_REPO, filename)
            downloaded = hf_hub_download(
                repo_id=MULTIDOMAIN_REPO, filename=filename,
                repo_type="dataset", local_dir=cache_dir,
            )
            local = downloaded
            logger.info("[%s] Downloaded to %s", source, local)
        paths[source] = local
    return paths


def load_multidomain(only_sources: set[str] | None = None):
    """Yield (text, source) from locally cached multidomain CSV files."""
    import csv
    csv.field_size_limit(10_000_000)  # some texts are very large (books)

    cached = _download_multidomain_csvs()
    for source, filepath in cached.items():
        if only_sources and source not in only_sources:
            continue
        logger.info("[%s] Reading local CSV %s ...", source, filepath)
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row.get("text", "") or "", source
        except Exception as e:
            logger.error("[%s] Failed to load: %s", source, e)


def load_parquets(directory: str, only_sources: set[str] | None = None):
    """Yield (text, source) from parquet files in a directory."""
    import pyarrow.parquet as pq

    dirpath = Path(directory)
    if not dirpath.exists():
        logger.warning("Directory %s not found, skipping", directory)
        return

    for f in sorted(dirpath.glob("*.parquet")):
        source = f.stem
        if source == "hplt":  # corrupted file, use hplt_new instead
            continue
        if only_sources and source not in only_sources:
            continue
        logger.info("[%s] Loading %s ...", source, f)
        try:
            table = pq.read_table(str(f), columns=["text"])
            for i in range(len(table)):
                yield table.column("text")[i].as_py() or "", source
        except Exception as e:
            logger.error("[%s] Failed: %s", source, e)


def load_all_sources(only_sources: set[str] | None = None):
    """Yield (text, source) from all data sources."""
    # 1. Multidomain
    yield from load_multidomain(only_sources)
    # 2. Wave 1
    yield from load_parquets("/root/slm/data/collected", only_sources)
    # 3. Wave 2
    yield from load_parquets("/root/slm/data/collected_wave2", only_sources)


# ──────────────────────────────────────────────
# Sample mode: pick N random texts per source
# ──────────────────────────────────────────────


def sample_from_parquets(n_per_source: int, seed: int, only_sources: set[str] | None = None):
    """Fast random sampling from parquet files (no streaming)."""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    samples = []

    for d in ["/root/slm/data/collected", "/root/slm/data/collected_wave2"]:
        dirpath = Path(d)
        if not dirpath.exists():
            continue
        for f in sorted(dirpath.glob("*.parquet")):
            source = f.stem
            if source == "hplt":  # corrupted file
                continue
            if only_sources and source not in only_sources:
                continue
            try:
                table = pq.read_table(str(f), columns=["text"])
                n = len(table)
                k = min(n_per_source, n)
                idxs = rng.choice(n, size=k, replace=False)
                for i in idxs:
                    text = table.column("text")[int(i)].as_py() or ""
                    samples.append((text, source))
                logger.info("[%s] sampled %d / %d", source, k, n)
            except Exception as e:
                logger.error("[%s] Failed: %s", source, e)

    return samples


def sample_from_multidomain(n_per_source: int, seed: int, only_sources: set[str] | None = None):
    """Sample from multidomain by reading first N*50 rows (fast) and picking random N."""
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    samples = []
    read_limit = max(n_per_source * 50, 1000)

    for filename, source in MULTIDOMAIN_FILES.items():
        if only_sources and source not in only_sources:
            continue
        url = f"https://huggingface.co/datasets/{MULTIDOMAIN_REPO}/resolve/main/{filename}"
        logger.info("[%s] Sampling %d from first %d rows ...", source, n_per_source, read_limit)
        try:
            ds = load_dataset("csv", data_files=url, split="train", streaming=True)
            buf = []
            for i, row in enumerate(ds):
                if i >= read_limit:
                    break
                buf.append(row.get("text", "") or "")
            if buf:
                k = min(n_per_source, len(buf))
                idxs = rng.choice(len(buf), size=k, replace=False)
                for i in idxs:
                    samples.append((buf[int(i)], source))
                logger.info("[%s] sampled %d from %d buffered", source, k, len(buf))
        except Exception as e:
            logger.error("[%s] Failed: %s", source, e)

    return samples


def run_sample(n: int, seed: int, only_sources: set[str] | None, use_lid: bool):
    """Sample N texts total, run pipeline, print detailed results."""
    logger.info("=== SAMPLE MODE: %d texts ===", n)

    # Calculate per-source budget
    all_source_names = list(MULTIDOMAIN_FILES.values()) + [
        "culturax", "hplt", "hplt_new", "madlad400", "mc4", "moscar", "wikipedia",
        "belebele", "cc100", "kazparc", "kazparc_sync", "kazsandra", "sib200", "wikiann",
    ]
    if only_sources:
        all_source_names = [s for s in all_source_names if s in only_sources]
    n_sources = len(all_source_names) or 1
    n_per = max(2, n // n_sources + 1)

    # Sample from parquets only (fast); multidomain is too slow for sampling
    samples = sample_from_parquets(n_per, seed, only_sources)
    # Optionally add multidomain (very slow, skip by default in sample mode)
    if os.environ.get("SAMPLE_MULTIDOMAIN"):
        samples += sample_from_multidomain(n_per, seed, only_sources)

    # Shuffle and trim to n
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    samples = samples[:n]
    logger.info("Sampled %d texts from %d sources", len(samples), len(set(s for _, s in samples)))

    stats = defaultdict(int)
    passed = []
    rejected = []

    for text, source in samples:
        results = clean_one(text, source, use_lid=use_lid)
        for cleaned, reason in results:
            stats[reason] += 1
            if cleaned:
                passed.append((source, cleaned))
            else:
                rejected.append((source, text[:300], reason))

    # Print rejected
    print("\n" + "=" * 70)
    print("REJECTED TEXTS")
    print("=" * 70)
    for i, (src, preview, reason) in enumerate(rejected):
        print(f"\n--- Rejected #{i+1} [{src}] reason={reason} ---")
        print(preview.replace("\n", "\\n"))

    # Print passed
    print("\n" + "=" * 70)
    print("PASSED TEXTS")
    print("=" * 70)
    for i, (src, text) in enumerate(passed):
        preview = text[:300].replace("\n", "\\n")
        if len(text) > 300:
            preview += "..."
        print(f"\n--- Passed #{i+1} [{src}] len={len(text)} ---")
        print(preview)

    # Stats
    print("\n" + "=" * 70)
    print("STATS")
    print("=" * 70)
    for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {reason:20s}: {count}")
    print(f"  {'TOTAL':20s}: {len(samples)}")


# ──────────────────────────────────────────────
# Full mode: process everything, dedup, push
# ──────────────────────────────────────────────


def _worker_init(use_lid_flag: bool):
    """Initialize FastText model in each worker process."""
    global _worker_use_lid
    _worker_use_lid = use_lid_flag
    if use_lid_flag:
        get_ft_model()


def _worker_process(batch: list[tuple[str, str]]) -> list[tuple[str | None, str, str]]:
    """Process a batch of (text, source) in worker. Returns [(cleaned, reason, source), ...]."""
    results = []
    for text, source in batch:
        for cleaned, reason in clean_one(text, source, use_lid=_worker_use_lid):
            results.append((cleaned, reason, source))
    return results


BATCH_SIZE = 5000
NUM_WORKERS = 8  # limited to avoid OOM (FastText model ~2GB per worker)


def run_full(push_repo: str | None, only_sources: set[str] | None, use_lid: bool):
    """Full processing pipeline with dedup and optional HF push."""
    from datasets import Dataset, DatasetDict

    n_workers = NUM_WORKERS or max(1, mp.cpu_count())
    logger.info("=== FULL MODE === (workers=%d, batch=%d)", n_workers, BATCH_SIZE)

    # Pre-load FastText in main process before fork (copy-on-write sharing)
    if use_lid:
        get_ft_model()

    stats = defaultdict(lambda: defaultdict(int))
    seen_hashes: set[str] = set()
    clean_texts: list[str] = []
    clean_sources: list[str] = []
    total = 0
    dedup_count = 0

    pool = mp.Pool(n_workers, initializer=_worker_init, initargs=(use_lid,))
    batch: list[tuple[str, str]] = []

    def flush_batch(batch: list[tuple[str, str]]):
        nonlocal total, dedup_count
        if not batch:
            return
        # Count raw per source
        for _, source in batch:
            stats[source]["raw"] += 1
        # Split batch into sub-batches for workers
        sub_size = max(1, len(batch) // n_workers)
        sub_batches = [batch[i:i + sub_size] for i in range(0, len(batch), sub_size)]
        for worker_results in pool.imap_unordered(_worker_process, sub_batches):
            for cleaned, reason, source in worker_results:
                if cleaned is None:
                    stats[source][reason] += 1
                    continue
                # 9. Exact dedup (MD5)
                h = hashlib.md5(cleaned.encode()).hexdigest()
                if h in seen_hashes:
                    stats[source]["dedup"] += 1
                    dedup_count += 1
                    continue
                seen_hashes.add(h)
                clean_texts.append(cleaned)
                clean_sources.append(source)
                stats[source]["ok"] += 1

    for text, source in load_all_sources(only_sources):
        total += 1
        batch.append((text, source))
        if len(batch) >= BATCH_SIZE:
            flush_batch(batch)
            batch = []
            if total % 100_000 == 0:
                logger.info(
                    "Processed %d, kept %d, dedup_removed %d",
                    total, len(clean_texts), dedup_count,
                )

    flush_batch(batch)
    pool.close()
    pool.join()
    logger.info("Processed %d total, kept %d, dedup_removed %d", total, len(clean_texts), dedup_count)

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("CLEANING REPORT")
    logger.info("=" * 70)

    for source in sorted(stats.keys()):
        logger.info("--- %s ---", source)
        for reason in sorted(stats[source].keys()):
            logger.info("  %-20s: %8d", reason, stats[source][reason])

    total_raw = sum(s["raw"] for s in stats.values())
    logger.info("")
    logger.info("TOTAL: %d raw -> %d clean (dedup removed %d)", total_raw, len(clean_texts), dedup_count)

    if not clean_texts:
        logger.warning("No texts passed cleaning!")
        return

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(clean_texts))
    clean_texts = [clean_texts[i] for i in perm]
    clean_sources = [clean_sources[i] for i in perm]

    # Train/val split (1% val)
    n_val = max(1, int(len(clean_texts) * 0.01))
    n_train = len(clean_texts) - n_val

    train_ds = Dataset.from_dict({
        "text": clean_texts[:n_train],
        "source": clean_sources[:n_train],
    })
    val_ds = Dataset.from_dict({
        "text": clean_texts[n_train:],
        "source": clean_sources[n_train:],
    })
    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    logger.info("Train: %d, Validation: %d", len(train_ds), len(val_ds))

    if push_repo:
        logger.info("Pushing to %s ...", push_repo)
        ds_dict.push_to_hub(push_repo, private=False)
        logger.info("Published: https://huggingface.co/datasets/%s", push_repo)
    else:
        save_path = "/root/slm/data/clean_all_output"
        os.makedirs(save_path, exist_ok=True)
        ds_dict.save_to_disk(save_path)
        logger.info("Saved to %s", save_path)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Unified Kazakh corpus cleaner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int, help="Sample N random texts and show results")
    group.add_argument("--full", action="store_true", help="Full run on all data")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push results")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated source names to process")
    parser.add_argument("--no-lid", action="store_true", help="Skip FastText LID (faster)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    only_sources = set(args.only.split(",")) if args.only else None
    use_lid = not args.no_lid

    if use_lid and not os.path.exists(FT_MODEL_PATH):
        logger.warning("FastText model not found at %s, disabling LID", FT_MODEL_PATH)
        use_lid = False

    if args.sample:
        run_sample(args.sample, args.seed, only_sources, use_lid)
    else:
        run_full(args.push_to_hub, only_sources, use_lid)


if __name__ == "__main__":
    main()
