#!/usr/bin/env python3
"""Filter foreign/garbage characters from Kazakh text datasets.

Loads text datasets, removes texts containing CJK, Thai, Korean, Devanagari,
and other non-Kazakh scripts. Supports iterative quality checking with random
samples before committing to the full filter.

Datasets:
  1. sozkz-corpus-clean-kk-text-v2         (column: "text")
  2. sozkz-corpus-clean-enkk-fineweb-edu-v1 (columns: "text_en", "text_kk")

Usage:
    # Step 1: Sample and inspect (dry run)
    python scripts/data/filter_foreign_chars.py --sample 50

    # Step 2: Run full filter and push
    python scripts/data/filter_foreign_chars.py --run \
        --output-kk stukenov/sozkz-corpus-clean-kk-text-v4 \
        --output-enkk stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v2
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
import unicodedata
from collections import Counter

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Unicode ranges for foreign scripts we want to REJECT
# ============================================================

# CJK (Chinese, Japanese Kanji)
# Note: \u only handles 4 hex digits. For supplementary planes use \U with 8 digits.
# We skip Extension B/C/D (supplementary plane, extremely rare in web text).
RE_CJK = re.compile(
    "[\u4e00-\u9fff"        # CJK Unified Ideographs
    "\u3400-\u4dbf"         # CJK Extension A
    "\uf900-\ufaff"         # CJK Compatibility Ideographs
    "\u3000-\u303f"         # CJK Symbols and Punctuation
    "]"
)

# Japanese Kana
RE_JAPANESE = re.compile(
    r"[\u3040-\u309f"       # Hiragana
    r"\u30a0-\u30ff"        # Katakana
    r"\u31f0-\u31ff"        # Katakana Phonetic Extensions
    r"]"
)

# Korean Hangul
RE_KOREAN = re.compile(
    r"[\uac00-\ud7af"       # Hangul Syllables
    r"\u1100-\u11ff"        # Hangul Jamo
    r"\u3130-\u318f"        # Hangul Compatibility Jamo
    r"]"
)

# Thai
RE_THAI = re.compile(r"[\u0e00-\u0e7f]")

# Devanagari (Hindi, Sanskrit, etc.)
RE_DEVANAGARI = re.compile(r"[\u0900-\u097f]")

# Bengali
RE_BENGALI = re.compile(r"[\u0980-\u09ff]")

# Tamil
RE_TAMIL = re.compile(r"[\u0b80-\u0bff]")

# Telugu
RE_TELUGU = re.compile(r"[\u0c00-\u0c7f]")

# Georgian
RE_GEORGIAN = re.compile(r"[\u10a0-\u10ff]")

# Armenian
RE_ARMENIAN = re.compile(r"[\u0530-\u058f]")

# Ethiopic
RE_ETHIOPIC = re.compile(r"[\u1200-\u137f]")

# Myanmar
RE_MYANMAR = re.compile(r"[\u1000-\u109f]")

# Khmer
RE_KHMER = re.compile(r"[\u1780-\u17ff]")

# Sinhala
RE_SINHALA = re.compile(r"[\u0d80-\u0dff]")

# Tibetan
RE_TIBETAN = re.compile(r"[\u0f00-\u0fff]")

# Hebrew (small amounts OK in academic texts, but bulk = wrong language)
RE_HEBREW = re.compile(r"[\u0590-\u05ff]")

# All foreign script patterns with names for logging
FOREIGN_PATTERNS = [
    ("CJK", RE_CJK),
    ("Japanese", RE_JAPANESE),
    ("Korean", RE_KOREAN),
    ("Thai", RE_THAI),
    ("Devanagari", RE_DEVANAGARI),
    ("Bengali", RE_BENGALI),
    ("Tamil", RE_TAMIL),
    ("Telugu", RE_TELUGU),
    ("Georgian", RE_GEORGIAN),
    ("Armenian", RE_ARMENIAN),
    ("Ethiopic", RE_ETHIOPIC),
    ("Myanmar", RE_MYANMAR),
    ("Khmer", RE_KHMER),
    ("Sinhala", RE_SINHALA),
    ("Tibetan", RE_TIBETAN),
    ("Hebrew", RE_HEBREW),
]

# ============================================================
# Threshold-based filtering
# ============================================================

# Min number of foreign chars to trigger rejection.
# A single stray char (copy-paste artifact) is OK — we care about
# actual foreign-language content or heavy contamination.
MIN_FOREIGN_CHARS = 3

# Max ratio of foreign chars to total non-space chars.
# Even with MIN_FOREIGN_CHARS=3, a text of 10000 chars with 3 CJK
# is fine (0.03%). We reject if foreign chars are a significant chunk.
MAX_FOREIGN_RATIO = 0.005  # 0.5% — catches mixed-language texts

# Additional: reject texts that are mostly non-Cyrillic non-Latin non-digit
# (catches garbage Unicode, emojis spam, etc.)
MIN_USEFUL_RATIO = 0.70  # at least 70% of chars should be Cyrillic/Latin/digit/punct

# Max Latin letter ratio — catches machine-translated code/English, WordPress junk, spam
# 5% is aggressive but keeps data very clean: removes translated code, API docs,
# WordPress boilerplate, and garbled machine translations with English remnants.
MAX_LATIN_RATIO = 0.05

# Encoding artifacts: suspicious patterns that indicate broken encoding
# e.g. "Астаналы?тарды" where ? replaces Kazakh chars like Қ, Ғ, Ұ
RE_REPLACEMENT_CHAR = re.compile("\ufffd")  # Unicode replacement character

# Pattern: Cyrillic letter, then ?, then Cyrillic letter — broken Kazakh char
RE_BROKEN_KAZAKH = re.compile(r"[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ]\?[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІ]")

# Max allowed ? density in text (legitimate texts rarely have many ?)
MAX_QUESTION_MARK_RATIO = 0.08  # 8% — only catches severe encoding issues/spam

# Repetition filter: catches ►►►, ▲▲▲, ===, ---  and similar garbage
MAX_REPEATED_CHAR_RATIO = 0.15  # 15% of non-space chars are same repeated char
MIN_UNIQUE_CHAR_RATIO = 0.05   # at least 5% unique chars (catches ▲▲▲▲▲▲...)


def classify_text(text: str) -> tuple[bool, str]:
    """Check if text should be kept.

    Returns (keep, reason). If keep=False, reason explains why.
    """
    if not text or len(text.strip()) < 10:
        return False, "too_short"

    # Check for repetitive garbage (►►►, ▲▲▲, ===, etc.)
    non_space_chars = [c for c in text if not c.isspace()]
    if len(non_space_chars) > 20:
        char_counts = Counter(non_space_chars)
        most_common_char, most_common_count = char_counts.most_common(1)[0]
        if most_common_count / len(non_space_chars) > MAX_REPEATED_CHAR_RATIO:
            return False, f"repeated_char('{most_common_char}'={most_common_count}, ratio={most_common_count/len(non_space_chars):.3f})"
        unique_ratio = len(char_counts) / len(non_space_chars)
        if unique_ratio < MIN_UNIQUE_CHAR_RATIO:
            return False, f"low_unique_chars(unique={len(char_counts)}, ratio={unique_ratio:.4f})"

    # Check for Unicode replacement characters
    replacement_count = len(RE_REPLACEMENT_CHAR.findall(text))
    if replacement_count >= 3:
        return False, f"replacement_chars(count={replacement_count})"

    # Check for broken encoding: Cyrillic?Cyrillic pattern
    broken_count = len(RE_BROKEN_KAZAKH.findall(text))
    if broken_count >= 2:
        return False, f"broken_encoding(patterns={broken_count})"

    # Check question mark density (catches subtler encoding issues)
    qmark_count = text.count("?")
    non_space = sum(1 for c in text if not c.isspace())
    if non_space > 0 and qmark_count >= 5:
        ratio = qmark_count / non_space
        if ratio >= MAX_QUESTION_MARK_RATIO:
            return False, f"high_qmark_ratio(count={qmark_count}, ratio={ratio:.4f})"

    # Count foreign chars
    foreign_counts = Counter()
    for name, pattern in FOREIGN_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            foreign_counts[name] = len(matches)

    total_foreign = sum(foreign_counts.values())

    if total_foreign >= MIN_FOREIGN_CHARS:
        # Check ratio
        non_space = sum(1 for c in text if not c.isspace())
        if non_space == 0:
            return False, "empty"
        ratio = total_foreign / non_space
        if ratio >= MAX_FOREIGN_RATIO:
            top_scripts = sorted(foreign_counts.items(), key=lambda x: -x[1])[:3]
            scripts_str = ", ".join(f"{n}={c}" for n, c in top_scripts)
            return False, f"foreign_chars({scripts_str}, ratio={ratio:.4f})"

    # Check useful char ratio (Cyrillic + Latin + digits + common punct)
    useful = 0
    total_chars = 0
    for ch in text:
        if ch.isspace():
            continue
        total_chars += 1
        # Cyrillic (including Kazakh-specific)
        if "\u0400" <= ch <= "\u04ff" or "\u0500" <= ch <= "\u052f":
            useful += 1
        # Latin
        elif ch.isascii() and ch.isalpha():
            useful += 1
        # Digits
        elif ch.isdigit():
            useful += 1
        # Common punctuation
        elif unicodedata.category(ch).startswith(("P", "S")):
            useful += 1

    if total_chars > 0 and useful / total_chars < MIN_USEFUL_RATIO:
        return False, f"low_useful_ratio({useful/total_chars:.3f})"

    # Check Latin letter ratio
    latin_count = sum(1 for c in text if c.isascii() and c.isalpha() and not c.isspace())
    if total_chars > 0 and latin_count / total_chars > MAX_LATIN_RATIO:
        return False, f"high_latin_ratio({latin_count/total_chars:.3f})"

    return True, "ok"


# ============================================================
# Sampling and inspection
# ============================================================


def sample_and_inspect(dataset_name: str, text_column: str, n_samples: int, seed: int):
    """Load dataset, run filter on random sample, show rejected texts."""
    logger.info("Loading %s...", dataset_name)
    ds = load_dataset(dataset_name, split="train", streaming=True)

    # Collect a pool of texts to sample from
    pool_size = min(n_samples * 100, 500_000)
    logger.info("Collecting pool of %d texts...", pool_size)
    pool = []
    for i, row in enumerate(ds):
        if i >= pool_size:
            break
        text = row.get(text_column, "")
        if text:
            pool.append(text)

    logger.info("Pool: %d texts. Sampling %d...", len(pool), n_samples)
    random.seed(seed)
    samples = random.sample(pool, min(n_samples, len(pool)))

    kept = 0
    rejected = 0
    reject_reasons = Counter()
    rejected_examples = []

    for text in samples:
        keep, reason = classify_text(text)
        if keep:
            kept += 1
        else:
            rejected += 1
            reject_reasons[reason.split("(")[0]] += 1
            rejected_examples.append((text, reason))

    print(f"\n{'='*70}")
    print(f"SAMPLE RESULTS: {dataset_name}")
    print(f"{'='*70}")
    print(f"  Sampled: {len(samples)}")
    print(f"  Kept:    {kept} ({100*kept/len(samples):.1f}%)")
    print(f"  Rejected: {rejected} ({100*rejected/len(samples):.1f}%)")
    print(f"\n  Rejection reasons:")
    for reason, count in reject_reasons.most_common():
        print(f"    {reason}: {count}")

    if rejected_examples:
        print(f"\n{'='*70}")
        print("REJECTED EXAMPLES (first 200 chars):")
        print(f"{'='*70}")
        for text, reason in rejected_examples[:20]:
            snippet = text[:200].replace("\n", "\\n")
            print(f"\n  [{reason}]")
            print(f"  {snippet}")

    # Also show some KEPT examples to verify we're not over-filtering
    kept_examples = [(t, r) for t, r in zip(samples, [classify_text(t) for t in samples]) if r[0]]
    if kept_examples:
        print(f"\n{'='*70}")
        print("KEPT EXAMPLES (random 5, first 200 chars):")
        print(f"{'='*70}")
        random.shuffle(kept_examples)
        for text, _ in kept_examples[:5]:
            snippet = text[:200].replace("\n", "\\n")
            print(f"\n  {snippet}")

    return kept, rejected


def estimate_full_dataset(dataset_name: str, text_column: str, n_estimate: int = 10_000):
    """Estimate rejection rate on a larger sample."""
    logger.info("Estimating rejection rate on %d texts from %s...", n_estimate, dataset_name)
    ds = load_dataset(dataset_name, split="train", streaming=True)

    kept = 0
    rejected = 0
    reject_reasons = Counter()

    for i, row in enumerate(ds):
        if i >= n_estimate:
            break
        text = row.get(text_column, "")
        if not text:
            continue
        keep, reason = classify_text(text)
        if keep:
            kept += 1
        else:
            rejected += 1
            reject_reasons[reason.split("(")[0]] += 1

    total = kept + rejected
    print(f"\n{'='*70}")
    print(f"ESTIMATE: {dataset_name} ({total} texts scanned)")
    print(f"{'='*70}")
    print(f"  Keep rate: {100*kept/total:.2f}%")
    print(f"  Reject rate: {100*rejected/total:.2f}%")
    print(f"  Rejection breakdown:")
    for reason, count in reject_reasons.most_common():
        print(f"    {reason}: {count} ({100*count/total:.2f}%)")

    return kept, rejected


# ============================================================
# Full filter + push
# ============================================================


def run_full_filter(
    dataset_name: str,
    text_column: str,
    output_repo: str,
    num_proc: int = 16,
):
    """Filter dataset and push to HF Hub."""
    logger.info("Loading full dataset: %s", dataset_name)
    ds = load_dataset(dataset_name, verification_mode="no_checks")

    if "train" in ds:
        data = ds["train"]
    else:
        data = next(iter(ds.values()))

    original_size = len(data)
    logger.info("Original size: %d", original_size)

    # For multi-column datasets (fineweb-edu), check text_kk
    cols_to_check = [c.strip() for c in text_column.split(",")]

    def filter_fn(examples):
        results = []
        for i in range(len(examples[cols_to_check[0]])):
            keep = True
            for col in cols_to_check:
                text = examples[col][i]
                if text:
                    k, _ = classify_text(text)
                    if not k:
                        keep = False
                        break
            results.append(keep)
        return results

    logger.info("Filtering with %d workers...", num_proc)
    t0 = time.time()
    filtered = data.filter(filter_fn, batched=True, batch_size=1000, num_proc=num_proc)
    elapsed = time.time() - t0

    new_size = len(filtered)
    removed = original_size - new_size
    logger.info(
        "Filtered: %d → %d (removed %d, %.2f%%) in %.0fs",
        original_size, new_size, removed, 100 * removed / original_size, elapsed,
    )

    logger.info("Pushing to %s...", output_repo)
    filtered.push_to_hub(output_repo, private=False)
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Filter foreign chars from Kazakh datasets")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N texts and show rejected/kept (dry run)")
    parser.add_argument("--estimate", type=int, default=0,
                        help="Estimate rejection rate on N texts")
    parser.add_argument("--run", action="store_true",
                        help="Run full filter and push to HF Hub")
    parser.add_argument("--dataset", default="all",
                        choices=["kk", "enkk", "all"],
                        help="Which dataset to process")
    parser.add_argument("--output-kk", default="stukenov/sozkz-corpus-clean-kk-text-v4",
                        help="Output repo for KK dataset")
    parser.add_argument("--output-enkk", default="stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v2",
                        help="Output repo for EN-KK dataset")
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    KK_DATASET = "saken-tukenov/sozkz-corpus-clean-kk-text-v2"
    ENKK_DATASET = "saken-tukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1"

    datasets_to_process = []
    if args.dataset in ("kk", "all"):
        datasets_to_process.append((KK_DATASET, "text", args.output_kk))
    if args.dataset in ("enkk", "all"):
        datasets_to_process.append((ENKK_DATASET, "text_kk", args.output_enkk))

    if args.sample > 0:
        for ds_name, text_col, _ in datasets_to_process:
            sample_and_inspect(ds_name, text_col, args.sample, args.seed)

    if args.estimate > 0:
        for ds_name, text_col, _ in datasets_to_process:
            estimate_full_dataset(ds_name, text_col, args.estimate)

    if args.run:
        for ds_name, text_col, output in datasets_to_process:
            run_full_filter(ds_name, text_col, output, args.num_proc)

    if not args.sample and not args.estimate and not args.run:
        print("No action specified. Use --sample N, --estimate N, or --run")
        print("\nRecommended workflow:")
        print("  1. python scripts/data/filter_foreign_chars.py --sample 50 --dataset kk")
        print("  2. Inspect rejected/kept examples, adjust thresholds if needed")
        print("  3. python scripts/data/filter_foreign_chars.py --estimate 10000 --dataset all")
        print("  4. python scripts/data/filter_foreign_chars.py --run --dataset all")


if __name__ == "__main__":
    main()
