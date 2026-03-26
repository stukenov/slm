#!/usr/bin/env python3
"""
E2E test script for translation pipeline v2.

Runs the full pipeline on random rows from FineWeb-Edu and outputs
a detailed report for manual inspection and threshold tuning.

Usage:
    # Test on 100 random rows (GPU)
    python test_pipeline.py --num-rows 100

    # Validation run on 1000 rows
    python test_pipeline.py --num-rows 1000 --seed 42

    # CPU mode for quick testing
    python test_pipeline.py --num-rows 10 --cpu

    # Sequential rows (faster than random sampling)
    python test_pipeline.py --num-rows 100 --sequential
"""

import argparse
import random
import time
from itertools import islice

from datasets import load_dataset

from config import (
    SOURCE_DATASET,
    SOURCE_CONFIG,
    TEST_SAMPLE_SIZE,
)
from translator import Translator
from pipeline import process_rows


def sample_random_rows(num_rows: int, seed: int) -> list[dict]:
    """Sample random rows from FineWeb-Edu by picking random offsets."""
    rng = random.Random(seed)
    total_approx = 9_700_000
    offsets = sorted(rng.sample(range(total_approx), min(num_rows, total_approx)))

    print(f"Sampling {num_rows} random rows (seed={seed})...", flush=True)
    ds = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)

    rows = []
    ds_iter = iter(ds)
    prev_offset = 0
    for offset in offsets:
        skip = offset - prev_offset
        for _ in range(skip):
            next(ds_iter, None)
        row = next(ds_iter, None)
        if row:
            rows.append(row)
        prev_offset = offset + 1

    print(f"Loaded {len(rows)} rows.", flush=True)
    return rows


def print_report(results: list[dict]):
    """Print detailed analysis report for manual inspection."""
    total = len(results)
    translated = sum(1 for r in results if r["text_kk"])
    empty = total - translated

    all_conf_mean = [r["confidence_mean"] for r in results if r["confidence_mean"] > 0]
    all_conf_min = [r["confidence_min"] for r in results if r["confidence_min"] > 0]

    total_sents = sum(r["sentences_total"] for r in results)
    total_translated_sents = sum(r["sentences_translated"] for r in results)
    total_skipped_sents = sum(r["sentences_skipped"] for r in results)

    print(f"\n{'='*70}")
    print(f"TRANSLATION PIPELINE V2 — TEST REPORT")
    print(f"{'='*70}")

    print(f"\n## Document-level stats")
    print(f"  Total documents:       {total}")
    print(f"  With translation:      {translated} ({translated/max(total,1)*100:.1f}%)")
    print(f"  Empty (text_kk=''):    {empty} ({empty/max(total,1)*100:.1f}%)")

    print(f"\n## Sentence-level stats")
    print(f"  Total sentences:       {total_sents}")
    print(f"  Translated:            {total_translated_sents} ({total_translated_sents/max(total_sents,1)*100:.1f}%)")
    print(f"  Skipped:               {total_skipped_sents} ({total_skipped_sents/max(total_sents,1)*100:.1f}%)")

    if all_conf_mean:
        print(f"\n## Confidence distribution")
        print(f"  Mean confidence — avg: {sum(all_conf_mean)/len(all_conf_mean):.4f}, "
              f"min: {min(all_conf_mean):.4f}, max: {max(all_conf_mean):.4f}")
        print(f"  Min confidence  — avg: {sum(all_conf_min)/len(all_conf_min):.4f}, "
              f"min: {min(all_conf_min):.4f}, max: {max(all_conf_min):.4f}")

        buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        print(f"\n  Confidence_mean histogram:")
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            count = sum(1 for c in all_conf_mean if lo <= c < hi)
            bar = "#" * count
            print(f"    [{lo:.1f}, {hi:.1f}): {count:4d} {bar}")

    # Top 5 best translations
    good = sorted([r for r in results if r["confidence_mean"] > 0], key=lambda r: -r["confidence_mean"])
    print(f"\n## Top 5 best translations (highest confidence)")
    for r in good[:5]:
        print(f"\n  conf={r['confidence_mean']:.4f} | translated={r['sentences_translated']}/{r['sentences_total']}")
        print(f"  EN: {r['text_en'][:150]}...")
        print(f"  KK: {r['text_kk'][:150]}...")

    # Borderline translations
    borderline = sorted([r for r in results if 0 < r["confidence_mean"] < 0.5], key=lambda r: r["confidence_mean"])
    if borderline:
        print(f"\n## Top 5 borderline translations (lowest confidence)")
        for r in borderline[:5]:
            print(f"\n  conf={r['confidence_mean']:.4f} | translated={r['sentences_translated']}/{r['sentences_total']}")
            print(f"  EN: {r['text_en'][:150]}...")
            print(f"  KK: {r['text_kk'][:150]}...")

    # Empty translations
    empties = [r for r in results if not r["text_kk"]]
    if empties:
        print(f"\n## Examples of fully skipped documents ({len(empties)} total)")
        for r in empties[:5]:
            print(f"\n  total_sents={r['sentences_total']} | skipped={r['sentences_skipped']}")
            print(f"  EN: {r['text_en'][:200]}...")

    # Partially translated
    partial = [r for r in results if r["sentences_skipped"] > 0 and r["text_kk"]]
    if partial:
        partial.sort(key=lambda r: r["sentences_skipped"] / max(r["sentences_total"], 1), reverse=True)
        print(f"\n## Top 5 partially translated documents (most skips)")
        for r in partial[:5]:
            skip_pct = r["sentences_skipped"] / max(r["sentences_total"], 1) * 100
            print(f"\n  {r['sentences_skipped']}/{r['sentences_total']} skipped ({skip_pct:.0f}%) | conf={r['confidence_mean']:.4f}")
            print(f"  EN: {r['text_en'][:150]}...")
            print(f"  KK: {r['text_kk'][:150]}...")


def main():
    parser = argparse.ArgumentParser(description="E2E test for translation pipeline v2")
    parser.add_argument("--num-rows", type=int, default=TEST_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--sequential", action="store_true",
                        help="Load rows sequentially from start (faster than random sampling)")
    args = parser.parse_args()

    t_start = time.time()

    if args.sequential:
        print(f"Loading first {args.num_rows} rows sequentially...", flush=True)
        ds = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)
        rows = list(islice(ds, args.num_rows))
    else:
        rows = sample_random_rows(args.num_rows, args.seed)

    device = "cpu" if args.cpu else "cuda"
    print(f"Initializing translator (device={device})...", flush=True)
    translator = Translator(device=device, device_index=0)

    print(f"Processing {len(rows)} rows...", flush=True)
    results = process_rows(rows, translator)

    elapsed = time.time() - t_start
    print(f"\nProcessing completed in {elapsed:.1f}s")

    print_report(results)

    # Save raw results for further analysis
    from datasets import Dataset
    ds_out = Dataset.from_list(results)
    output_path = f"test_results_n{args.num_rows}_s{args.seed}.parquet"
    ds_out.to_parquet(output_path)
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    main()
