#!/usr/bin/env python3
"""exp032: Count tokens in MDBKD dataset per language using TinyLlama tokenizer.

Loads full kz-transformers/multidomain-kazakh-dataset, shuffles with seed=42,
tokenizes with TinyLlama tokenizer, reports stats per language.
Saves results to JSON for later use in training pipeline.

Usage:
    python exp032_count_tokens.py [--num-proc 8] [--output /tmp/exp032_token_stats.json]
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Count tokens per language in MDBKD")
    parser.add_argument("--dataset", default="kz-transformers/multidomain-kazakh-dataset")
    parser.add_argument("--tokenizer", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/exp032_token_stats.json")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # --- Load tokenizer ---
    log.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    log.info("Vocab size: %d", tokenizer.vocab_size)

    # --- Load dataset ---
    log.info("Loading dataset: %s", args.dataset)
    t0 = time.time()
    ds = load_dataset(args.dataset, split="train")
    log.info("Loaded %d rows in %.1fs", len(ds), time.time() - t0)

    # --- Report column info ---
    log.info("Columns: %s", ds.column_names)
    log.info("First row keys: %s", list(ds[0].keys()))

    # --- Shuffle ---
    log.info("Shuffling with seed=%d", args.seed)
    ds = ds.shuffle(seed=args.seed)

    # --- Count rows per language ---
    log.info("Counting rows per language...")
    t0 = time.time()
    lang_col = "predicted_language"
    if lang_col not in ds.column_names:
        log.warning("Column '%s' not found. Available: %s", lang_col, ds.column_names)
        log.warning("Will count all as 'unknown'")
        lang_col = None

    if lang_col:
        langs = ds[lang_col]
        row_counts = defaultdict(int)
        for lang in langs:
            row_counts[lang] += 1
        log.info("Row counts per language:")
        for lang, count in sorted(row_counts.items(), key=lambda x: -x[1]):
            log.info("  %s: %d rows (%.1f%%)", lang, count, 100.0 * count / len(ds))
    log.info("Row counting took %.1fs", time.time() - t0)

    # --- Tokenize and count tokens per language ---
    log.info("Tokenizing full dataset with num_proc=%d...", args.num_proc)
    log.info("This will take a while on 80M rows. Progress logged every batch.")

    # We'll process in batches to track progress and count per-language
    # Use map() with batched=True for efficiency
    batch_size = 10000
    total_rows = len(ds)
    token_counts = defaultdict(int)  # lang -> total tokens
    char_counts = defaultdict(int)   # lang -> total chars
    text_lengths = defaultdict(list) # lang -> list of token counts (sampled)
    total_tokens = 0
    sample_every = 100  # sample text lengths for distribution stats

    t0 = time.time()
    processed = 0

    # Process in slices for memory efficiency and progress reporting
    slice_size = 500_000
    num_slices = (total_rows + slice_size - 1) // slice_size

    for slice_idx in range(num_slices):
        start = slice_idx * slice_size
        end = min(start + slice_size, total_rows)
        batch = ds.select(range(start, end))

        texts_raw = batch["text"]
        langs_batch = batch[lang_col] if lang_col else ["unknown"] * len(texts_raw)

        # Filter out None/non-string texts
        valid = [(t if isinstance(t, str) else "", l) for t, l in zip(texts_raw, langs_batch)]
        texts = [t for t, _ in valid]
        langs_clean = [l for _, l in valid]

        # Tokenize all texts in this slice
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_length=True,
        )

        lengths = encoded["length"]

        for i, (lang, text, length) in enumerate(zip(langs_clean, texts, lengths)):
            token_counts[lang] += length
            char_counts[lang] += len(text)
            total_tokens += length

            if (processed + i) % sample_every == 0:
                text_lengths[lang].append(length)

        processed += len(texts)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total_rows - processed) / rate if rate > 0 else 0

        log.info(
            "[%d/%d] %.1f%% | %d tok so far | %.0f rows/s | ETA %.0fm",
            processed, total_rows,
            100.0 * processed / total_rows,
            total_tokens,
            rate,
            eta / 60,
        )

    elapsed_total = time.time() - t0

    # --- Compute stats ---
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info("Total rows: %d", total_rows)
    log.info("Total tokens: %d (%.2fB)", total_tokens, total_tokens / 1e9)
    log.info("Total time: %.1f min", elapsed_total / 60)
    log.info("")

    results = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "seed": args.seed,
        "total_rows": total_rows,
        "total_tokens": total_tokens,
        "total_tokens_billions": round(total_tokens / 1e9, 3),
        "elapsed_seconds": round(elapsed_total, 1),
        "per_language": {},
    }

    for lang in sorted(token_counts.keys()):
        tok = token_counts[lang]
        chars = char_counts[lang]
        rows = row_counts.get(lang, 0) if lang_col else total_rows
        avg_tokens_per_row = tok / rows if rows > 0 else 0
        avg_chars_per_row = chars / rows if rows > 0 else 0
        tokens_per_char = tok / chars if chars > 0 else 0

        # Distribution stats from sampled lengths
        sampled = sorted(text_lengths.get(lang, []))
        p50 = sampled[len(sampled) // 2] if sampled else 0
        p95 = sampled[int(len(sampled) * 0.95)] if sampled else 0
        p99 = sampled[int(len(sampled) * 0.99)] if sampled else 0

        log.info("Language: %s", lang)
        log.info("  Rows:          %d (%.1f%%)", rows, 100.0 * rows / total_rows)
        log.info("  Tokens:        %d (%.2fB, %.1f%%)", tok, tok / 1e9, 100.0 * tok / total_tokens)
        log.info("  Chars:         %d", chars)
        log.info("  Avg tok/row:   %.1f", avg_tokens_per_row)
        log.info("  Avg char/row:  %.1f", avg_chars_per_row)
        log.info("  Tok/char:      %.3f", tokens_per_char)
        log.info("  Length p50/p95/p99: %d / %d / %d", p50, p95, p99)
        log.info("")

        results["per_language"][lang] = {
            "rows": rows,
            "rows_pct": round(100.0 * rows / total_rows, 2),
            "tokens": tok,
            "tokens_billions": round(tok / 1e9, 3),
            "tokens_pct": round(100.0 * tok / total_tokens, 2),
            "chars": chars,
            "avg_tokens_per_row": round(avg_tokens_per_row, 1),
            "avg_chars_per_row": round(avg_chars_per_row, 1),
            "tokens_per_char": round(tokens_per_char, 3),
            "length_p50": p50,
            "length_p95": p95,
            "length_p99": p99,
        }

    # Also measure fertility rate on Kazakh-specific text
    log.info("Fertility rate test (tokens per word):")
    kaz_sample = "Қазақстан Республикасы — Орталық Азиядағы мемлекет"
    rus_sample = "Республика Казахстан — государство в Центральной Азии"
    eng_sample = "Republic of Kazakhstan — a state in Central Asia"

    for label, text in [("kaz", kaz_sample), ("rus", rus_sample), ("eng", eng_sample)]:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        fertility = len(tokens) / len(words)
        log.info("  %s: '%s'", label, text)
        log.info("       %d words -> %d tokens (fertility: %.2f)", len(words), len(tokens), fertility)
        results[f"fertility_{label}"] = {
            "text": text,
            "words": len(words),
            "tokens": len(tokens),
            "fertility": round(fertility, 2),
        }

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
