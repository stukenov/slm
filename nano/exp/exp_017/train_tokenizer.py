#!/usr/bin/env python3
"""Train a morpheme-aware byte-level BPE tokenizer (100K vocab) for Kazakh.

Inspired by HyperCLOVA X's approach:
  1. Pre-segment text into morphemes using rule-based Kazakh morphology
  2. Train byte-level BPE with morpheme boundaries as split points
  3. BPE merges cannot cross morpheme boundaries

Usage:
    # Train with rule-based segmentation (default, no external deps)
    python train_tokenizer.py

    # Train with Morfessor (unsupervised, needs training first)
    python train_tokenizer.py --backend morfessor

    # Train standard BPE baseline (no morpheme awareness) for comparison
    python train_tokenizer.py --baseline

    # Push to HuggingFace
    python train_tokenizer.py --push stukenov/sozkz-morphbpe-100k-kk-v1

    # Limit corpus size for testing
    python train_tokenizer.py --max-samples 100000
"""
from __future__ import annotations

import argparse
import os
import time
import json
import unicodedata
from collections import Counter

from datasets import load_dataset
from tokenizers import Tokenizer, Regex
from tokenizers import models, trainers, pre_tokenizers, decoders, processors
from tqdm import tqdm

from morpheme_segmenter import MorphemeSegmenter, MORPH_SEP

VOCAB_SIZE = 100_000
OUTPUT_DIR = "./output"

DATASETS = [
    ("stukenov/ekitil-corpus-annotated-kk-v1", "text", {"detected_lang": "kk", "lang_confidence_min": 0.95}),
]

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
    "<|startoftext|>",
]


def get_unicode_digits() -> list[str]:
    """Get non-ASCII digit characters."""
    digits = []
    for cp in range(0x10000):
        ch = chr(cp)
        if unicodedata.category(ch) == "Nd" and ch not in "0123456789":
            digits.append(ch)
    return sorted(set(digits))


def load_texts(max_samples: int | None = None) -> list[tuple]:
    """Load text from datasets in streaming mode (OOM-safe).
    Returns list of (IterableDataset, text_column, filter_fn, max_samples) tuples.
    """
    all_data = []

    for repo, text_col, filter_cond in DATASETS:
        print(f"[Stage 1] Loading {repo} (streaming)...", flush=True)
        try:
            ds = load_dataset(repo, split="train", streaming=True)

            # Peek at first row to find text column
            peek = next(iter(ds))
            if text_col not in peek:
                for alt in ["text", "text_kk", "content", "sentence"]:
                    if alt in peek:
                        text_col = alt
                        break
                else:
                    print(f"  WARNING: no text column in {repo}", flush=True)
                    continue

            # Build filter function
            min_confidence = filter_cond.pop("lang_confidence_min", None) if filter_cond else None
            eq_filters = {c: v for c, v in filter_cond.items()} if filter_cond else {}

            def _filter(example, filters=eq_filters, min_conf=min_confidence):
                if filters and not all(example.get(c) == v for c, v in filters.items()):
                    return False
                if min_conf is not None and example.get("lang_confidence", 0) < min_conf:
                    return False
                return True

            ds = ds.filter(_filter)
            if max_samples:
                ds = ds.take(max_samples)

            print(f"  Streaming with filter: lang=kk, conf>={min_confidence}", flush=True)
            all_data.append((ds, text_col))
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    return all_data


def build_morfessor_model(datasets_with_cols, save_path: str):
    """Train a Morfessor model on word frequencies from the corpus."""
    from morpheme_segmenter import MorfessorSegmenter
    import re

    print("[Stage M] Counting word frequencies for Morfessor training...", flush=True)
    word_counts = Counter()
    for ds, text_col in datasets_with_cols:
        for row in ds:
            text = row[text_col]
            if text:
                words = re.findall(r'[а-яА-ЯәғқңөұүһіӘҒҚҢӨҰҮҺІёЁ]+', text)
                word_counts.update(w.lower() for w in words if len(w) > 3)

    print(f"  Unique words: {len(word_counts):,}", flush=True)
    seg = MorfessorSegmenter()
    seg.train(dict(word_counts.most_common(500_000)), save_path)
    return save_path


def segmented_text_iterator(
    datasets_with_cols: list[tuple],
    segmenter: MorphemeSegmenter,
    batch_size: int = 1000,
):
    """Yield batches of morpheme-segmented text with progress logging."""
    processed = 0
    skipped = 0
    t_stage = time.time()
    print(f"[Stage 2] Segmenting + feeding to BPE trainer...", flush=True)
    for ds, text_col in datasets_with_cols:
        batch = []
        for row in ds:
            text = row[text_col]
            if text and text.strip():
                segmented = segmenter.segment(text)
                batch.append(segmented)
            else:
                skipped += 1
            if len(batch) >= batch_size:
                yield batch
                processed += len(batch)
                if processed % 100_000 == 0:
                    elapsed = time.time() - t_stage
                    backend = segmenter._backend
                    stats = getattr(backend, 'cache_stats', "")
                    print(f"  [seg] {processed:,} docs | skipped={skipped:,} | {elapsed:.0f}s {stats}", flush=True)
                batch = []
        if batch:
            yield batch
            processed += len(batch)
    print(f"  [seg] Done: {processed:,} docs, {skipped:,} skipped, {time.time()-t_stage:.0f}s total", flush=True)


def raw_text_iterator(datasets_with_cols: list[tuple], batch_size: int = 1000):
    """Yield batches of raw (unsegmented) text for baseline."""
    processed = 0
    for ds, text_col in datasets_with_cols:
        batch = []
        for row in ds:
            text = row[text_col]
            if text and text.strip():
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                processed += len(batch)
                if processed % 100_000 == 0:
                    print(f"  [raw] {processed:,} docs", flush=True)
                batch = []
        if batch:
            yield batch


def write_segmented_file(
    datasets_with_cols: list[tuple],
    segmenter,
    out_path: str,
) -> int:
    """Stage 2: stream + segment + write to file. Returns total docs written."""
    processed = 0
    skipped = 0
    t0 = time.time()
    print(f"[Stage 2] Segmenting corpus -> {out_path}", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ds, text_col in datasets_with_cols:
            for row in ds:
                text = row[text_col]
                if text and text.strip():
                    f.write(segmenter.segment(text) + "\n")
                    processed += 1
                    if processed % 100_000 == 0:
                        elapsed = time.time() - t0
                        backend = segmenter._backend
                        stats = getattr(backend, 'cache_stats', "")
                        print(f"  [seg] {processed:,} docs | {elapsed:.0f}s {stats}", flush=True)
                else:
                    skipped += 1
    print(f"  [seg] Done: {processed:,} docs, {skipped:,} skipped, {time.time()-t0:.0f}s", flush=True)
    return processed


def write_raw_file(datasets_with_cols: list[tuple], out_path: str) -> int:
    """Stage 2 for baseline: stream + write raw text to file."""
    processed = 0
    t0 = time.time()
    print(f"[Stage 2] Writing raw corpus -> {out_path}", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ds, text_col in datasets_with_cols:
            for row in ds:
                text = row[text_col]
                if text and text.strip():
                    f.write(text + "\n")
                    processed += 1
                    if processed % 100_000 == 0:
                        print(f"  [raw] {processed:,} docs | {time.time()-t0:.0f}s", flush=True)
    print(f"  [raw] Done: {processed:,} docs, {time.time()-t0:.0f}s", flush=True)
    return processed


def build_morpheme_aware_tokenizer(vocab_size: int, extra_tokens: list[str]) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Build a tokenizer that respects morpheme boundaries.

    The key: pre-tokenizer splits on MORPH_SEP before byte-level encoding.
    BPE merges happen within morpheme segments only.
    """
    tokenizer = Tokenizer(models.BPE())

    # Custom pre-tokenizer: split on morpheme boundaries THEN byte-level encode
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        # First: split on morpheme boundary marker (removed from output)
        pre_tokenizers.Split(pattern=Regex(r"\x1F"), behavior="removed"),
        # Then: standard byte-level encoding within each morpheme
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=extra_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    return tokenizer, trainer


def build_baseline_tokenizer(vocab_size: int, extra_tokens: list[str]) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Build a standard byte-level BPE tokenizer (no morpheme awareness)."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=extra_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    return tokenizer, trainer


def verify_tokenizer(hf_tokenizer, label: str):
    """Run verification on test sentences."""
    test_texts = [
        "Қазақстан — Орталық Азиядағы мемлекет.",
        "Бүгін ауа райы жақсы болады.",
        "2024 жылы халықаралық конференция өтеді.",
        "Мектепте оқушылар математика сабағына дайындалуда.",
        "Алматы қаласында жаңа метро стансасы ашылды.",
        "Үйлерімізде кітаптар көп.",
        "Университеттердегі студенттер емтихандарға дайындалуда.",
    ]

    print(f"\n{'='*60}")
    print(f"Verification: {label}")
    print(f"{'='*60}")
    total_tokens = 0
    total_words = 0
    for t in test_texts:
        ids = hf_tokenizer.encode(t)
        tokens = hf_tokenizer.convert_ids_to_tokens(ids)
        total_tokens += len(ids)
        total_words += len(t.split())
        print(f"  [{len(ids):2d} tok] {t}")
        print(f"          {tokens}")
    fertility = total_tokens / total_words
    print(f"\n  Avg fertility: {fertility:.2f} tokens/word")
    return fertility


def main():
    parser = argparse.ArgumentParser(description="Train morpheme-aware BPE tokenizer for Kazakh")
    parser.add_argument("--push", type=str, default=None, help="HF repo to push")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--backend", choices=["qazcorpora", "rule", "apertium", "morfessor"], default="qazcorpora",
                        help="Morpheme segmentation backend")
    parser.add_argument("--baseline", action="store_true",
                        help="Train standard BPE baseline (no morpheme awareness)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit corpus size (for testing)")
    parser.add_argument("--morfessor-model", type=str, default=None,
                        help="Path to pre-trained Morfessor model")
    args = parser.parse_args()

    t0 = time.time()

    # Determine output dir and name
    if args.baseline:
        name = "baseline-bpe-100k"
        output_dir = os.path.join(args.output_dir, name)
    else:
        name = f"morphbpe-{args.backend}-100k"
        output_dir = os.path.join(args.output_dir, name)

    # Load data (streaming, OOM-safe)
    datasets_with_cols = load_texts(args.max_samples)

    # Extra tokens
    unicode_digits = get_unicode_digits()
    extra_tokens = SPECIAL_TOKENS + unicode_digits

    os.makedirs(args.output_dir, exist_ok=True)
    corpus_file = os.path.join(args.output_dir, "corpus_segmented.txt")

    if args.baseline:
        # --- Baseline: standard BPE ---
        corpus_docs = write_raw_file(datasets_with_cols, corpus_file)
        print(f"\n[Stage 3] Training BASELINE byte-level BPE (vocab={args.vocab_size})...", flush=True)
        tokenizer, trainer = build_baseline_tokenizer(args.vocab_size, extra_tokens)
        tokenizer.train([corpus_file], trainer=trainer)
    else:
        # --- Morpheme-aware BPE ---
        print(f"\nBackend: {args.backend}", flush=True)

        # Initialize segmenter
        if args.backend == "morfessor" and not args.morfessor_model:
            morf_path = os.path.join(args.output_dir, "morfessor.bin")
            if os.path.exists(morf_path):
                print(f"Using existing Morfessor model: {morf_path}", flush=True)
            else:
                build_morfessor_model(datasets_with_cols, morf_path)
            args.morfessor_model = morf_path

        segmenter = MorphemeSegmenter(
            backend=args.backend,
            morfessor_model=args.morfessor_model,
        )

        # Show segmentation examples
        print("\nSegmentation examples:", flush=True)
        examples = ["үйлерімізде", "мектептегі", "оқушылар", "Қазақстанның", "университеттерде"]
        for w in examples:
            seg = segmenter.segment(w).replace(MORPH_SEP, "|")
            print(f"  {w} -> {seg}", flush=True)

        corpus_docs = write_segmented_file(datasets_with_cols, segmenter, corpus_file)

        print(f"\n[Stage 3] Training morpheme-aware BPE (vocab={args.vocab_size}, backend={args.backend})...", flush=True)
        tokenizer, trainer = build_morpheme_aware_tokenizer(args.vocab_size, extra_tokens)
        tokenizer.train([corpus_file], trainer=trainer)

        if segmenter.coverage >= 0:
            print(f"Morphological coverage: {segmenter.coverage:.1%}", flush=True)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        pad_token="<|padding|>",
        unk_token=None,
        model_max_length=4096,
    )
    hf_tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")

    # Save metadata
    metadata = {
        "name": name,
        "vocab_size": tokenizer.get_vocab_size(),
        "method": "baseline-bpe" if args.baseline else f"morpheme-aware-bpe-{args.backend}",
        "corpus_samples": corpus_docs,
        "training_time_sec": elapsed,
        "special_tokens": SPECIAL_TOKENS,
        "datasets": [d[0] for d in DATASETS],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Verify
    fertility = verify_tokenizer(hf_tokenizer, name)
    metadata["fertility"] = fertility
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Push to HF
    if args.push:
        print(f"\nPushing to {args.push}...")
        hf_tokenizer.push_to_hub(args.push)
        print("Upload complete!")

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print("Done!")


if __name__ == "__main__":
    main()
