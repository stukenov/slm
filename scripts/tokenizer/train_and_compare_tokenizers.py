#!/usr/bin/env python3
"""Train BPE and SentencePiece tokenizers at multiple vocab sizes on clean Kazakh data.

Compares compression ratio, fertility, and token/character efficiency.

Usage:
    python scripts/train_and_compare_tokenizers.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from pathlib import Path

from datasets import load_dataset

DATASET_REPO = "saken-tukenov/sozkz-corpus-clean-kk-text-v2"
OUTPUT_BASE = "./tokenizers/comparison"
VOCAB_SIZES = [32_000, 48_000, 64_000, 95_000, 128_000, 192_000, 256_000]

SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>", "<|startoftext|>"]

TEST_TEXTS = [
    "Қазақстан — Орталық Азиядағы мемлекет.",
    "Бүгін ауа райы жақсы болады.",
    "2024 жылы халықаралық конференция өтеді.",
    "Алматы қаласында жаңа технологиялық парк ашылды. Бұл жоба мемлекеттік бағдарлама аясында жүзеге асырылды.",
    "Қазақ тілі — түркі тілдерінің қыпшақ тобына жататын тіл. Қазақстан Республикасының мемлекеттік тілі.",
    "Ғылым мен білім саласында елімізде көптеген жетістіктерге қол жеткізілді.",
    "Жасанды интеллект технологиялары күнделікті өмірімізге терең еніп келеді.",
]


def get_corpus_file(ds, output_base: str) -> str:
    """Dump dataset text to a file for training."""
    corpus_file = os.path.join(output_base, "_corpus.txt")
    if os.path.exists(corpus_file):
        print(f"  Using cached corpus: {corpus_file}")
        return corpus_file

    print(f"  Dumping {len(ds)} texts to {corpus_file}...")
    os.makedirs(output_base, exist_ok=True)
    with open(corpus_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            f.write(example["text"] + "\n")
            if (i + 1) % 10_000 == 0:
                print(f"    {i+1}/{len(ds)}", end="\r")
    print(f"  Done: {os.path.getsize(corpus_file) / 1e6:.1f} MB")
    return corpus_file


# ── BPE (HuggingFace tokenizers library) ──────────────────────────────────


def train_bpe(corpus_file: str, vocab_size: int, output_dir: str):
    """Train ByteLevel BPE tokenizer."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    print(f"  Training BPE vocab_size={vocab_size}...")
    t0 = time.time()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files=[corpus_file], trainer=trainer)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    from transformers import PreTrainedTokenizerFast

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        pad_token="<|padding|>",
        model_max_length=2048,
    )
    hf_tok.save_pretrained(output_dir)

    elapsed = time.time() - t0
    print(f"  BPE {vocab_size}: done in {elapsed:.1f}s, actual vocab={hf_tok.vocab_size}")
    return hf_tok


# ── SentencePiece ──────────────────────────────────────────────────────────


def train_sentencepiece(corpus_file: str, vocab_size: int, output_dir: str):
    """Train SentencePiece Unigram tokenizer."""
    import sentencepiece as spm

    print(f"  Training SentencePiece (Unigram) vocab_size={vocab_size}...")
    t0 = time.time()

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "sp")

    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9999,
        num_threads=os.cpu_count() or 4,
        byte_fallback=True,
        pad_id=1,
        eos_id=2,
        bos_id=0,
        unk_id=3,
        pad_piece="<|padding|>",
        eos_piece="<|endoftext|>",
        bos_piece="<|startoftext|>",
        normalization_rule_name="identity",
        max_sentence_length=16384,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )

    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    # Also wrap as HF tokenizer for consistent evaluation
    from transformers import PreTrainedTokenizerFast
    from tokenizers import SentencePieceBPETokenizer

    # Save as HF-compatible
    # We'll use the sp processor directly for evaluation
    elapsed = time.time() - t0
    print(f"  SP {vocab_size}: done in {elapsed:.1f}s, actual vocab={sp.get_piece_size()}")
    return sp


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate_bpe(tokenizer, texts: list[str], sample_ds) -> dict:
    """Evaluate BPE tokenizer."""
    # Test texts
    total_tokens = 0
    total_chars = 0
    for t in texts:
        ids = tokenizer.encode(t)
        total_tokens += len(ids)
        total_chars += len(t)

    # Large-scale on dataset sample
    ds_tokens = 0
    ds_chars = 0
    for ex in sample_ds:
        text = ex["text"]
        ids = tokenizer.encode(text)
        ds_tokens += len(ids)
        ds_chars += len(text)

    fertility = ds_tokens / max(1, len(ds_chars if isinstance(ds_chars, str) else str(ds_chars)))
    return {
        "test_tokens": total_tokens,
        "test_chars": total_chars,
        "test_ratio": total_chars / max(1, total_tokens),
        "ds_tokens": ds_tokens,
        "ds_chars": ds_chars,
        "ds_chars_per_token": ds_chars / max(1, ds_tokens),
        "ds_fertility": ds_tokens / max(1, len(sample_ds)),
    }


def evaluate_sp(sp, texts: list[str], sample_ds) -> dict:
    """Evaluate SentencePiece tokenizer."""
    total_tokens = 0
    total_chars = 0
    for t in texts:
        ids = sp.encode(t)
        total_tokens += len(ids)
        total_chars += len(t)

    ds_tokens = 0
    ds_chars = 0
    for ex in sample_ds:
        text = ex["text"]
        ids = sp.encode(text)
        ds_tokens += len(ids)
        ds_chars += len(text)

    return {
        "test_tokens": total_tokens,
        "test_chars": total_chars,
        "test_ratio": total_chars / max(1, total_tokens),
        "ds_tokens": ds_tokens,
        "ds_chars": ds_chars,
        "ds_chars_per_token": ds_chars / max(1, ds_tokens),
        "ds_fertility": ds_tokens / max(1, len(sample_ds)),
    }


def show_tokenization_examples(name, encode_fn, decode_fn, texts):
    """Show how tokenizer segments sample texts."""
    print(f"\n  {name} examples:")
    for t in texts[:3]:
        ids = encode_fn(t)
        decoded = decode_fn(ids)
        print(f"    [{len(ids):3d} tok] {t}")
        print(f"            → {decoded}")


def main():
    print("=" * 70)
    print("Kazakh Tokenizer Comparison: BPE vs SentencePiece")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset(DATASET_REPO, split="train")
    print(f"  Total: {len(ds)} texts")

    # Prepare corpus file
    corpus_file = get_corpus_file(ds, OUTPUT_BASE)

    # Sample for evaluation (use 5000 random texts)
    eval_sample = ds.shuffle(seed=42).select(range(min(5000, len(ds))))
    print(f"  Eval sample: {len(eval_sample)} texts")

    results = []

    # ── Train all BPE tokenizers ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: ByteLevel BPE Tokenizers")
    print("=" * 70)

    for vs in VOCAB_SIZES:
        out_dir = os.path.join(OUTPUT_BASE, f"bpe_{vs // 1000}k")
        try:
            tok = train_bpe(corpus_file, vs, out_dir)
            metrics = evaluate_bpe(tok, TEST_TEXTS, eval_sample)
            metrics["type"] = "BPE"
            metrics["vocab_size"] = vs
            metrics["actual_vocab"] = tok.vocab_size
            results.append(metrics)

            show_tokenization_examples(
                f"BPE-{vs//1000}k",
                tok.encode,
                tok.decode,
                TEST_TEXTS,
            )
        except Exception as e:
            print(f"  ERROR training BPE {vs}: {e}")

    # ── Train all SentencePiece tokenizers ─────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: SentencePiece (Unigram) Tokenizers")
    print("=" * 70)

    for vs in VOCAB_SIZES:
        out_dir = os.path.join(OUTPUT_BASE, f"sp_{vs // 1000}k")
        try:
            sp = train_sentencepiece(corpus_file, vs, out_dir)
            metrics = evaluate_sp(sp, TEST_TEXTS, eval_sample)
            metrics["type"] = "SentencePiece"
            metrics["vocab_size"] = vs
            metrics["actual_vocab"] = sp.get_piece_size()
            results.append(metrics)

            show_tokenization_examples(
                f"SP-{vs//1000}k",
                sp.encode,
                lambda ids: sp.decode(ids),
                TEST_TEXTS,
            )
        except Exception as e:
            print(f"  ERROR training SP {vs}: {e}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Type':<15} {'Vocab':>8} {'Actual':>8} {'Chars/Tok':>10} {'TestTok':>8} {'DStok':>10} {'AvgTok/Doc':>11}")
    print("-" * 75)

    # Sort by chars_per_token (higher = better compression)
    results.sort(key=lambda r: r["ds_chars_per_token"], reverse=True)

    for r in results:
        print(
            f"{r['type']:<15} {r['vocab_size']:>8,} {r['actual_vocab']:>8,} "
            f"{r['ds_chars_per_token']:>10.3f} {r['test_tokens']:>8} "
            f"{r['ds_tokens']:>10,} {r['ds_fertility']:>11.1f}"
        )

    # Best overall
    best = results[0]
    print(f"\nBEST: {best['type']} {best['vocab_size']//1000}k — "
          f"{best['ds_chars_per_token']:.3f} chars/token")

    # Best per type
    for ttype in ["BPE", "SentencePiece"]:
        type_results = [r for r in results if r["type"] == ttype]
        if type_results:
            best_t = type_results[0]
            print(f"  Best {ttype}: {best_t['vocab_size']//1000}k — "
                  f"{best_t['ds_chars_per_token']:.3f} chars/token")

    # Save results
    results_file = os.path.join(OUTPUT_BASE, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
