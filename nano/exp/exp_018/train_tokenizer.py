#!/usr/bin/env python3
"""Train a morpheme-aware byte-level BPE tokenizer (256K vocab) for Kazakh.

Inspired by HyperCLOVA X's approach:
  1. Pre-segment text into morphemes using BiLSTM Kazakh morphology
  2. Train byte-level BPE with morpheme boundaries as split points
  3. BPE merges cannot cross morpheme boundaries

Usage:
    python train_tokenizer.py
    python train_tokenizer.py --push stukenov/sozkz-morphbpe-256k-kk-v1
    python train_tokenizer.py --save-corpus-hf stukenov/sozkz-corpus-segmented-kk-v1
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

VOCAB_SIZE = 256_000
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
    digits = []
    for cp in range(0x10000):
        ch = chr(cp)
        if unicodedata.category(ch) == "Nd" and ch not in "0123456789":
            digits.append(ch)
    return sorted(set(digits))


def load_texts(max_samples: int | None = None) -> list[tuple]:
    all_data = []
    for repo, text_col, filter_cond in DATASETS:
        print(f"[Stage 1] Loading {repo} (streaming)...", flush=True)
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            peek = next(iter(ds))
            if text_col not in peek:
                for alt in ["text", "text_kk", "content", "sentence"]:
                    if alt in peek:
                        text_col = alt
                        break
                else:
                    print(f"  WARNING: no text column in {repo}", flush=True)
                    continue

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


def write_segmented_file(datasets_with_cols, segmenter, out_path):
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


def upload_corpus_to_hf(corpus_path: str, repo_id: str):
    """Upload segmented corpus file as a HuggingFace dataset."""
    from datasets import Dataset
    import pyarrow as pa

    print(f"\n[Upload] Reading segmented corpus from {corpus_path}...", flush=True)
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                texts.append(line)
    print(f"  Loaded {len(texts):,} lines", flush=True)

    print(f"[Upload] Creating HF dataset and pushing to {repo_id}...", flush=True)
    ds = Dataset.from_dict({"text_segmented": texts})
    ds.push_to_hub(repo_id, private=False)
    print(f"[Upload] Done! Published to https://huggingface.co/datasets/{repo_id}", flush=True)


def build_morpheme_aware_tokenizer(vocab_size, extra_tokens):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(r"\x1F"), behavior="removed"),
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


def build_baseline_tokenizer(vocab_size, extra_tokens):
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


def verify_tokenizer(hf_tokenizer, label):
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
    parser = argparse.ArgumentParser(description="Train morpheme-aware BPE tokenizer (256K) for Kazakh")
    parser.add_argument("--push", type=str, default=None, help="HF repo to push tokenizer")
    parser.add_argument("--save-corpus-hf", type=str, default=None,
                        help="HF repo to upload segmented corpus as dataset")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--backend", choices=["qazcorpora", "rule", "apertium", "morfessor"], default="qazcorpora")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--morfessor-model", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    vocab_k = args.vocab_size // 1000

    if args.baseline:
        name = f"baseline-bpe-{vocab_k}k"
    else:
        name = f"morphbpe-{args.backend}-{vocab_k}k"

    output_dir = os.path.join(args.output_dir, name)
    datasets_with_cols = load_texts(args.max_samples)

    unicode_digits = get_unicode_digits()
    extra_tokens = SPECIAL_TOKENS + unicode_digits

    os.makedirs(args.output_dir, exist_ok=True)
    corpus_file = os.path.join(args.output_dir, "corpus_segmented.txt")

    if args.baseline:
        from train_tokenizer_baseline import write_raw_file
        corpus_docs = write_raw_file(datasets_with_cols, corpus_file)
        print(f"\n[Stage 3] Training BASELINE byte-level BPE (vocab={args.vocab_size})...", flush=True)
        tokenizer, trainer = build_baseline_tokenizer(args.vocab_size, extra_tokens)
        tokenizer.train([corpus_file], trainer=trainer)
    else:
        print(f"\nBackend: {args.backend}", flush=True)

        if args.backend == "morfessor" and not args.morfessor_model:
            morf_path = os.path.join(args.output_dir, "morfessor.bin")
            if os.path.exists(morf_path):
                print(f"Using existing Morfessor model: {morf_path}", flush=True)
            else:
                from collections import Counter
                print("[Stage M] Would need to train Morfessor model first.", flush=True)
            args.morfessor_model = morf_path

        segmenter = MorphemeSegmenter(
            backend=args.backend,
            morfessor_model=args.morfessor_model,
        )

        print("\nSegmentation examples:", flush=True)
        examples = ["үйлерімізде", "мектептегі", "оқушылар", "Қазақстанның", "университеттерде"]
        for w in examples:
            seg = segmenter.segment(w).replace(MORPH_SEP, "|")
            print(f"  {w} -> {seg}", flush=True)

        corpus_docs = write_segmented_file(datasets_with_cols, segmenter, corpus_file)

        # Upload segmented corpus to HF before training (so it's saved even if training fails)
        if args.save_corpus_hf:
            upload_corpus_to_hf(corpus_file, args.save_corpus_hf)

        print(f"\n[Stage 3] Training morpheme-aware BPE (vocab={args.vocab_size}, backend={args.backend})...", flush=True)
        tokenizer, trainer = build_morpheme_aware_tokenizer(args.vocab_size, extra_tokens)
        tokenizer.train([corpus_file], trainer=trainer)

        if segmenter.coverage >= 0:
            print(f"Morphological coverage: {segmenter.coverage:.1%}", flush=True)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

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

    fertility = verify_tokenizer(hf_tokenizer, name)
    metadata["fertility"] = fertility
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if args.push:
        print(f"\nPushing tokenizer to {args.push}...")
        hf_tokenizer.push_to_hub(args.push)
        print("Upload complete!")

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print("Done!")


if __name__ == "__main__":
    main()
