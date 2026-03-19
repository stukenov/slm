#!/usr/bin/env python3
"""Train a GPT-2 style BPE tokenizer (100K vocab) on all available Kazakh text.

Uses multiple datasets:
  1. saken-tukenov/sozkz-corpus-clean-kk-text-v2 (main Kazakh corpus)
  2. kz-transformers/multidomain-kazakh-dataset (23.6M samples)

Final vocab: 100,000 tokens
  - 256 byte-level base tokens
  - BPE merges to reach target
  - Special tokens: <|endoftext|>, <|padding|>, <|startoftext|>

Usage:
    python train_tokenizer_100k.py
    python train_tokenizer_100k.py --push stukenov/sozkz-core-gpt2-100k-kk-base-v1
"""
from __future__ import annotations

import argparse
import os
import time
import unicodedata

from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

VOCAB_SIZE = 100_000

DATASETS = [
    "saken-tukenov/sozkz-corpus-clean-kk-text-v2",
    "kz-transformers/multidomain-kazakh-dataset",
]

OUTPUT_DIR = "./tokenizers/sozkz-core-gpt2-100k-kk-base-v1"

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


def load_all_texts():
    """Load text from all available datasets."""
    all_texts = []

    for repo in DATASETS:
        print(f"Loading {repo}...")
        try:
            ds = load_dataset(repo, split="train")
            # Find text column
            text_col = None
            for col in ["text", "text_kk", "content", "sentence"]:
                if col in ds.column_names:
                    text_col = col
                    break
            if text_col is None:
                print(f"  WARNING: no text column found in {repo}, columns: {ds.column_names}")
                continue
            print(f"  {len(ds):,} samples, column: '{text_col}'")
            all_texts.append((ds, text_col))
        except Exception as e:
            print(f"  ERROR loading {repo}: {e}")

    return all_texts


def multi_dataset_iterator(datasets_with_cols, batch_size=1000):
    """Yield batches of text from multiple datasets."""
    for ds, text_col in datasets_with_cols:
        for i in range(0, len(ds), batch_size):
            batch = ds[i : i + batch_size][text_col]
            # Filter None/empty
            batch = [t for t in batch if t and len(t.strip()) > 0]
            if batch:
                yield batch


def count_total(datasets_with_cols):
    return sum(len(ds) for ds, _ in datasets_with_cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", type=str, default=None,
                        help="HF repo to push tokenizer (e.g. stukenov/sozkz-core-gpt2-100k-kk-base-v1)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args()

    t0 = time.time()

    # Load datasets
    datasets_with_cols = load_all_texts()
    total = count_total(datasets_with_cols)
    print(f"\nTotal samples: {total:,}")

    # Extra tokens
    unicode_digits = get_unicode_digits()
    extra_tokens = SPECIAL_TOKENS + unicode_digits
    print(f"Special tokens: {len(SPECIAL_TOKENS)}, Unicode digits: {len(unicode_digits)}")

    # Build tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=extra_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"\nTraining tokenizer (vocab_size={args.vocab_size})...")
    tokenizer.train_from_iterator(
        multi_dataset_iterator(datasets_with_cols, batch_size=1000),
        trainer=trainer,
        length=total,
    )
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Save locally
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save(f"{args.output_dir}/tokenizer.json")

    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        pad_token="<|padding|>",
        unk_token=None,
        model_max_length=2048,
    )
    hf_tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

    # Verify
    test_texts = [
        "Қазақстан — Орталық Азиядағы мемлекет.",
        "Бүгін ауа райы жақсы болады.",
        "2024 жылы халықаралық конференция өтеді.",
        "Мектепте оқушылар математика сабағына дайындалуда.",
        "Алматы қаласында жаңа метро стансасы ашылды.",
    ]
    print("\nVerification:")
    for t in test_texts:
        ids = hf_tokenizer.encode(t)
        decoded = hf_tokenizer.decode(ids)
        print(f"  [{len(ids):2d} tokens] {t}")

    # Compare with 50K if available
    try:
        from transformers import AutoTokenizer
        tok50k = AutoTokenizer.from_pretrained("stukenov/sozkz-core-gpt2-50k-kk-base-v1")
        print("\nComparison 50K vs 100K:")
        for t in test_texts:
            ids_50k = tok50k.encode(t)
            ids_100k = hf_tokenizer.encode(t)
            saving = (1 - len(ids_100k) / len(ids_50k)) * 100
            print(f"  50K: {len(ids_50k):2d} | 100K: {len(ids_100k):2d} | saving: {saving:+.0f}% | {t[:40]}")
    except Exception:
        pass

    print(f"\nFinal vocab size: {hf_tokenizer.vocab_size}")

    # Push to HF
    if args.push:
        print(f"\nPushing to {args.push}...")
        hf_tokenizer.push_to_hub(args.push)
        print("Upload complete!")

    print(f"Total time: {(time.time()-t0)/60:.1f} min")
    print("Done!")


if __name__ == "__main__":
    main()
