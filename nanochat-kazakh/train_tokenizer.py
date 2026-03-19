#!/usr/bin/env python3
"""Step 1: Train a GPT-2 style BPE tokenizer on clean Kazakh text.

Produces a 50,257-token ByteLevel BPE tokenizer:
  - 256 byte-level base tokens
  - ~49,634 BPE merges learned from Kazakh text
  - 3 special tokens: <|endoftext|>, <|padding|>, <|startoftext|>
  - ~360 Unicode digit characters

Usage:
    python train_tokenizer.py [--output ./tokenizers/sozkz-core-gpt2-50k-kk-base-v1]

Published tokenizer: https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1
"""

from __future__ import annotations

import argparse
import os
import unicodedata

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

DATASET_REPO = "saken-tukenov/sozkz-corpus-clean-kk-text-v2"
VOCAB_SIZE = 50_257

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
    "<|startoftext|>",
]


def get_unicode_digits() -> list[str]:
    """Collect all non-ASCII Unicode digit characters."""
    digits = []
    for cp in range(0x10000):
        ch = chr(cp)
        if unicodedata.category(ch) == "Nd" and ch not in "0123456789":
            digits.append(ch)
    return sorted(set(digits))


def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./tokenizers/sozkz-core-gpt2-50k-kk-base-v1")
    parser.add_argument("--push-to-hub", default=None, help="HF repo to push, e.g. user/tokenizer-name")
    args = parser.parse_args()

    print(f"Loading dataset: {DATASET_REPO}")
    ds = load_dataset(DATASET_REPO, split="train")
    print(f"  {len(ds)} texts loaded")

    unicode_digits = get_unicode_digits()
    extra_tokens = SPECIAL_TOKENS + unicode_digits
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}, Unicode digits: {len(unicode_digits)}")

    # GPT-2 style: ByteLevel BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=extra_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"Training tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer.train_from_iterator(
        batch_iterator(ds, batch_size=1000),
        trainer=trainer,
        length=len(ds),
    )
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")

    # Save raw tokenizer.json
    os.makedirs(args.output, exist_ok=True)
    tokenizer.save(f"{args.output}/tokenizer.json")

    # Save as HF PreTrainedTokenizerFast for easy loading
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        pad_token="<|padding|>",
        unk_token=None,
        model_max_length=1024,
    )
    hf_tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")

    # Verify
    for text in [
        "Қазақстан — Орталық Азиядағы мемлекет.",
        "Бүгін ауа райы жақсы болады.",
        "2024 жылы халықаралық конференция өтеді.",
    ]:
        ids = hf_tokenizer.encode(text)
        print(f"  [{len(ids)} tokens] {text}")

    if args.push_to_hub:
        hf_tokenizer.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
