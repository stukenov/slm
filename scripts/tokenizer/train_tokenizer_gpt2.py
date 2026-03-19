#!/usr/bin/env python3
"""Train a GPT-2 style BPE tokenizer on the clean Kazakh corpus.

Final vocab: 50,257 tokens
  - 256 byte-level base tokens
  - BPE merges to reach target
  - Special tokens: <|endoftext|>, <|padding|>, <|startoftext|>
  - All Unicode digit characters (٠-٩, ۰-۹, ०-९, etc.)

Usage:
    python scripts/train_tokenizer_gpt2.py
"""

from __future__ import annotations

import unicodedata
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

DATASET_REPO = "saken-tukenov/sozkz-corpus-clean-kk-text-v2"
OUTPUT_DIR = "./tokenizers/sozkz-core-gpt2-50k-kk-base-v1"
VOCAB_SIZE = 50_257

# Special tokens
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
    "<|startoftext|>",
]


def get_unicode_digits() -> list[str]:
    """Get all Unicode digit characters."""
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
    print("Loading dataset...")
    ds = load_dataset(DATASET_REPO, split="train")
    print(f"  {len(ds)} texts loaded")

    # Collect extra tokens: special + unicode digits
    unicode_digits = get_unicode_digits()
    extra_tokens = SPECIAL_TOKENS + unicode_digits
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  Unicode digits: {len(unicode_digits)}")
    print(f"  Total extra tokens: {len(extra_tokens)}")

    # Build GPT-2 style tokenizer: ByteLevel BPE
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

    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Save
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer.save(f"{OUTPUT_DIR}/tokenizer.json")

    # Also save as HF PreTrainedTokenizerFast for easy loading
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        pad_token="<|padding|>",
        unk_token=None,
        model_max_length=1024,
    )
    hf_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

    # Verify
    test_texts = [
        "Қазақстан — Орталық Азиядағы мемлекет.",
        "Бүгін ауа райы жақсы болады.",
        "2024 жылы халықаралық конференция өтеді.",
    ]
    for t in test_texts:
        ids = hf_tokenizer.encode(t)
        decoded = hf_tokenizer.decode(ids)
        print(f"  [{len(ids)} tokens] {t}")
        print(f"  -> {decoded}")

    print(f"\nFinal vocab size: {hf_tokenizer.vocab_size}")
    print("Done!")


if __name__ == "__main__":
    main()
