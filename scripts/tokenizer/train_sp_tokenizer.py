"""Train a SentencePiece tokenizer for T5 on Kazakh data.

Usage:
    python scripts/train_sp_tokenizer.py \
        --dataset saken-tukenov/sozkz-corpus-clean-kk-text-v2 \
        --vocab_size 32000 \
        --num_sentinels 128 \
        --output_dir tokenizers/sozkz-vocab-sp-32k-kk-t5-v1

Produces a T5-compatible tokenizer with:
- SentencePiece unigram model (like original T5)
- <pad>=0, </s>=1, <unk>=2 (T5 convention)
- 128 sentinel tokens <extra_id_0> ... <extra_id_127>
- Saved in HuggingFace T5Tokenizer format
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset
from transformers import T5Config, T5Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def export_text(dataset_name: str, output_path: str, max_samples: int = 0) -> str:
    """Export dataset text to a plain text file for SentencePiece training."""
    log.info("Loading dataset %s ...", dataset_name)
    ds = load_dataset(dataset_name, split="train")

    # Find text column
    text_col = "text"
    if text_col not in ds.column_names:
        for alt in ["content", "sentence"]:
            if alt in ds.column_names:
                text_col = alt
                break

    log.info("Exporting %d samples (column: %s) to %s", len(ds), text_col, output_path)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example[text_col].strip()
            if text:
                f.write(text + "\n")
                count += 1
                if max_samples and count >= max_samples:
                    break

    log.info("Exported %d lines", count)
    return output_path


def train_sentencepiece(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
) -> str:
    """Train a SentencePiece unigram model (T5-style)."""
    log.info("Training SentencePiece (vocab=%d) ...", vocab_size)
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        # T5 special tokens: pad=0, eos=1, unk=2
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,  # T5 doesn't use BOS
        # Character coverage for non-Latin scripts
        character_coverage=0.9995,
        # Training params
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=True,
        # Normalization matching T5
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        # Byte fallback for unknown characters
        byte_fallback=True,
        # Max sentence length
        max_sentence_length=16384,
    )
    model_path = f"{model_prefix}.model"
    log.info("SentencePiece model saved: %s", model_path)
    return model_path


def convert_to_hf(
    sp_model_path: str,
    output_dir: str,
    num_sentinels: int = 128,
) -> None:
    """Convert SentencePiece model to HuggingFace T5Tokenizer format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load as T5Tokenizer (it natively supports SentencePiece)
    extra_ids = num_sentinels
    tokenizer = T5Tokenizer(
        vocab_file=sp_model_path,
        extra_ids=extra_ids,
    )

    # Copy spiece.model to output dir (save_pretrained may not copy it in transformers 5.x)
    dest_model = output_path / "spiece.model"
    shutil.copy2(sp_model_path, dest_model)
    log.info("Copied SentencePiece model to %s", dest_model)

    tokenizer.save_pretrained(str(output_path))

    # Verify by reloading from saved dir
    tokenizer = T5Tokenizer.from_pretrained(str(output_path))
    total_vocab = len(tokenizer)
    log.info("HF tokenizer saved to %s (vocab=%d, sentinels=%d)", output_dir, total_vocab, extra_ids)

    # Quick test
    test_text = "Қазақстан — Орталық Азиядағы ел."
    tokens = tokenizer.tokenize(test_text)
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    log.info("Test: '%s'", test_text)
    log.info("Tokens: %s", tokens[:20])
    log.info("IDs: %s", ids[:20])
    log.info("Decoded: '%s'", decoded)


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for T5")
    parser.add_argument("--dataset", default="saken-tukenov/sozkz-corpus-clean-kk-text-v2")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--num_sentinels", type=int, default=128)
    parser.add_argument("--output_dir", default="tokenizers/sozkz-vocab-sp-32k-kk-t5-v1")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0=all)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Export text
        text_file = os.path.join(tmpdir, "corpus.txt")
        export_text(args.dataset, text_file, max_samples=args.max_samples)

        # 2. Train SentencePiece
        model_prefix = os.path.join(tmpdir, "sp_kk")
        sp_model = train_sentencepiece(text_file, model_prefix, vocab_size=args.vocab_size)

        # 3. Convert to HF format
        convert_to_hf(sp_model, args.output_dir, num_sentinels=args.num_sentinels)

    log.info("Done! Tokenizer at: %s", args.output_dir)


if __name__ == "__main__":
    main()
