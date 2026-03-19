"""Tokenizer training and extension for Kazakh language."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def train_kazakh_bpe(
    dataset_name: str = "kz-transformers/multidomain-kazakh-dataset",
    vocab_size: int = 32000,
    min_frequency: int = 2,
    output_dir: str = "./tokenizers/kazakh-bpe",
    text_column: str = "text",
) -> PreTrainedTokenizerFast:
    """Train a new BPE tokenizer on Kazakh text.

    Uses file-based training for 10-100x speedup over iterator-based approach.
    """
    import tempfile

    logger.info("Training BPE tokenizer, vocab_size=%d", vocab_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dump dataset to temp file for fast Rust-based training
    temp_file = output_path / "_train_corpus.txt"

    if not temp_file.exists():
        logger.info("Dumping dataset to file for fast tokenizer training...")
        ds = load_dataset(dataset_name, split="train")

        # Check text column
        if text_column not in ds.column_names:
            for alt in ["text", "content", "sentence"]:
                if alt in ds.column_names:
                    text_column = alt
                    break

        # Write to file in chunks (fast)
        with open(temp_file, "w", encoding="utf-8") as f:
            for i, example in enumerate(ds):
                f.write(example[text_column] + "\n")
                if (i + 1) % 1_000_000 == 0:
                    logger.info("Written %dM samples...", (i + 1) // 1_000_000)

        logger.info("Dataset dumped to %s", temp_file)
    else:
        logger.info("Using cached corpus file: %s", temp_file)

    # Train on file (10-100x faster than iterator)
    logger.info("Training BPE on file...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(temp_file)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<|endoftext|>", "<|padding|>"],
    )

    tokenizer.save_model(str(output_path))

    # Convert to HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
    )
    hf_tokenizer.save_pretrained(str(output_path))

    # Cleanup temp file
    temp_file.unlink(missing_ok=True)

    logger.info("Tokenizer saved to %s", output_path)
    return hf_tokenizer


def extend_tokenizer(
    base_model_name: str,
    new_tokens_path: str | None = None,
    dataset_name: str = "kz-transformers/multidomain-kazakh-dataset",
    num_new_tokens: int = 5000,
    output_dir: str = "./tokenizers/extended",
) -> AutoTokenizer:
    """Extend an existing tokenizer with Kazakh-specific tokens.

    Either provide a file with new tokens (one per line) or automatically
    find the most frequent Kazakh subwords not in the original vocabulary.
    """
    logger.info("Extending tokenizer from %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if new_tokens_path:
        with open(new_tokens_path) as f:
            new_tokens = [line.strip() for line in f if line.strip()]
    else:
        # Train a temporary tokenizer and extract tokens not in original vocab
        logger.info("Finding new Kazakh tokens via temporary BPE...")
        kz_tokenizer = train_kazakh_bpe(
            dataset_name=dataset_name,
            vocab_size=num_new_tokens * 2,
            output_dir=str(Path(output_dir) / "_tmp_kz_bpe"),
        )
        kz_vocab = set(kz_tokenizer.get_vocab().keys())
        orig_vocab = set(tokenizer.get_vocab().keys())
        new_tokens = list(kz_vocab - orig_vocab)[:num_new_tokens]

    added = tokenizer.add_tokens(new_tokens)
    logger.info("Added %d new tokens to tokenizer", added)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_path))

    logger.info("Extended tokenizer saved to %s", output_path)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenizer tools")
    sub = parser.add_subparsers(dest="command")

    # Train new tokenizer
    train_parser = sub.add_parser("train", help="Train new BPE tokenizer")
    train_parser.add_argument("--dataset", default="kz-transformers/multidomain-kazakh-dataset")
    train_parser.add_argument("--vocab_size", type=int, default=32000)
    train_parser.add_argument("--output_dir", default="./tokenizers/kazakh-bpe")

    # Extend existing tokenizer
    extend_parser = sub.add_parser("extend", help="Extend existing tokenizer")
    extend_parser.add_argument("--base_model", required=True)
    extend_parser.add_argument("--num_new_tokens", type=int, default=5000)
    extend_parser.add_argument("--output_dir", default="./tokenizers/extended")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "train":
        train_kazakh_bpe(
            dataset_name=args.dataset,
            vocab_size=args.vocab_size,
            output_dir=args.output_dir,
        )
    elif args.command == "extend":
        extend_tokenizer(
            base_model_name=args.base_model,
            num_new_tokens=args.num_new_tokens,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
