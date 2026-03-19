"""Train SentencePiece tokenizer on Kazakh text corpus."""

from __future__ import annotations

import argparse
import logging

from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--input", required=True, help="Text file (one sentence per line)")
    parser.add_argument("--model_prefix", default="tokenizers/kz_tts_sp", help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Training SentencePiece: vocab_size={args.vocab_size}, input={args.input}")

    KazakhTokenizer.train(
        input_file=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
    )
    logger.info(f"Saved: {args.model_prefix}.model, {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
