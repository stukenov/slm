"""Prepare instruction SFT dataset for MoE fine-tuning.

Tries to load:
  - nurkhan5l/kazakh-ift (10.6K instruction pairs) — may be gated
  - issai/kazqad-retrieval (6K QA) — may be gated
  - kz-transformers/kazakh-dastur-mc (1K MC questions) — public

Gated datasets are skipped gracefully if access is not granted.
"""

import argparse
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def try_load_kazakh_ift() -> Dataset | None:
    """Load nurkhan5l/kazakh-ift (gated)."""
    logger.info("Trying nurkhan5l/kazakh-ift...")
    try:
        ds = load_dataset("nurkhan5l/kazakh-ift", split="train")
        logger.info("  Loaded %d samples", len(ds))

        def format_ift(example):
            instruction = example.get("instruction", example.get("input", ""))
            output = example.get("output", example.get("response", ""))
            text = f"<s>Нұсқаулық: {instruction}\nЖауап: {output}</s>"
            return {"text": text}

        ds = ds.map(format_ift, remove_columns=ds.column_names)
        return ds
    except Exception as e:
        logger.warning("  Skipped (gated/unavailable): %s", e)
        return None


def try_load_kazqad() -> Dataset | None:
    """Load issai/kazqad-retrieval (gated)."""
    logger.info("Trying issai/kazqad-retrieval...")
    try:
        ds = load_dataset("issai/kazqad-retrieval", split="queries")
        logger.info("  Loaded %d samples", len(ds))

        def format_qa(example):
            question = example.get("query", example.get("question", ""))
            answer = example.get("answer", example.get("text", ""))
            if not answer:
                return {"text": ""}
            return {"text": f"<s>Нұсқаулық: {question}\nЖауап: {answer}</s>"}

        ds = ds.map(format_qa, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 10)
        return ds
    except Exception as e:
        logger.warning("  Skipped (gated/unavailable): %s", e)
        return None


def load_dastur_mc() -> Dataset | None:
    """Load kz-transformers/kazakh-dastur-mc (public, test split)."""
    logger.info("Loading kz-transformers/kazakh-dastur-mc...")
    try:
        ds = load_dataset("kz-transformers/kazakh-dastur-mc", split="test")
        logger.info("  Loaded %d samples, columns: %s", len(ds), ds.column_names)

        def format_mc(example):
            q = example.get("question", "")
            choices = []
            for key in ["A", "B", "C", "D"]:
                val = example.get(key, "")
                if val:
                    choices.append(f"{key}) {val}")
            answer_key = example.get("answer", "")
            answer_text = example.get(answer_key, answer_key)
            choices_str = "\n".join(choices)
            text = f"<s>Нұсқаулық: {q}\n{choices_str}\nЖауап: {answer_text}</s>"
            return {"text": text}

        ds = ds.map(format_mc, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 20)
        return ds
    except Exception as e:
        logger.warning("  Failed: %s", e)
        return None


def tokenize_instruction_data(ds: Dataset, tokenizer, block_size: int, num_proc: int) -> Dataset:
    """Tokenize instruction data (each example = separate sequence)."""

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            max_length=block_size,
            truncation=True,
            return_attention_mask=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = ds.map(
        tokenize_fn, batched=True, num_proc=num_proc,
        remove_columns=["text"], desc="Tokenizing instructions",
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Prepare MoE instruction SFT data")
    parser.add_argument("--tokenizer", default="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--output", default="outputs/sozkz-corpus-synthetic-kk-moe-sft-v1")
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    parts = []

    ift = try_load_kazakh_ift()
    if ift is not None:
        parts.append(ift)

    kazqad = try_load_kazqad()
    if kazqad is not None:
        parts.append(kazqad)

    dastur = load_dastur_mc()
    if dastur is not None:
        # Upsample to be more balanced with other sources
        repeat = 10
        dastur = concatenate_datasets([dastur] * repeat)
        logger.info("Upsampled dastur-mc to %d (x%d)", len(dastur), repeat)
        parts.append(dastur)

    if not parts:
        raise ValueError("No instruction datasets available! Request access to gated datasets.")

    full_ds = concatenate_datasets(parts).shuffle(seed=42)
    logger.info("Total instruction samples: %d", len(full_ds))

    tokenized = tokenize_instruction_data(full_ds, tokenizer, args.block_size, args.num_proc)
    logger.info("Tokenized samples: %d", len(tokenized))

    split = tokenized.train_test_split(test_size=args.val_ratio, seed=42)
    ds_dict = DatasetDict({"train": split["train"], "validation": split["test"]})
    logger.info("Train: %d, Validation: %d", len(ds_dict["train"]), len(ds_dict["validation"]))

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(output_path))
    logger.info("Saved to %s", output_path)

    if args.push_to_hub:
        logger.info("Pushing to HF Hub: %s", args.push_to_hub)
        ds_dict.push_to_hub(args.push_to_hub)
        logger.info("Done!")


if __name__ == "__main__":
    main()
