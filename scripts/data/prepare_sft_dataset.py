"""Prepare Kazakh SFT dataset from multiple open sources.

Sources:
  - AmanMussa/kazakh-instruction-v2 (52K, MIT)

Output: saken-tukenov/sozkz-corpus-synthetic-kk-instruct-v1
"""

from __future__ import annotations

import argparse
import hashlib
import logging

from datasets import Dataset, DatasetDict, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_REPO = "saken-tukenov/sozkz-corpus-synthetic-kk-instruct-v1"


def load_kazakh_instruction_v2() -> list[dict]:
    """AmanMussa/kazakh-instruction-v2 — instruction/input/output format."""
    logger.info("Loading AmanMussa/kazakh-instruction-v2...")
    ds = load_dataset("AmanMussa/kazakh-instruction-v2", split="train")
    rows = []
    for r in ds:
        rows.append({
            "instruction": (r.get("instruction") or "").strip(),
            "input": (r.get("input") or "").strip(),
            "output": (r.get("output") or "").strip(),
            "source": "kazakh-instruction-v2",
        })
    logger.info("  -> %d rows", len(rows))
    return rows



def load_openhermes_kz() -> list[dict]:
    """Vikhrmodels/OpenHermes-2.5-kz — ShareGPT conversations, extract first turn."""
    logger.info("Loading Vikhrmodels/OpenHermes-2.5-kz...")
    try:
        ds = load_dataset("Vikhrmodels/OpenHermes-2.5-kz", split="train")
    except Exception as e:
        logger.warning("  Standard load failed (%s), trying with parquet...", e)
        ds = load_dataset(
            "parquet",
            data_files="hf://datasets/Vikhrmodels/OpenHermes-2.5-kz/**/*.parquet",
            split="train",
        )
    rows = []
    for r in ds:
        convs = r.get("conversations") or []
        human_msg, gpt_msg = "", ""
        for msg in convs:
            role = msg.get("from") or msg.get("role") or ""
            value = msg.get("value") or msg.get("content") or ""
            if role in ("human", "user") and not human_msg:
                human_msg = value.strip()
            elif role in ("gpt", "assistant") and not gpt_msg:
                gpt_msg = value.strip()
            if human_msg and gpt_msg:
                break
        if human_msg and gpt_msg:
            rows.append({
                "instruction": human_msg,
                "input": "",
                "output": gpt_msg,
                "source": "openhermes-kz",
            })
    logger.info("  -> %d rows", len(rows))
    return rows


def dedup_and_filter(rows: list[dict], min_instruction: int = 5, min_output: int = 10) -> list[dict]:
    """Deduplicate by MD5(instruction) and filter short texts."""
    seen = set()
    result = []
    skipped_dup, skipped_short = 0, 0
    for r in rows:
        h = hashlib.md5(r["instruction"].encode()).hexdigest()
        if h in seen:
            skipped_dup += 1
            continue
        seen.add(h)
        if len(r["instruction"]) < min_instruction or len(r["output"]) < min_output:
            skipped_short += 1
            continue
        result.append(r)
    logger.info("Dedup: removed %d duplicates, %d too short. Final: %d", skipped_dup, skipped_short, len(result))
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare Kazakh SFT dataset")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--repo", default=HF_REPO, help="HF repo name")
    parser.add_argument("--save-local", default=None, help="Save to local directory")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load all sources
    all_rows = []
    all_rows.extend(load_kazakh_instruction_v2())
    logger.info("Total raw: %d", len(all_rows))

    # Dedup + filter
    clean = dedup_and_filter(all_rows)

    # Create dataset and split
    ds = Dataset.from_list(clean)
    splits = ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
    dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})

    logger.info("Train: %d, Validation: %d", len(dataset["train"]), len(dataset["validation"]))

    # Show sample
    sample = dataset["train"][0]
    logger.info("Sample instruction: %s", sample["instruction"][:100])
    logger.info("Sample output: %s", sample["output"][:100])

    if args.save_local:
        dataset.save_to_disk(args.save_local)
        logger.info("Saved to %s", args.save_local)

    if args.push:
        dataset.push_to_hub(args.repo, private=False)
        logger.info("Pushed to %s", args.repo)


if __name__ == "__main__":
    main()
