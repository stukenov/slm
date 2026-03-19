"""Prepare mixed SFT dataset: ChatML (1.3M) + Alpaca sources (converted to ChatML).

Usage:
    python scripts/prepare_sft_mix.py
    python scripts/prepare_sft_mix.py --smoke-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_REPO = "stukenov/sozkz-corpus-chatml-kk-instruct-mix-v1"

CHATML_SOURCE = "stukenov/sozkz-instruct-chatml-kk-v1"
ALPACA_SOURCES = [
    ("saillab/alpaca_kazakh_taco", "alpaca_taco"),
    ("AmanMussa/kazakh-instruction-v2", "amanmussa"),
]


def alpaca_to_messages(row: dict) -> list[dict]:
    """Convert Alpaca row to ChatML messages list."""
    instruction = row.get("instruction", "")
    input_text = row.get("input", "")
    output = row.get("output", "")

    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def first_user_message(messages: list[dict]) -> str:
    """Extract first user message content for dedup."""
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def count_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] in ("user", "assistant"))


def load_chatml_source(name: str, smoke: bool) -> list[dict]:
    """Load ChatML dataset."""
    logger.info("Loading %s...", name)
    ds = load_dataset(name, split="train")
    if smoke:
        ds = ds.select(range(min(100, len(ds))))

    rows = []
    for row in ds:
        raw = row.get("messages", "[]")
        messages = json.loads(raw) if isinstance(raw, str) else raw
        if not messages:
            continue
        rows.append({
            "messages": json.dumps(messages, ensure_ascii=False),
            "source": "chatml_kk_v1",
            "num_turns": count_turns(messages),
            "_dedup_key": md5(first_user_message(messages)),
        })
    logger.info("  -> %d rows from %s", len(rows), name)
    return rows


def load_alpaca_source(hf_name: str, source_tag: str, smoke: bool) -> list[dict]:
    """Load Alpaca dataset and convert to ChatML."""
    logger.info("Loading %s...", hf_name)
    try:
        ds = load_dataset(hf_name, split="train")
    except Exception as e:
        logger.warning("Failed to load %s: %s — skipping", hf_name, e)
        return []

    if smoke:
        ds = ds.select(range(min(100, len(ds))))

    rows = []
    for row in ds:
        messages = alpaca_to_messages(row)
        user_content = first_user_message(messages)
        if not user_content or len(user_content.strip()) < 5:
            continue
        assistant_content = messages[-1]["content"] if messages else ""
        if not assistant_content or len(assistant_content.strip()) < 10:
            continue

        rows.append({
            "messages": json.dumps(messages, ensure_ascii=False),
            "source": source_tag,
            "num_turns": count_turns(messages),
            "_dedup_key": md5(user_content),
        })
    logger.info("  -> %d rows from %s", len(rows), hf_name)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Use 100 rows per source")
    parser.add_argument("--no-push", action="store_true", help="Don't push to HF")
    args = parser.parse_args()

    all_rows = []

    # ChatML source
    all_rows.extend(load_chatml_source(CHATML_SOURCE, args.smoke_test))

    # Alpaca sources
    for hf_name, tag in ALPACA_SOURCES:
        all_rows.extend(load_alpaca_source(hf_name, tag, args.smoke_test))

    logger.info("Total before dedup: %d", len(all_rows))

    # Dedup by MD5(first user message)
    seen = set()
    deduped = []
    for row in all_rows:
        key = row.pop("_dedup_key")
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    logger.info("After dedup: %d (removed %d)", len(deduped), len(all_rows) - len(deduped))

    # Add IDs
    import random
    random.seed(42)
    random.shuffle(deduped)
    for i, row in enumerate(deduped):
        row["id"] = f"mix-{i:07d}"

    # Create dataset
    ds = Dataset.from_list(deduped)

    # Split: 99% train, 1% val
    split = ds.train_test_split(test_size=0.01, seed=42)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    logger.info("Train: %d, Validation: %d", len(dataset["train"]), len(dataset["validation"]))

    # Source distribution
    from collections import Counter
    source_counts = Counter(row["source"] for row in deduped)
    for src, cnt in source_counts.most_common():
        logger.info("  %s: %d", src, cnt)

    # Always save locally for train_sft.py to load directly
    local_dir = "/root/slm/data/sft_mix"
    logger.info("Saving locally to %s...", local_dir)
    dataset.save_to_disk(local_dir)
    logger.info("Saved locally.")

    if not args.no_push and not args.smoke_test:
        logger.info("Pushing to %s...", HF_REPO)
        dataset.push_to_hub(HF_REPO, private=False)
        logger.info("Done!")
    else:
        logger.info("Skipping push (smoke-test or --no-push)")
        logger.info("Sample:\n%s", json.dumps(deduped[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
