"""Data utilities for GEC (Grammatical Error Correction) fine-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

TASK_FIX = "<TASK_FIX>"
SRC = "<SRC>"
SEP = "<SEP>"

SPECIAL_TOKENS = [TASK_FIX, SRC, SEP]


def load_gec_dataset(
    dataset_name: str = "saken-tukenov/sozkz-corpus-synthetic-kk-gec-v1",
    identity_ratio: float = 0.2,
    seed: int = 42,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> dict[str, Dataset]:
    """Load GEC dataset and add identity examples.

    Returns dict with 'train', 'validation', 'test' splits.
    """
    # Dataset has multiple subdirs with different columns.
    # Load each via data_dir, keep only input/target, and concatenate.
    DATA_DIRS = [
        "data/grammar_balanced_v2", "data/grammar_combined", "data/grammar_focused",
        "data/grammar_v2", "data/hybrid_v1", "data/nusqaulyq_v1", "data/nusqaulyq_v2",
        "data/processed", "data/processed_v2", "data/processed_v3",
    ]
    from datasets import concatenate_datasets as _concat

    parts = {"train": [], "validation": [], "test": []}
    for data_dir in DATA_DIRS:
        try:
            sub = load_dataset(dataset_name, data_dir=data_dir)
        except Exception as e:
            logger.warning("Skipping %s: %s", data_dir, e)
            continue
        for split in parts:
            if split in sub:
                cols_to_remove = [c for c in sub[split].column_names if c not in ("input", "target")]
                cleaned = sub[split].remove_columns(cols_to_remove) if cols_to_remove else sub[split]
                parts[split].append(cleaned)
                logger.info("  Loaded %s/%s: %d examples", data_dir, split, len(cleaned))

    ds = {}
    for split, dss in parts.items():
        if dss:
            ds[split] = _concat(dss)

    if not ds:
        raise ValueError(f"Could not load any data from {dataset_name}")

    # Use existing splits if available, otherwise create them
    if "train" in ds and "validation" in ds and "test" in ds:
        train_ds = ds["train"]
        val_ds = ds["validation"]
        test_ds = ds["test"]
    elif "train" in ds and "validation" in ds:
        train_ds = ds["train"]
        val_ds = ds["validation"]
        test_ds = None
    elif "train" in ds:
        # Split manually
        split1 = ds["train"].train_test_split(test_size=val_ratio + test_ratio, seed=seed)
        train_ds = split1["train"]
        if test_ratio > 0:
            split2 = split1["test"].train_test_split(
                test_size=test_ratio / (val_ratio + test_ratio), seed=seed
            )
            val_ds = split2["train"]
            test_ds = split2["test"]
        else:
            val_ds = split1["test"]
            test_ds = None
    else:
        raise ValueError(f"Unexpected splits in dataset: {list(ds.keys())}")

    # Add identity examples (target == target) to train set
    if identity_ratio > 0:
        n_identity = int(len(train_ds) * identity_ratio)
        logger.info("Adding %d identity examples (%.0f%%)", n_identity, identity_ratio * 100)
        identity_indices = list(range(min(n_identity, len(train_ds))))
        identity_examples = train_ds.select(identity_indices)
        identity_data = {
            "input": identity_examples["target"],
            "target": identity_examples["target"],
        }
        identity_ds = Dataset.from_dict(identity_data)
        from datasets import concatenate_datasets
        train_ds = concatenate_datasets([train_ds, identity_ds]).shuffle(seed=seed)

    result = {"train": train_ds, "validation": val_ds}
    if test_ds is not None:
        result["test"] = test_ds

    for name, split in result.items():
        logger.info("  %s: %d examples", name, len(split))

    return result


def format_gec_example(input_text: str, target_text: str) -> str:
    """Format a single GEC example as a string."""
    return f"{TASK_FIX}{SRC}{input_text}{SEP}{target_text}"


@dataclass
class GECDataCollator:
    """Collator that tokenizes GEC examples on-the-fly and masks loss before <SEP>."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sep_id = self.tokenizer.convert_tokens_to_ids(SEP)
        eos_id = self.tokenizer.eos_token_id

        all_input_ids = []
        all_labels = []

        for ex in features:
            text = format_gec_example(ex["input"], ex["target"])
            text += self.tokenizer.eos_token

            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"]

            # Find <SEP> position — mask everything up to and including it
            labels = [-100] * len(input_ids)
            sep_found = False
            for i, tid in enumerate(input_ids):
                if tid == sep_id:
                    sep_found = True
                    continue
                if sep_found:
                    labels[i] = input_ids[i]

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ids, labs in zip(all_input_ids, all_labels):
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_attention_mask.append([1] * len(ids) + [0] * pad_len)
            batch_labels.append(labs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
