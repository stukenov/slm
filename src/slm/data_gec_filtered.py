"""Filtered GEC datasets by error type: morphology, word_order, orthography."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Literal

from datasets import Dataset, concatenate_datasets, load_from_disk

from slm.data_gec import load_gec_dataset

logger = logging.getLogger(__name__)

ErrorType = Literal["morphology", "word_order", "orthography"]


def classify_error(input_text: str, target_text: str) -> str:
    """Fast classification of GEC error type.

    Returns: 'morphology', 'word_order', 'mixed', or 'identity'.
    """
    inp = input_text.strip()
    tgt = target_text.strip()

    if inp == tgt:
        return "identity"

    inp_words = inp.split()
    tgt_words = tgt.split()

    # Same words, different order
    if len(inp_words) == len(tgt_words) and sorted(inp_words) == sorted(tgt_words):
        return "word_order"

    # Different word count = mixed (insertions/deletions)
    if len(inp_words) != len(tgt_words):
        return "mixed"

    # Same word count — compare word by word
    changed = 0
    all_small = True
    for sw, tw in zip(inp_words, tgt_words):
        if sw != tw:
            changed += 1
            # Check if change is small (<=3 char diffs)
            if len(sw) != len(tw) or sum(1 for a, b in zip(sw, tw) if a != b) > 3:
                all_small = False

    if changed == 0:
        return "identity"
    return "morphology" if all_small else "orthography"


def _classify_batch(batch: dict) -> dict:
    """Batch classification for datasets.map()."""
    labels = [
        classify_error(inp, tgt)
        for inp, tgt in zip(batch["input"], batch["target"])
    ]
    return {"_error_type": labels}


def load_filtered_gec_dataset(
    error_type: ErrorType,
    dataset_name: str = "saken-tukenov/sozkz-corpus-synthetic-kk-gec-v1",
    identity_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Load GEC dataset filtered to a specific error type.

    DDP-safe: rank 0 filters and saves to disk, other ranks wait and load.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cache_dir = Path(f"/tmp/gec_filtered_{error_type}")
    done_marker = cache_dir / "_done"

    if local_rank == 0:
        ds = load_gec_dataset(dataset_name=dataset_name, identity_ratio=0.0)
        result = {}
        for split_name, split_ds in ds.items():
            classified = split_ds.map(
                _classify_batch, batched=True, batch_size=10000,
                desc=f"Classifying {split_name}",
            )
            filtered = classified.filter(
                lambda x: x["_error_type"] == error_type,
                desc=f"Filtering {split_name}",
            )
            filtered = filtered.remove_columns(["_error_type"])

            logger.info(
                "%s/%s: %d -> %d examples (%.1f%%)",
                error_type, split_name, len(split_ds), len(filtered),
                len(filtered) * 100 / len(split_ds) if len(split_ds) > 0 else 0,
            )

            if split_name == "train" and identity_ratio > 0 and len(filtered) > 0:
                n_identity = int(len(filtered) * identity_ratio)
                identity_indices = list(range(min(n_identity, len(filtered))))
                identity_examples = filtered.select(identity_indices)
                identity_ds = Dataset.from_dict({
                    "input": identity_examples["target"],
                    "target": identity_examples["target"],
                })
                filtered = concatenate_datasets([filtered, identity_ds]).shuffle(seed=seed)
                logger.info("  + %d identity examples -> %d total", n_identity, len(filtered))

            filtered.save_to_disk(str(cache_dir / split_name))
            result[split_name] = filtered

        done_marker.touch()
        logger.info("Rank 0: filtered data saved to %s", cache_dir)
        return result
    else:
        logger.info("Rank %d: waiting for rank 0 to finish filtering...", local_rank)
        for _ in range(600):  # wait up to 10 min
            if done_marker.exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Rank {local_rank}: timed out waiting for filtered data")

        result = {}
        for split_name in ["train", "validation", "test"]:
            split_path = cache_dir / split_name
            if split_path.exists():
                result[split_name] = load_from_disk(str(split_path))
                logger.info("Rank %d: loaded %s (%d examples)", local_rank, split_name, len(result[split_name]))
        return result
