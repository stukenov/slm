"""Data loading and preprocessing for Kazakh language training."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    """Check if this is the main DDP process (rank 0) or single-process."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    return rank == 0


def _wait_for_cache(cache_path: Path, timeout: int = 7200) -> None:
    """Wait until cache directory is fully written by rank 0."""
    marker = cache_path / ".done"
    start = time.time()
    while not marker.exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for cache at {cache_path}")
        time.sleep(10)
        logger.info("Rank %s waiting for data cache...", os.environ.get("RANK", "0"))


def load_kazakh_dataset(
    dataset_name: str = "kz-transformers/multidomain-kazakh-dataset",
    text_column: str = "text",
    val_ratio: float = 0.05,
    seed: int = 42,
    dataset_split: str | None = None,
) -> dict[str, Dataset]:
    """Load the Kazakh dataset and create train/val split.

    Args:
        dataset_split: Optional HF split selector, e.g. "train[:1%]" to load
            only a subset. Useful for smoke tests.

    Returns a dict with ``train`` and ``validation`` keys.
    """
    logger.info("Loading dataset %s (split=%s)", dataset_name, dataset_split or "all")
    ds = load_dataset(dataset_name, split=dataset_split)

    # The dataset may already have splits; use train split if available
    if isinstance(ds, dict):
        if "train" in ds and "validation" in ds:
            return {"train": ds["train"], "validation": ds["validation"]}
        if "train" in ds:
            raw = ds["train"]
        else:
            # Take the first available split
            raw = next(iter(ds.values()))
    else:
        raw = ds

    # Check text column exists
    if text_column not in raw.column_names:
        available = raw.column_names
        # Try common alternatives
        for alt in ["text", "content", "sentence"]:
            if alt in available:
                text_column = alt
                break
        else:
            raise ValueError(
                f"Text column '{text_column}' not found. Available: {available}"
            )

    # Keep only text column
    raw = raw.select_columns([text_column])
    if text_column != "text":
        raw = raw.rename_column(text_column, "text")

    # Split
    split = raw.train_test_split(test_size=val_ratio, seed=seed)
    logger.info(
        "Dataset loaded: %d train, %d validation samples",
        len(split["train"]),
        len(split["test"]),
    )
    return {"train": split["train"], "validation": split["test"]}


def tokenize_and_group(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int = 512,
    num_proc: int = 4,
) -> Dataset:
    """Tokenize text and group into fixed-length blocks for CLM training."""

    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_attention_mask=False)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        # Concatenate all input_ids
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)

        # Truncate to multiple of block_size
        total_length = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_length]

        # Split into chunks
        result = {
            "input_ids": [
                concatenated[i : i + block_size]
                for i in range(0, total_length, block_size)
            ],
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping texts",
    )

    logger.info("Grouped dataset: %d blocks of size %d", len(grouped), block_size)
    return grouped


def _cache_key(tokenizer_name: str, dataset_name: str, block_size: int, seed: int) -> str:
    """Build a short hash key for caching tokenized datasets."""
    raw = f"{tokenizer_name}|{dataset_name}|{block_size}|{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "kz-transformers/multidomain-kazakh-dataset",
    block_size: int = 512,
    val_ratio: float = 0.05,
    seed: int = 42,
    num_proc: int = 4,
    cache_dir: str | None = "./data_cache",
    dataset_split: str | None = None,
) -> dict[str, Dataset]:
    """Full pipeline: load, tokenize, and group datasets.

    If ``cache_dir`` is set, tokenized+grouped datasets are saved to disk and
    reused on subsequent runs with the same tokenizer/block_size/seed combo.

    In multi-GPU (DDP) mode, only rank 0 prepares and caches the data.
    Other ranks wait for the cache and load from disk.
    """
    tok_name = getattr(tokenizer, "name_or_path", None) or str(type(tokenizer))
    key = _cache_key(tok_name, dataset_name, block_size, seed)
    main_process = _is_main_process()

    if cache_dir:
        cache_path = Path(cache_dir) / key
        marker = cache_path / ".done"

        # All ranks: if cache is ready, load it
        if marker.exists():
            logger.info("Loading cached tokenized dataset from %s", cache_path)
            return {
                "train": load_from_disk(str(cache_path / "train")),
                "validation": load_from_disk(str(cache_path / "validation")),
            }

        if not main_process:
            # Non-main ranks: wait for rank 0 to finish caching
            logger.info("Rank %s waiting for rank 0 to prepare data...", os.environ.get("RANK"))
            _wait_for_cache(cache_path)
            return {
                "train": load_from_disk(str(cache_path / "train")),
                "validation": load_from_disk(str(cache_path / "validation")),
            }

    # Only rank 0 (or single process) reaches here
    splits = load_kazakh_dataset(
        dataset_name=dataset_name,
        val_ratio=val_ratio,
        seed=seed,
        dataset_split=dataset_split,
    )

    result = {}
    for split_name, ds in splits.items():
        result[split_name] = tokenize_and_group(
            ds, tokenizer, block_size=block_size, num_proc=num_proc
        )

    if cache_dir:
        cache_path = Path(cache_dir) / key
        cache_path.mkdir(parents=True, exist_ok=True)
        for split_name, ds in result.items():
            ds.save_to_disk(str(cache_path / split_name))
        # Write marker so other ranks know cache is ready
        (cache_path / ".done").touch()
        logger.info("Tokenized dataset cached to %s", cache_path)

    return result
