"""T5 span corruption data pipeline for Kazakh language pretraining."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from slm.data import (
    _cache_key,
    _is_main_process,
    _wait_for_cache,
    load_kazakh_dataset,
    tokenize_and_group,
)

logger = logging.getLogger(__name__)

NUM_SENTINEL_TOKENS = 128


def add_sentinel_tokens(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Add <extra_id_0> ... <extra_id_127> sentinel tokens to tokenizer if missing."""
    sentinels = [f"<extra_id_{i}>" for i in range(NUM_SENTINEL_TOKENS)]
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in sentinels if t not in existing]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info("Added %d sentinel tokens to tokenizer (vocab=%d)", len(new_tokens), len(tokenizer))
    return tokenizer


def prepare_t5_datasets(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    block_size: int = 512,
    val_ratio: float = 0.05,
    seed: int = 42,
    num_proc: int = 4,
    cache_dir: str | None = "./data_cache",
    dataset_split: str | None = None,
) -> dict[str, Dataset]:
    """Load text, tokenize into fixed-length blocks (no labels — collator adds them).

    Reuses the same caching logic as the CLM pipeline but with a t5- prefix key.
    """
    from datasets import load_from_disk

    tok_name = getattr(tokenizer, "name_or_path", None) or str(type(tokenizer))
    key = "t5-" + _cache_key(tok_name, dataset_name, block_size, seed)
    main_process = _is_main_process()

    if cache_dir:
        cache_path = Path(cache_dir) / key
        marker = cache_path / ".done"

        if marker.exists():
            logger.info("Loading cached T5 dataset from %s", cache_path)
            return {
                "train": load_from_disk(str(cache_path / "train")),
                "validation": load_from_disk(str(cache_path / "validation")),
            }

        if not main_process:
            logger.info("Rank waiting for rank 0 to prepare T5 data...")
            _wait_for_cache(cache_path)
            return {
                "train": load_from_disk(str(cache_path / "train")),
                "validation": load_from_disk(str(cache_path / "validation")),
            }

    splits = load_kazakh_dataset(
        dataset_name=dataset_name,
        val_ratio=val_ratio,
        seed=seed,
        dataset_split=dataset_split,
    )

    # Tokenize and group into blocks (labels column will be ignored by collator)
    result = {}
    for split_name, ds in splits.items():
        grouped = tokenize_and_group(ds, tokenizer, block_size=block_size, num_proc=num_proc)
        # Remove labels column — T5 collator creates its own targets
        if "labels" in grouped.column_names:
            grouped = grouped.remove_columns(["labels"])
        result[split_name] = grouped

    if cache_dir:
        cache_path = Path(cache_dir) / key
        cache_path.mkdir(parents=True, exist_ok=True)
        for split_name, ds in result.items():
            ds.save_to_disk(str(cache_path / split_name))
        (cache_path / ".done").touch()
        logger.info("T5 dataset cached to %s", cache_path)

    return result


@dataclass
class T5SpanCorruptionCollator:
    """On-the-fly T5 span corruption collator (follows original T5 paper).

    Takes batches of input_ids (flat token blocks) and produces:
    - input_ids: tokens with masked spans replaced by sentinel tokens
    - labels: sentinel token + original span tokens for each masked span

    Sentinel ordering: <extra_id_0> for the first span, <extra_id_1> for the
    second, etc. — matching the T5 convention.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_prob: float = 0.15
    mean_noise_span_length: float = 3.0

    def __post_init__(self):
        vocab = self.tokenizer.get_vocab()
        # sentinel_ids[0] = token id for <extra_id_0>, etc.
        self.sentinel_ids = [vocab[f"<extra_id_{i}>"] for i in range(NUM_SENTINEL_TOKENS)]
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def __call__(self, examples: list[dict]) -> dict:
        input_ids_list = [ex["input_ids"] for ex in examples]
        batch_inputs = []
        batch_targets = []

        for ids in input_ids_list:
            ids = np.array(ids, dtype=np.int64)
            inp, tgt = self._corrupt(ids)
            batch_inputs.append(inp)
            batch_targets.append(tgt)

        # Pad to max length in batch
        max_inp = max(len(x) for x in batch_inputs)
        max_tgt = max(len(x) for x in batch_targets)

        padded_inputs = np.full((len(batch_inputs), max_inp), self.pad_token_id, dtype=np.int64)
        padded_targets = np.full((len(batch_targets), max_tgt), -100, dtype=np.int64)
        attention_mask = np.zeros((len(batch_inputs), max_inp), dtype=np.int64)

        for i, (inp, tgt) in enumerate(zip(batch_inputs, batch_targets)):
            padded_inputs[i, : len(inp)] = inp
            padded_targets[i, : len(tgt)] = tgt
            attention_mask[i, : len(inp)] = 1

        return {
            "input_ids": torch.tensor(padded_inputs),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(padded_targets),
        }

    def _corrupt(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply span corruption to a single sequence.

        1. Create a boolean noise mask via random span selection.
        2. Build encoder input: replace each contiguous noise span with one sentinel.
        3. Build decoder target: for each span, sentinel + original tokens.
        """
        length = len(ids)
        noise_mask = self._random_spans_noise_mask(length)

        # Assign a span id to each noise token (contiguous noise regions get same id)
        # Non-noise tokens get -1
        span_ids = self._noise_span_ids(noise_mask)
        num_spans = span_ids.max() + 1 if noise_mask.any() else 0
        num_spans = min(num_spans, NUM_SENTINEL_TOKENS)

        # Build encoder input: non-noise tokens + sentinels replacing each span
        input_tokens = []
        prev_span_id = -1
        for i in range(length):
            if noise_mask[i]:
                sid = span_ids[i]
                if sid != prev_span_id and sid < NUM_SENTINEL_TOKENS:
                    input_tokens.append(self.sentinel_ids[sid])
                    prev_span_id = sid
            else:
                input_tokens.append(ids[i])
                prev_span_id = -1

        # Build decoder target: sentinel + span tokens for each span
        target_tokens = []
        prev_span_id = -1
        for i in range(length):
            if noise_mask[i]:
                sid = span_ids[i]
                if sid >= NUM_SENTINEL_TOKENS:
                    continue
                if sid != prev_span_id:
                    target_tokens.append(self.sentinel_ids[sid])
                    prev_span_id = sid
                target_tokens.append(ids[i])

        # Final sentinel to mark end of target
        if num_spans < NUM_SENTINEL_TOKENS:
            target_tokens.append(self.sentinel_ids[num_spans])

        return np.array(input_tokens, dtype=np.int64), np.array(target_tokens, dtype=np.int64)

    def _random_spans_noise_mask(self, length: int) -> np.ndarray:
        """Create a random boolean mask for span corruption.

        Follows the T5 approach: compute number of noise/nonnoise spans,
        then interleave random-length spans.
        """
        num_noise_tokens = int(round(length * self.mask_prob))
        num_noise_tokens = max(num_noise_tokens, 1)
        num_noise_tokens = min(num_noise_tokens, length - 1)  # keep at least 1 non-noise

        num_noise_spans = int(round(num_noise_tokens / self.mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        num_noise_spans = min(num_noise_spans, num_noise_tokens)  # can't have more spans than noise tokens
        num_noise_spans = min(num_noise_spans, NUM_SENTINEL_TOKENS)

        num_nonnoise_tokens = length - num_noise_tokens

        # Randomly split noise tokens into num_noise_spans groups
        noise_span_lengths = self._random_partition(num_noise_tokens, num_noise_spans)
        # Randomly split non-noise tokens into (num_noise_spans + 1) groups
        # (before first span, between spans, after last span)
        nonnoise_span_lengths = self._random_partition(num_nonnoise_tokens, num_noise_spans + 1)

        # Interleave: [nonnoise_0, noise_0, nonnoise_1, noise_1, ..., nonnoise_n]
        mask = np.zeros(length, dtype=bool)
        pos = 0
        for i in range(num_noise_spans):
            pos += nonnoise_span_lengths[i]  # skip non-noise segment
            end = min(pos + noise_span_lengths[i], length)
            mask[pos:end] = True
            pos = end
        # remaining nonnoise_span_lengths[-1] goes to end (already False)

        return mask

    @staticmethod
    def _noise_span_ids(noise_mask: np.ndarray) -> np.ndarray:
        """Assign incrementing span IDs to contiguous noise regions.

        Non-noise tokens get -1.
        """
        span_ids = np.full(len(noise_mask), -1, dtype=np.int32)
        current_id = -1
        prev_noise = False
        for i in range(len(noise_mask)):
            if noise_mask[i]:
                if not prev_noise:
                    current_id += 1
                span_ids[i] = current_id
                prev_noise = True
            else:
                prev_noise = False
        return span_ids

    @staticmethod
    def _random_partition(total: int, num_parts: int) -> list[int]:
        """Randomly partition `total` into `num_parts` non-negative integers.

        Uses the multinomial/Dirichlet-like approach: place (num_parts - 1)
        dividers among `total` items.
        """
        if num_parts <= 0:
            return []
        if num_parts == 1:
            return [total]
        if total <= 0:
            return [0] * num_parts

        # Place (num_parts - 1) dividers at random positions in [0, total]
        dividers = sorted(np.random.randint(0, total + 1, size=num_parts - 1))
        dividers = [0] + list(dividers) + [total]
        return [dividers[i + 1] - dividers[i] for i in range(num_parts)]
