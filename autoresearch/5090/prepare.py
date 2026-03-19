"""
Data preparation for Kazakh autoresearch.
Downloads pretokenized Kazakh data from HuggingFace, caches locally.
Provides dataloader and validation functions. DO NOT MODIFY.

Usage:
    uv run prepare.py
    uv run prepare.py --num-shards 2   # fewer shards for testing
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024        # context length (matches pretokenized block size)
TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
VOCAB_SIZE = 50257         # sozkz-core-gpt2-50k-kk-base-v1
ASSESSMENT_TOKENS = 10_000_000  # ~10M tokens for validation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-kazakh")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")

HF_REPO = "stukenov/sozkz-corpus-tokenized-kk-llama50k-v3"

# ---------------------------------------------------------------------------
# Data download — direct parquet download (no full dataset cache)
# ---------------------------------------------------------------------------

def download_file(url, dest):
    """Download a file with progress."""
    import requests
    print(f"  Downloading {os.path.basename(dest)}...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def prepare_data(num_train_shards=3, num_val_shards=1):
    """Download parquet shards directly and convert to flat .bin files."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        train_tokens = os.path.getsize(TRAIN_BIN) // 2  # uint16
        val_tokens = os.path.getsize(VAL_BIN) // 2
        print(f"Data cached: train={train_tokens:,} tokens, val={val_tokens:,} tokens")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    train_files = sorted([f for f in files if "/train/" in f and f.endswith(".parquet")])
    val_files = sorted([f for f in files if "/validation/" in f and f.endswith(".parquet")])
    # Fallback: some repos use data/train-XXXXX pattern
    if not train_files:
        train_files = sorted([f for f in files if f.startswith("data/train-") and f.endswith(".parquet")])
    if not val_files:
        val_files = sorted([f for f in files if f.startswith("data/validation-") and f.endswith(".parquet")])

    print(f"Found {len(train_files)} train shards, {len(val_files)} val shards")

    # Download and convert training data
    print(f"\nPreparing train data ({num_train_shards} shards)...")
    train_tokens = []
    for shard_file in train_files[:num_train_shards]:
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{shard_file}"
        local = os.path.join(DATA_DIR, os.path.basename(shard_file))
        if not os.path.exists(local):
            download_file(url, local)
        table = pq.read_table(local, columns=["input_ids"])
        for row in table.to_pydict()["input_ids"]:
            train_tokens.extend(row)
        os.remove(local)  # free disk immediately
        print(f"    {os.path.basename(shard_file)}: +{len(row)} tokens/row, total {len(train_tokens):,}")

    train_arr = np.array(train_tokens, dtype=np.uint16)
    train_arr.tofile(TRAIN_BIN)
    print(f"  Train: {len(train_arr):,} tokens -> {TRAIN_BIN} ({os.path.getsize(TRAIN_BIN)/1e6:.0f} MB)")
    del train_tokens, train_arr

    # Download and convert validation data
    print(f"\nPreparing val data ({num_val_shards} shards)...")
    val_tokens = []
    for shard_file in val_files[:num_val_shards]:
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{shard_file}"
        local = os.path.join(DATA_DIR, os.path.basename(shard_file))
        if not os.path.exists(local):
            download_file(url, local)
        table = pq.read_table(local, columns=["input_ids"])
        for row in table.to_pydict()["input_ids"]:
            val_tokens.extend(row)
        os.remove(local)
        print(f"    {os.path.basename(shard_file)}: total {len(val_tokens):,}")

    val_arr = np.array(val_tokens, dtype=np.uint16)
    val_arr.tofile(VAL_BIN)
    print(f"  Val: {len(val_arr):,} tokens -> {VAL_BIN} ({os.path.getsize(VAL_BIN)/1e6:.0f} MB)")

    print("\nDone!")


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

class Dataloader:
    """Flat-array random-offset dataloader (mmap, zero-copy)."""

    def __init__(self, split, batch_size, seq_len=MAX_SEQ_LEN):
        path = TRAIN_BIN if split == "train" else VAL_BIN
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_tokens = len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        max_start = self.n_tokens - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=self.batch_size)
        x = np.stack([self.data[s : s + self.seq_len] for s in starts])
        y = np.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        return x, y


def make_dataloader(split, batch_size, seq_len=MAX_SEQ_LEN):
    return Dataloader(split, batch_size, seq_len)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_bpb(model, batch_size, device="cuda", seq_len=MAX_SEQ_LEN):
    """
    Run model on validation set. Returns bits-per-byte (BPB).
    Lower is better. Vocab-size-independent metric.
    """
    was_training = model.training
    model.eval()
    loader = make_dataloader("val", batch_size, seq_len)
    total_loss = 0.0
    total_tokens = 0
    target_tokens = min(ASSESSMENT_TOKENS, loader.n_tokens - seq_len)

    while total_tokens < target_tokens:
        x, y = next(loader)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    # Kazakh BPE-50K: ~5.2 bytes per token
    BYTES_PER_TOKEN = 5.2
    avg_loss_nats = total_loss / total_tokens
    bpb = avg_loss_nats / math.log(2) / BYTES_PER_TOKEN

    if was_training:
        model.train()
    return bpb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=3,
                        help="Training shards to download (each ~200K rows, default=3 = ~600K rows = ~600M tokens)")
    parser.add_argument("--num-val-shards", type=int, default=1,
                        help="Validation shards (default=1)")
    args = parser.parse_args()
    prepare_data(args.num_shards, args.num_val_shards)
