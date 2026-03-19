"""
Data preparation for Kazakh autoresearch. Fast parallel download.
Usage: uv run prepare.py [--num-shards 44]
"""
import os, sys, math, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch

MAX_SEQ_LEN = 1024
TIME_BUDGET = 300
VOCAB_SIZE = 50257
ASSESSMENT_TOKENS = 10_000_000

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-kazakh")
DATA_DIR = os.path.join(CACHE_DIR, "data")
SHARD_DIR = os.path.join(DATA_DIR, "shards")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
HF_REPO = "stukenov/sozkz-corpus-tokenized-kk-llama50k-v3"


def download_one_shard(args):
    """Download and convert one shard to numpy. Returns (index, flat_array)."""
    idx, shard_file, total = args
    import pyarrow.parquet as pq
    import requests

    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{shard_file}"
    local = os.path.join(SHARD_DIR, f"shard_{idx:04d}.npy")

    if os.path.exists(local):
        arr = np.load(local)
        print(f"  [{idx+1}/{total}] cached: {len(arr):,} tokens")
        return idx, arr

    t0 = time.time()
    # Download parquet (with retry)
    pq_path = os.path.join(SHARD_DIR, os.path.basename(shard_file))
    for attempt in range(3):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(pq_path, "wb") as f:
                for chunk in resp.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            break
        except (requests.RequestException, IOError) as e:
            if attempt == 2:
                raise
            print(f"  [{idx+1}/{total}] retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    dl_time = time.time() - t0

    # Parse parquet -> flat numpy (fast path)
    t1 = time.time()
    table = pq.read_table(pq_path, columns=["input_ids"])
    # Each row is a list of 1024 ints. Flatten via pyarrow.
    col = table.column("input_ids")
    # list<int32> -> flatten to 1D
    flat = col.combine_chunks()
    values = flat.values  # pyarrow.Int32Array (all values concatenated)
    arr = values.to_numpy(zero_copy_only=False).astype(np.uint16)
    parse_time = time.time() - t1

    # Cache as .npy and remove parquet
    np.save(local, arr)
    os.remove(pq_path)

    print(f"  [{idx+1}/{total}] {os.path.basename(shard_file)}: {len(arr):,} tok, dl={dl_time:.0f}s parse={parse_time:.0f}s")
    return idx, arr


def prepare_data(num_train_shards=44, num_val_shards=1):
    os.makedirs(SHARD_DIR, exist_ok=True)

    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        train_tok = os.path.getsize(TRAIN_BIN) // 2
        val_tok = os.path.getsize(VAL_BIN) // 2
        print(f"Data cached: train={train_tok:,} val={val_tok:,}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    train_files = sorted([f for f in files if "/train/" in f and f.endswith(".parquet")])
    val_files = sorted([f for f in files if "/validation/" in f and f.endswith(".parquet")])
    if not train_files:
        train_files = sorted([f for f in files if f.startswith("data/train-") and f.endswith(".parquet")])
    if not val_files:
        val_files = sorted([f for f in files if f.startswith("data/validation-") and f.endswith(".parquet")])
    print(f"Found {len(train_files)} train, {len(val_files)} val shards")

    # Parallel download train shards (8 threads)
    n = min(num_train_shards, len(train_files))
    print(f"\nDownloading {n} train shards (8 threads)...")
    t0 = time.time()
    tasks = [(i, train_files[i], n) for i in range(n)]
    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_one_shard, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            idx, arr = fut.result()
            results[idx] = arr

    # Concatenate in order
    print("  Concatenating...")
    ordered = [results[i] for i in range(n)]
    train_arr = np.concatenate(ordered)
    train_arr.tofile(TRAIN_BIN)
    dt = time.time() - t0
    print(f"  Train: {len(train_arr):,} tokens ({os.path.getsize(TRAIN_BIN)/1e9:.1f}GB) in {dt:.0f}s")
    del ordered, results, train_arr

    # Validation (single shard)
    print(f"\nDownloading {num_val_shards} val shard...")
    _, val_arr = download_one_shard((0, val_files[0], 1))
    val_arr.tofile(VAL_BIN)
    print(f"  Val: {len(val_arr):,} tokens ({os.path.getsize(VAL_BIN)/1e9:.1f}GB)")

    # Cleanup .npy cache
    import shutil
    shutil.rmtree(SHARD_DIR, ignore_errors=True)
    print("\nDone!")


class Dataloader:
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
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def make_dataloader(split, batch_size, seq_len=MAX_SEQ_LEN):
    return Dataloader(split, batch_size, seq_len)


@torch.no_grad()
def compute_val_bpb(model, batch_size, device="cuda", seq_len=MAX_SEQ_LEN):
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
    BYTES_PER_TOKEN = 5.2
    avg_loss_nats = total_loss / total_tokens
    bpb = avg_loss_nats / math.log(2) / BYTES_PER_TOKEN
    if was_training:
        model.train()
    return bpb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=44)
    parser.add_argument("--num-val-shards", type=int, default=1)
    args = parser.parse_args()
    prepare_data(args.num_shards, args.num_val_shards)
