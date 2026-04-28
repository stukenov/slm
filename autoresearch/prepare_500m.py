"""
Data preparation for 500M Qwen2 Kazakh model. Downloads KK-only tokenized dataset
(morphbpe-100k), writes train.bin + val.bin.

Usage: uv run prepare_500m.py
"""
import os, sys, math, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch

MAX_SEQ_LEN = 1024
VOCAB_SIZE = 100_000
ASSESSMENT_TOKENS = 10_000_000

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-500m")
DATA_DIR = os.path.join(CACHE_DIR, "data")
SHARD_DIR_KK = os.path.join(DATA_DIR, "shards_kk")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")

HF_REPO_KK = "stukenov/sozkz-corpus-tokenized-kk-morphbpe100k-v1"


def download_one_shard(args):
    idx, shard_file, total, repo, shard_dir = args
    import pyarrow.parquet as pq
    import requests

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{shard_file}"
    local = os.path.join(shard_dir, f"shard_{idx:04d}.npy")

    if os.path.exists(local):
        arr = np.load(local)
        print(f"  [{idx+1}/{total}] cached: {len(arr):,} tokens")
        return idx, arr

    t0 = time.time()
    pq_path = os.path.join(shard_dir, os.path.basename(shard_file))
    for attempt in range(3):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(pq_path, "wb") as f:
                for chunk in resp.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  [{idx+1}/{total}] retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    dl_time = time.time() - t0

    t1 = time.time()
    try:
        table = pq.read_table(pq_path, columns=["input_ids"])
        col = table.column("input_ids")
        flat = col.combine_chunks()
        values = flat.values
        # uint32 because vocab=100K > 65535
        arr = values.to_numpy(zero_copy_only=False).astype(np.uint32)
    except Exception as e:
        print(f"  [{idx+1}/{total}] corrupt parquet, retrying: {e}")
        if os.path.exists(pq_path):
            os.remove(pq_path)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(pq_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        table = pq.read_table(pq_path, columns=["input_ids"])
        col = table.column("input_ids")
        flat = col.combine_chunks()
        values = flat.values
        arr = values.to_numpy(zero_copy_only=False).astype(np.uint32)
    parse_time = time.time() - t1

    np.save(local, arr)
    if os.path.exists(pq_path):
        os.remove(pq_path)
    print(f"  [{idx+1}/{total}] {os.path.basename(shard_file)}: {len(arr):,} tok, dl={dl_time:.0f}s parse={parse_time:.0f}s")
    return idx, arr


def download_dataset(repo, shard_dir, num_shards):
    os.makedirs(shard_dir, exist_ok=True)
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    shard_files = sorted([f for f in files if f.endswith(".parquet") and "/train" in f])
    if not shard_files:
        shard_files = sorted([f for f in files if f.startswith("data/train") and f.endswith(".parquet")])
    print(f"  {repo}: found {len(shard_files)} shards, using {min(num_shards, len(shard_files))}")

    n = min(num_shards, len(shard_files))
    tasks = [(i, shard_files[i], n, repo, shard_dir) for i in range(n)]
    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_one_shard, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            idx, arr = fut.result()
            results[idx] = arr

    ordered = [results[i] for i in range(n)]
    combined = np.concatenate(ordered)
    print(f"  {repo}: {len(combined):,} tokens total")
    return combined


def prepare_data(num_shards=999):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
        train_tok = os.path.getsize(TRAIN_BIN) // 4  # uint32 = 4 bytes
        val_tok = os.path.getsize(VAL_BIN) // 4
        print(f"Data cached: train={train_tok:,} val={val_tok:,}")
        return

    print(f"\n=== Downloading KK dataset ({num_shards} shards) ===")
    kk_data = download_dataset(HF_REPO_KK, SHARD_DIR_KK, num_shards)

    total = len(kk_data)
    print(f"\n=== Total: {total:,} tokens ({total/1e9:.2f}B) ===")

    # Val split: last 10M tokens
    val_size = min(total // 20, 10_000_000)
    val_arr = kk_data[-val_size:]
    train_arr = kk_data[:-val_size]

    train_arr.tofile(TRAIN_BIN)
    val_arr.tofile(VAL_BIN)
    print(f"  Train: {len(train_arr):,} tokens ({os.path.getsize(TRAIN_BIN)/1e9:.1f}GB)")
    print(f"  Val:   {len(val_arr):,} tokens ({os.path.getsize(VAL_BIN)/1e9:.1f}GB)")

    import shutil
    shutil.rmtree(SHARD_DIR_KK, ignore_errors=True)
    print("Done!")


class Dataloader:
    def __init__(self, split, batch_size, seq_len=MAX_SEQ_LEN):
        path = TRAIN_BIN if split == "train" else VAL_BIN
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
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
    BYTES_PER_TOKEN = 6.5  # morphbpe-100k has longer tokens than 50K BPE
    avg_loss_nats = total_loss / total_tokens
    bpb = avg_loss_nats / math.log(2) / BYTES_PER_TOKEN
    if was_training:
        model.train()
    return bpb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=999)
    args = parser.parse_args()
    prepare_data(args.shards)
