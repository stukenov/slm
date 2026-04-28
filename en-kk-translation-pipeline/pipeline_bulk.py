#!/usr/bin/env python3
"""
Bulk FineWeb-Edu EN→KK translation pipeline.

Translates rows in chunks of 1M, uploads each as a parquet shard to HF.
Expandable: chunk indices are not tied to a fixed total count.

Usage:
    python pipeline_bulk.py --num-gpus 2 --filter --start-chunk 10 --end-chunk 20
    python pipeline_bulk.py --num-gpus 2 --filter --start-chunk auto  # auto-detect next chunk
"""

import argparse
import os
import time
from itertools import islice
from multiprocessing import Process, Queue

from datasets import Dataset
from huggingface_hub import HfApi

from pipeline import (
    worker_translate,
    split_sentences,
    translate_batch_sentences,
    DEFAULT_OUTPUT,
)
from filters import TextFilter

BASE = os.path.dirname(os.path.abspath(__file__))
HF_REPO = "saken-tukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1"
ROWS_PER_CHUNK = 1_000_000


def shard_filename(chunk_idx: int) -> str:
    return f"data/train-{chunk_idx:05d}.parquet"


def upload_shard(api: HfApi, local_path: str, chunk_idx: int):
    """Upload a shard parquet to HF and verify it exists."""
    remote_path = shard_filename(chunk_idx)
    print(f"Uploading {local_path} → {HF_REPO}/{remote_path}...", flush=True)
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    elapsed = time.time() - t0
    # Verify upload
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    if remote_path not in files:
        raise RuntimeError(f"Upload verification failed: {remote_path} not found on HF")
    print(f"Upload verified in {elapsed:.1f}s", flush=True)


def shard_exists(api: HfApi, chunk_idx: int) -> bool:
    """Check if shard already exists on HF."""
    remote_path = shard_filename(chunk_idx)
    try:
        files = api.list_repo_files(HF_REPO, repo_type="dataset")
        return remote_path in files
    except Exception:
        return False


def find_first_missing_chunk(api: HfApi) -> int:
    """Scan HF repo and return the first missing chunk index."""
    try:
        files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    except Exception:
        return 0
    idx = 0
    while shard_filename(idx) in files:
        idx += 1
    return idx


def process_chunk(
    chunk_idx: int,
    rows: list,
    args,
    api: HfApi,
):
    """Filter, translate across GPUs, merge, upload one chunk."""
    print(f"\n{'='*60}", flush=True)
    print(f"CHUNK {chunk_idx}: {len(rows)} rows loaded", flush=True)
    print(f"{'='*60}", flush=True)

    # Filter
    if args.filter:
        tf = TextFilter(fuzzy_dedup=False)
        t_filt = time.time()
        filtered = []
        for i, row in enumerate(rows):
            keep, reason = tf.filter(row.get("text", ""), doc_id=f"c{chunk_idx}_{i}")
            if keep:
                filtered.append(row)
        print(f"Filtered in {time.time() - t_filt:.1f}s", flush=True)
        print(tf.summary(), flush=True)
        rows = filtered

    print(f"Rows after filtering: {len(rows)}", flush=True)

    if len(rows) == 0:
        print(f"Chunk {chunk_idx}: no rows after filtering, skipping.", flush=True)
        return

    # Output path for this chunk
    chunk_output = os.path.join(BASE, f"chunk_{chunk_idx:05d}.parquet")

    num_gpus = args.num_gpus
    if num_gpus == 1:
        result_queue = Queue()
        worker_translate(
            gpu_id=0, rows=rows, row_offset=0,
            output_path=chunk_output,
            batch_size=args.batch_size, beam_size=args.beam_size,
            max_input_length=args.max_input_length,
            max_decoding_length=args.max_decoding_length,
            compute_type=args.compute_type,
            checkpoint_every=args.checkpoint_every,
            resume=True, result_queue=result_queue,
        )
        _, part_path, count = result_queue.get()
        os.rename(part_path, chunk_output)
    else:
        chunk_size = len(rows) // num_gpus
        result_queue = Queue()
        processes = []

        for gpu_id in range(num_gpus):
            start = gpu_id * chunk_size
            end = start + chunk_size if gpu_id < num_gpus - 1 else len(rows)
            gpu_rows = rows[start:end]
            print(f"[GPU:{gpu_id}] {len(gpu_rows)} rows", flush=True)

            p = Process(
                target=worker_translate,
                args=(
                    gpu_id, gpu_rows, start, chunk_output,
                    args.batch_size, args.beam_size,
                    args.max_input_length, args.max_decoding_length,
                    args.compute_type, args.checkpoint_every,
                    True, result_queue,
                ),
            )
            p.start()
            processes.append(p)

        results = []
        for _ in processes:
            results.append(result_queue.get())
        for p in processes:
            p.join()

        # Merge GPU parts
        all_rows = []
        for gpu_id, part_path, count in sorted(results):
            part_ds = Dataset.from_parquet(part_path)
            all_rows.extend(part_ds.to_list())
            print(f"  GPU:{gpu_id} → {count} rows", flush=True)

        Dataset.from_list(all_rows).to_parquet(chunk_output)
        print(f"Merged chunk: {chunk_output} ({len(all_rows)} rows)", flush=True)

        # Cleanup GPU part files
        for gpu_id, part_path, _ in results:
            if os.path.exists(part_path):
                os.remove(part_path)
        # Cleanup checkpoint files
        import glob as globmod
        for f in globmod.glob(chunk_output.replace(".parquet", "_gpu*_ckpt_*.parquet")):
            os.remove(f)

    # Upload (verified)
    upload_shard(api, chunk_output, chunk_idx)

    # Cleanup local chunk file only after verified upload
    if os.path.exists(chunk_output):
        os.remove(chunk_output)

    print(f"Chunk {chunk_idx} complete.\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Bulk FineWeb-Edu EN→KK translation pipeline")
    parser.add_argument("--start-chunk", type=str, default="0",
                        help="Start chunk index, or 'auto' to detect first missing")
    parser.add_argument("--end-chunk", type=int, default=None,
                        help="End chunk index (exclusive). Default: start + 10")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-decoding-length", type=int, default=200)
    parser.add_argument("--compute-type", type=str, default="float16")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--filter", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    t_total = time.time()

    # Resolve start chunk
    if args.start_chunk == "auto":
        start_chunk = find_first_missing_chunk(api)
        print(f"Auto-detected start chunk: {start_chunk}", flush=True)
    else:
        start_chunk = int(args.start_chunk)

    end_chunk = args.end_chunk if args.end_chunk is not None else start_chunk + 10

    print(f"Bulk pipeline: chunks {start_chunk}–{end_chunk - 1} × {ROWS_PER_CHUNK} rows", flush=True)
    print(f"HF repo: {HF_REPO}", flush=True)
    print(f"GPUs: {args.num_gpus}, filter: {args.filter}", flush=True)

    # Find which chunks actually need processing
    chunks_todo = []
    for chunk_idx in range(start_chunk, end_chunk):
        if shard_exists(api, chunk_idx):
            print(f"Chunk {chunk_idx}: shard already exists on HF, skipping.", flush=True)
        else:
            chunks_todo.append(chunk_idx)

    if not chunks_todo:
        print("All chunks already uploaded!", flush=True)
    else:
        # Stream once, skip to the first needed offset
        first_offset = chunks_todo[0] * ROWS_PER_CHUNK
        last_end = (chunks_todo[-1] + 1) * ROWS_PER_CHUNK
        total_to_stream = last_end - first_offset

        print(f"\nStreaming FineWeb-Edu rows {first_offset}–{last_end} ({total_to_stream} rows)...", flush=True)
        from datasets import load_dataset
        ds_stream = load_dataset(
            "HuggingFaceFW/fineweb-edu-score-2", split="train", streaming=True,
        )
        stream_iter = iter(islice(ds_stream, first_offset, last_end))

        for chunk_idx in range(chunks_todo[0], chunks_todo[-1] + 1):
            t_load = time.time()
            rows = list(islice(stream_iter, ROWS_PER_CHUNK))
            print(f"\nChunk {chunk_idx}: loaded {len(rows)} rows in {time.time() - t_load:.1f}s", flush=True)

            if chunk_idx not in chunks_todo:
                print(f"Chunk {chunk_idx}: already on HF, skipping processing.", flush=True)
                continue

            process_chunk(chunk_idx, rows, args, api)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE in {elapsed / 3600:.1f}h", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
