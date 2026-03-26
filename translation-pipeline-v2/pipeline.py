#!/usr/bin/env python3
"""
FineWeb-Edu EN→KK translation pipeline v2 — orchestrator.

Streams source dataset in 1M-doc chunks, translates sentence-by-sentence
with confidence scoring, applies quality filters, uploads to HF Hub.

Usage:
    # Smoke test (10 rows, CPU)
    python pipeline.py --smoke-test

    # Single GPU, first 2 chunks
    python pipeline.py --num-gpus 1 --start-chunk 0 --end-chunk 2

    # Multi-GPU, auto-resume
    python pipeline.py --num-gpus 2 --start-chunk auto
"""

import argparse
import glob as globmod
import os
import time
from itertools import islice
from multiprocessing import Process, Queue

import xxhash

from config import (
    SOURCE_DATASET,
    SOURCE_CONFIG,
    ROWS_PER_CHUNK,
    HF_REPO,
    CHECKPOINT_EVERY,
)
from sentence_splitter import split_document
from postprocessor import process_document

BASE = os.path.dirname(os.path.abspath(__file__))


def content_hash(text: str) -> str:
    """Compute xxhash of text for deduplication."""
    return xxhash.xxh64_hexdigest(text)


def shard_filename(chunk_idx: int) -> str:
    return f"data/sample-10BT-{chunk_idx:05d}.parquet"


def shard_exists(api, chunk_idx: int) -> bool:
    try:
        files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
        return shard_filename(chunk_idx) in files
    except Exception:
        return False


def find_first_missing_chunk(api) -> int:
    try:
        files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    except Exception:
        return 0
    idx = 0
    while shard_filename(idx) in files:
        idx += 1
    return idx


def process_rows(rows, translator, verbose=True):
    """Process a batch of rows: split → translate → postprocess.

    Returns list of output dicts ready for parquet.
    """
    docs = []
    sentence_texts = []
    sentence_map = []  # (global_sent_idx, doc_idx, sent_idx_in_doc)

    for doc_idx, row in enumerate(rows):
        text = row.get("text", "")
        doc = split_document(text)
        docs.append((row, doc))

        for sent in doc["sentences"]:
            if not sent["skipped"]:
                sentence_map.append((len(sentence_texts), doc_idx, sent["sent_idx"]))
                sentence_texts.append(sent["text"])

    if verbose:
        print(f"  Split {len(rows)} docs → {len(sentence_texts)} sentences to translate", flush=True)

    if sentence_texts:
        translation_results = translator.translate_sentences(sentence_texts, verbose=verbose)
    else:
        translation_results = []

    # Map translations back to documents
    doc_translations = {}
    for (global_idx, doc_idx, sent_idx), tr_result in zip(sentence_map, translation_results):
        doc_translations.setdefault(doc_idx, {})[sent_idx] = tr_result

    # Postprocess each document
    output_rows = []
    for doc_idx, (row, doc) in enumerate(docs):
        text_en = row.get("text", "")
        original_id = row.get("id", "")
        translations = doc_translations.get(doc_idx, {})
        result = process_document(doc, translations, text_en)

        output_rows.append({
            "original_id": original_id,
            "content_hash": content_hash(text_en),
            "text_en": text_en,
            "text_kk": result["text_kk"],
            "confidence_mean": result["confidence_mean"],
            "confidence_min": result["confidence_min"],
            "sentences_total": result["sentences_total"],
            "sentences_translated": result["sentences_translated"],
            "sentences_skipped": result["sentences_skipped"],
        })

    return output_rows


def worker_process_chunk(gpu_id, rows, output_path, checkpoint_every, result_queue):
    """Worker process for one GPU."""
    from translator import Translator
    from datasets import Dataset

    print(f"[GPU:{gpu_id}] Loading translator...", flush=True)
    translator = Translator(device="cuda", device_index=gpu_id)

    all_output = []
    t_start = time.time()

    for chunk_start in range(0, len(rows), checkpoint_every):
        chunk_end = min(chunk_start + checkpoint_every, len(rows))
        chunk_rows = rows[chunk_start:chunk_end]

        print(f"[GPU:{gpu_id}] Processing rows {chunk_start}–{chunk_end}...", flush=True)
        output = process_rows(chunk_rows, translator)
        all_output.extend(output)

        if chunk_end < len(rows):
            ckpt_path = output_path.replace(".parquet", f"_gpu{gpu_id}_ckpt_{chunk_end}.parquet")
            Dataset.from_list(all_output).to_parquet(ckpt_path)
            print(f"[GPU:{gpu_id}] Checkpoint at {chunk_end}: {len(all_output)} rows", flush=True)

    elapsed = time.time() - t_start
    print(f"[GPU:{gpu_id}] Done: {len(all_output)} rows in {elapsed:.1f}s", flush=True)

    final_path = output_path.replace(".parquet", f"_gpu{gpu_id}.parquet")
    Dataset.from_list(all_output).to_parquet(final_path)
    result_queue.put((gpu_id, final_path, len(all_output)))


def process_chunk(chunk_idx, rows, args, api):
    """Process one 1M-row chunk: translate, merge GPUs, upload."""
    from datasets import Dataset

    print(f"\n{'='*60}", flush=True)
    print(f"CHUNK {chunk_idx}: {len(rows)} rows", flush=True)
    print(f"{'='*60}", flush=True)

    chunk_output = os.path.join(BASE, f"chunk_{chunk_idx:05d}.parquet")

    if args.num_gpus == 1:
        result_queue = Queue()
        worker_process_chunk(0, rows, chunk_output, args.checkpoint_every, result_queue)
        _, final_path, count = result_queue.get()
        merged_path = chunk_output
        os.rename(final_path, merged_path)
    else:
        chunk_size = len(rows) // args.num_gpus
        result_queue = Queue()
        processes = []

        for gpu_id in range(args.num_gpus):
            start = gpu_id * chunk_size
            end = start + chunk_size if gpu_id < args.num_gpus - 1 else len(rows)
            gpu_rows = rows[start:end]

            p = Process(
                target=worker_process_chunk,
                args=(gpu_id, gpu_rows, chunk_output, args.checkpoint_every, result_queue),
            )
            p.start()
            processes.append(p)

        results = []
        for _ in processes:
            results.append(result_queue.get())
        for p in processes:
            p.join()

        all_rows = []
        for gpu_id, final_path, count in sorted(results):
            ds = Dataset.from_parquet(final_path)
            all_rows.extend(ds.to_list())
            os.remove(final_path)
            print(f"  GPU:{gpu_id} → {count} rows", flush=True)

        merged_path = chunk_output
        Dataset.from_list(all_rows).to_parquet(merged_path)

        for f in globmod.glob(chunk_output.replace(".parquet", "_gpu*")):
            os.remove(f)

    # Upload
    remote_path = shard_filename(chunk_idx)
    print(f"Uploading {merged_path} → {HF_REPO}/{remote_path}...", flush=True)
    api.upload_file(
        path_or_fileobj=merged_path,
        path_in_repo=remote_path,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"Chunk {chunk_idx} uploaded.", flush=True)
    os.remove(merged_path)


def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu EN→KK pipeline v2")
    parser.add_argument("--start-chunk", type=str, default="0")
    parser.add_argument("--end-chunk", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Translate 10 rows on CPU, print results")
    args = parser.parse_args()

    if args.smoke_test:
        from translator import Translator
        from datasets import load_dataset

        print("Smoke test: 10 rows, CPU...", flush=True)
        ds = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)
        rows = list(islice(ds, 10))
        translator = Translator(device="cpu", device_index=0)
        results = process_rows(rows, translator)
        for r in results:
            print(f"\n--- {r['original_id']} (conf={r['confidence_mean']:.3f}/{r['confidence_min']:.3f}, "
                  f"translated={r['sentences_translated']}/{r['sentences_total']}) ---")
            print(f"EN: {r['text_en'][:200]}...")
            print(f"KK: {r['text_kk'][:200]}...")
        return

    from huggingface_hub import HfApi
    from datasets import load_dataset

    api = HfApi()
    t_total = time.time()

    if args.start_chunk == "auto":
        start_chunk = find_first_missing_chunk(api)
        print(f"Auto-detected start chunk: {start_chunk}", flush=True)
    else:
        start_chunk = int(args.start_chunk)

    end_chunk = args.end_chunk if args.end_chunk is not None else start_chunk + 10

    print(f"Pipeline v2: chunks {start_chunk}–{end_chunk - 1}", flush=True)
    print(f"Repo: {HF_REPO}", flush=True)
    print(f"Source: {SOURCE_DATASET} ({SOURCE_CONFIG})", flush=True)
    print(f"GPUs: {args.num_gpus}", flush=True)

    chunks_todo = []
    for chunk_idx in range(start_chunk, end_chunk):
        if shard_exists(api, chunk_idx):
            print(f"Chunk {chunk_idx}: exists, skipping.", flush=True)
        else:
            chunks_todo.append(chunk_idx)

    if not chunks_todo:
        print("All chunks already uploaded!", flush=True)
        return

    first_offset = chunks_todo[0] * ROWS_PER_CHUNK
    last_end = (chunks_todo[-1] + 1) * ROWS_PER_CHUNK

    print(f"\nStreaming rows {first_offset}–{last_end}...", flush=True)
    ds_stream = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)
    stream_iter = iter(islice(ds_stream, first_offset, last_end))

    for chunk_idx in range(chunks_todo[0], chunks_todo[-1] + 1):
        rows = list(islice(stream_iter, ROWS_PER_CHUNK))
        if chunk_idx not in chunks_todo:
            continue
        process_chunk(chunk_idx, rows, args, api)

    elapsed = time.time() - t_total
    print(f"\nALL DONE in {elapsed / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
