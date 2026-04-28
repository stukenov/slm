#!/usr/bin/env python3
"""
FineWeb-Edu EN-to-KK translation pipeline v3 -- TPU-optimized.

Resume: checks HF Hub for completed chunks, skips them.
Loss on preemption: at most 1 chunk (1M docs).

Usage:
    # Smoke test (10 rows, CPU)
    python pipeline.py --smoke-test

    # Single TPU chip
    python pipeline.py --start-chunk 0 --end-chunk 2

    # Auto-resume (finds first missing chunk on HF)
    python pipeline.py --start-chunk auto

    # CPU/GPU fallback
    python pipeline.py --device cpu --start-chunk 0 --end-chunk 1
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from itertools import islice

import xxhash

from config import (
    SOURCE_DATASET,
    SOURCE_CONFIG,
    ROWS_PER_CHUNK,
    HF_REPO,
    PROGRESS_FILE,
    LOG_EVERY_ROWS,
)
from sentence_splitter import split_document
from postprocessor import process_document

BASE = os.path.dirname(os.path.abspath(__file__))


def content_hash(text: str) -> str:
    return xxhash.xxh64_hexdigest(text)


def shard_filename(chunk_idx: int, source_config: str) -> str:
    return f"data/{source_config}-{chunk_idx:05d}.parquet"


def get_completed_chunks(api, source_config: str) -> set[int]:
    """List completed chunk indices from HF repo."""
    try:
        files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    except Exception:
        return set()

    completed = set()
    prefix = f"data/{source_config}-"
    for f in files:
        if f.startswith(prefix) and f.endswith(".parquet"):
            try:
                idx = int(f[len(prefix):-len(".parquet")])
                completed.add(idx)
            except ValueError:
                pass
    return completed


def find_first_missing_chunk(api, source_config: str) -> int:
    completed = get_completed_chunks(api, source_config)
    idx = 0
    while idx in completed:
        idx += 1
    return idx


def process_rows(rows, translator, verbose=True):
    """Split -> translate -> postprocess a batch of rows."""
    docs = []
    sentence_texts = []
    sentence_map = []

    for doc_idx, row in enumerate(rows):
        text = row.get("text", "")
        doc = split_document(text)
        docs.append((row, doc))

        for sent in doc["sentences"]:
            if not sent["skipped"]:
                sentence_map.append((len(sentence_texts), doc_idx, sent["sent_idx"]))
                sentence_texts.append(sent["text"])

    if verbose:
        print(f"  Split {len(rows)} docs -> {len(sentence_texts)} sentences", flush=True)

    if sentence_texts:
        translation_results = translator.translate_sentences(sentence_texts, verbose=verbose)
    else:
        translation_results = []

    doc_translations = {}
    for (global_idx, doc_idx, sent_idx), tr_result in zip(sentence_map, translation_results):
        doc_translations.setdefault(doc_idx, {})[sent_idx] = tr_result

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


def upload_progress(api, completed_chunks, stats):
    """Upload progress.json to HF repo."""
    progress = {
        "pipeline_version": "v3-tpu",
        "source": SOURCE_DATASET,
        "source_config": SOURCE_CONFIG,
        "completed_chunks": sorted(completed_chunks),
        "stats": stats,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(BASE, PROGRESS_FILE)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)
    try:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=PROGRESS_FILE,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  Warning: failed to upload progress.json: {e}", flush=True)


def process_chunk(chunk_idx, rows, translator, api, source_config, stats):
    """Translate one chunk and upload to HF."""
    from datasets import Dataset

    print(f"\n{'=' * 60}", flush=True)
    print(f"CHUNK {chunk_idx}: {len(rows)} rows", flush=True)
    print(f"{'=' * 60}", flush=True)

    t_start = time.time()
    all_output = []

    # Process in sub-batches for progress logging
    sub_batch = LOG_EVERY_ROWS
    for offset in range(0, len(rows), sub_batch):
        batch_rows = rows[offset:offset + sub_batch]
        output = process_rows(batch_rows, translator)
        all_output.extend(output)

        elapsed = time.time() - t_start
        total_sents = sum(r["sentences_total"] for r in all_output)
        translated_sents = sum(r["sentences_translated"] for r in all_output)
        sps = translated_sents / elapsed if elapsed > 0 else 0
        print(f"  Progress: {len(all_output)}/{len(rows)} docs, "
              f"{translated_sents} sents translated, {sps:.0f} sents/sec", flush=True)

    # Save and upload
    chunk_path = os.path.join(BASE, f"chunk_{chunk_idx:05d}.parquet")
    Dataset.from_list(all_output).to_parquet(chunk_path)

    remote_path = shard_filename(chunk_idx, source_config)
    print(f"Uploading {remote_path} to {HF_REPO}...", flush=True)
    api.upload_file(
        path_or_fileobj=chunk_path,
        path_in_repo=remote_path,
        repo_id=HF_REPO,
        repo_type="dataset",
    )

    elapsed = time.time() - t_start
    total_sents = sum(r["sentences_total"] for r in all_output)
    translated_sents = sum(r["sentences_translated"] for r in all_output)

    print(f"Chunk {chunk_idx} done: {len(all_output)} docs, "
          f"{translated_sents}/{total_sents} sents, {elapsed:.0f}s", flush=True)
    os.remove(chunk_path)

    # Update stats
    stats["total_docs"] += len(all_output)
    stats["total_sentences_translated"] += translated_sents
    stats["total_sentences_total"] += total_sents
    if all_output:
        confs = [r["confidence_mean"] for r in all_output if r["confidence_mean"] > 0]
        if confs:
            stats["avg_confidence"] = sum(confs) / len(confs)

    return len(all_output)


def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu EN-to-KK pipeline v3 (TPU)")
    parser.add_argument("--start-chunk", type=str, default="0")
    parser.add_argument("--end-chunk", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu, cuda, cuda:0, etc. Default: auto-detect TPU")
    parser.add_argument("--source-config", type=str, default=SOURCE_CONFIG)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="Translate 200 rows and report speed")
    args = parser.parse_args()

    # Determine device
    device = None
    if args.device:
        import torch
        if args.device == "tpu":
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        else:
            device = torch.device(args.device)

    if args.smoke_test:
        from datasets import load_dataset
        print("Smoke test: 10 rows...", flush=True)
        ds = load_dataset(SOURCE_DATASET, args.source_config, split="train", streaming=True)
        rows = list(islice(ds, 10))
        translator = Translator(device=device)
        results = process_rows(rows, translator)
        for r in results:
            print(f"\n--- {r['original_id']} (conf={r['confidence_mean']:.3f}, "
                  f"translated={r['sentences_translated']}/{r['sentences_total']}) ---")
            print(f"EN: {r['text_en'][:200]}...")
            print(f"KK: {r['text_kk'][:200]}...")
        return

    if args.benchmark:
        from datasets import load_dataset
        print("Benchmark: 200 rows...", flush=True)
        ds = load_dataset(SOURCE_DATASET, args.source_config, split="train", streaming=True)
        rows = list(islice(ds, 200))
        translator = Translator(device=device)
        if args.batch_size:
            from config import BATCH_SIZE
            # Override via translator
            translator.translate_sentences.__defaults__ = (args.batch_size, True)

        t0 = time.time()
        results = process_rows(rows, translator)
        elapsed = time.time() - t0

        total_sents = sum(r["sentences_total"] for r in results)
        translated_sents = sum(r["sentences_translated"] for r in results)
        sps = translated_sents / elapsed

        print(f"\n{'=' * 40}")
        print(f"Benchmark results:")
        print(f"  Docs: {len(results)}")
        print(f"  Sentences: {total_sents} total, {translated_sents} translated")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Speed: {sps:.0f} sents/sec")
        print(f"  Avg confidence: {sum(r['confidence_mean'] for r in results) / len(results):.3f}")

        # Extrapolate
        est_10bt_sents = 319_000_000
        est_100bt_sents = 3_190_000_000
        print(f"\n  Estimated sample-10BT:  {est_10bt_sents / sps / 3600:.1f} hours")
        print(f"  Estimated sample-100BT: {est_100bt_sents / sps / 3600:.1f} hours")
        return

    # Full pipeline
    from translator import Translator
    from huggingface_hub import HfApi
    from datasets import load_dataset

    api = HfApi()
    t_total = time.time()

    # Ensure repo exists
    try:
        api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    except Exception:
        pass

    completed = get_completed_chunks(api, args.source_config)
    print(f"Completed chunks on HF: {sorted(completed) if completed else 'none'}", flush=True)

    if args.start_chunk == "auto":
        start_chunk = find_first_missing_chunk(api, args.source_config)
        print(f"Auto-resume: starting from chunk {start_chunk}", flush=True)
    else:
        start_chunk = int(args.start_chunk)

    end_chunk = args.end_chunk if args.end_chunk is not None else start_chunk + 10

    print(f"Pipeline v3 (TPU): chunks {start_chunk}-{end_chunk - 1}", flush=True)
    print(f"Source: {SOURCE_DATASET} ({args.source_config})", flush=True)
    print(f"Output: {HF_REPO}", flush=True)

    # Load translator
    translator = Translator(device=device)

    # Find chunks to process
    chunks_todo = [i for i in range(start_chunk, end_chunk) if i not in completed]
    if not chunks_todo:
        print("All chunks already uploaded!", flush=True)
        return

    print(f"Chunks to process: {chunks_todo}", flush=True)

    stats = {
        "total_docs": 0,
        "total_sentences_translated": 0,
        "total_sentences_total": 0,
        "avg_confidence": 0.0,
    }

    # Stream dataset
    first_offset = chunks_todo[0] * ROWS_PER_CHUNK
    last_end = (chunks_todo[-1] + 1) * ROWS_PER_CHUNK

    print(f"Streaming rows {first_offset}-{last_end}...", flush=True)
    ds_stream = load_dataset(SOURCE_DATASET, args.source_config, split="train", streaming=True)
    stream_iter = iter(islice(ds_stream, first_offset, last_end))

    for chunk_idx in range(chunks_todo[0], chunks_todo[-1] + 1):
        rows = list(islice(stream_iter, ROWS_PER_CHUNK))
        if not rows:
            print(f"No more rows at chunk {chunk_idx}", flush=True)
            break
        if chunk_idx not in chunks_todo:
            continue

        process_chunk(chunk_idx, rows, translator, api, args.source_config, stats)
        completed.add(chunk_idx)

        stats["elapsed_hours"] = (time.time() - t_total) / 3600
        if stats["total_sentences_translated"] > 0 and stats["elapsed_hours"] > 0:
            stats["sents_per_sec"] = stats["total_sentences_translated"] / (stats["elapsed_hours"] * 3600)
        upload_progress(api, completed, stats)

    elapsed = time.time() - t_total
    print(f"\nDONE in {elapsed / 3600:.1f}h", flush=True)
    print(f"Stats: {json.dumps(stats, indent=2)}", flush=True)


if __name__ == "__main__":
    # Import here to allow --smoke-test and --benchmark without full deps
    from translator import Translator
    main()
