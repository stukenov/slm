#!/usr/bin/env python3
"""
FineWeb-Edu sample-100BT EN→KK translation pipeline.

Robust checkpointing at every stage:
  1. Intra-chunk: saves translated rows every CHECKPOINT_EVERY rows to disk
  2. Chunk-level: uploads completed chunks to HF Hub as parquet shards
  3. Upload verification: downloads shard header to verify row count
  4. Progress tracking: progress_100bt.json with stats, speed, ETA

Usage:
    # Smoke test (10 rows, CPU)
    python pipeline_100bt.py --smoke-test

    # 4 GPUs, auto-resume from last incomplete chunk
    python pipeline_100bt.py --num-gpus 4 --start-chunk auto

    # 8 GPUs, specific range
    python pipeline_100bt.py --num-gpus 8 --start-chunk 0 --end-chunk 100

    # Multi-node: node 1 handles chunks 0-49, node 2 handles 50-99
    python pipeline_100bt.py --num-gpus 4 --start-chunk 0 --end-chunk 50
"""

import argparse
import glob as globmod
import json
import os
import time
from itertools import islice

import xxhash

from config_100bt import (
    SOURCE_DATASET,
    SOURCE_CONFIG,
    ROWS_PER_CHUNK,
    HF_REPO,
    CHECKPOINT_EVERY,
    UPLOAD_VERIFY_ROWS,
    PROGRESS_FILE,
)
from sentence_splitter import split_document
from postprocessor import process_document

BASE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE, "checkpoints_100bt")


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    path = os.path.join(BASE, PROGRESS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "chunks_completed": [],
        "chunks_verified": [],
        "total_rows_translated": 0,
        "total_sentences_translated": 0,
        "total_elapsed_sec": 0,
        "avg_sents_per_sec": 0,
        "last_updated": None,
    }


def save_progress(prog: dict):
    prog["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(BASE, PROGRESS_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prog, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Intra-chunk checkpoint (row-level resume)
# ---------------------------------------------------------------------------

def chunk_checkpoint_dir(chunk_idx: int) -> str:
    d = os.path.join(CHECKPOINT_DIR, f"chunk_{chunk_idx:05d}")
    os.makedirs(d, exist_ok=True)
    return d


def save_intra_checkpoint(chunk_idx: int, rows_done: int, output_rows: list):
    """Save translated rows to disk for intra-chunk resume."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = chunk_checkpoint_dir(chunk_idx)
    path = os.path.join(d, f"rows_{rows_done:07d}.parquet")
    tmp = path + ".tmp"
    table = pa.Table.from_pylist(output_rows)
    pq.write_table(table, tmp)
    os.replace(tmp, path)

    # Write marker
    marker = os.path.join(d, "last_checkpoint.json")
    with open(marker, "w") as f:
        json.dump({"rows_done": rows_done, "file": path, "count": len(output_rows)}, f)

    print(f"  [CHECKPOINT] chunk={chunk_idx} rows_done={rows_done} saved={len(output_rows)} rows", flush=True)


def load_intra_checkpoint(chunk_idx: int) -> tuple[int, list]:
    """Load last intra-chunk checkpoint. Returns (rows_done, output_rows) or (0, [])."""
    import pyarrow.parquet as pq

    d = chunk_checkpoint_dir(chunk_idx)
    marker = os.path.join(d, "last_checkpoint.json")
    if not os.path.exists(marker):
        return 0, []

    with open(marker) as f:
        info = json.load(f)

    path = info["file"]
    if not os.path.exists(path):
        print(f"  [WARN] Checkpoint marker exists but file missing: {path}", flush=True)
        return 0, []

    table = pq.read_table(path)
    rows = table.to_pylist()
    rows_done = info["rows_done"]
    print(f"  [RESUME] chunk={chunk_idx} resuming from row {rows_done}, loaded {len(rows)} translated rows", flush=True)
    return rows_done, rows


def cleanup_intra_checkpoint(chunk_idx: int):
    """Remove intra-chunk checkpoint files after successful upload."""
    import shutil
    d = chunk_checkpoint_dir(chunk_idx)
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  [CLEANUP] Removed checkpoint dir for chunk {chunk_idx}", flush=True)


# ---------------------------------------------------------------------------
# HF Hub helpers
# ---------------------------------------------------------------------------

def shard_filename(chunk_idx: int) -> str:
    return f"data/sample-100BT-{chunk_idx:05d}.parquet"


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


def verify_uploaded_shard(api, chunk_idx: int, expected_rows: int) -> bool:
    """Verify uploaded shard is valid by checking file info on HF Hub."""
    if not UPLOAD_VERIFY_ROWS:
        return True

    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        # Download just the metadata (parquet footer) to verify
        local = hf_hub_download(
            HF_REPO,
            shard_filename(chunk_idx),
            repo_type="dataset",
            force_download=True,
        )
        meta = pq.read_metadata(local)
        actual_rows = meta.num_rows

        if actual_rows == expected_rows:
            print(f"  [VERIFY OK] chunk={chunk_idx}: {actual_rows} rows confirmed", flush=True)
            return True
        else:
            print(f"  [VERIFY FAIL] chunk={chunk_idx}: expected {expected_rows}, got {actual_rows}", flush=True)
            return False
    except Exception as e:
        print(f"  [VERIFY ERROR] chunk={chunk_idx}: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Translation logic
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return xxhash.xxh64_hexdigest(text)


def process_rows(rows, translator, verbose=True):
    """Process a batch of rows: split → translate → postprocess."""
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
        print(f"  Split {len(rows)} docs → {len(sentence_texts)} sentences to translate", flush=True)

    if sentence_texts:
        translation_results = translator.translate_sentences(sentence_texts, verbose=verbose)
    else:
        translation_results = []

    doc_translations = {}
    for (global_idx, doc_idx, sent_idx), tr_result in zip(sentence_map, translation_results):
        doc_translations.setdefault(doc_idx, {})[sent_idx] = tr_result

    output_rows = []
    total_sents_translated = 0
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
        total_sents_translated += result["sentences_translated"]

    return output_rows, total_sents_translated


def process_chunk(chunk_idx: int, rows: list, num_gpus: int, api, progress: dict):
    """Process one chunk with intra-chunk checkpointing and multi-GPU."""
    from translator import Translator
    from datasets import Dataset
    import pyarrow as pa
    import pyarrow.parquet as pq

    total = len(rows)
    print(f"\n{'='*60}", flush=True)
    print(f"CHUNK {chunk_idx}: {total} rows, {num_gpus} GPU(s)", flush=True)
    print(f"{'='*60}", flush=True)

    # Check for intra-chunk resume
    resume_from, existing_output = load_intra_checkpoint(chunk_idx)

    if resume_from >= total:
        print(f"  Chunk {chunk_idx} already fully translated in checkpoint, uploading...", flush=True)
        all_output = existing_output
    else:
        # Load translator(s)
        # For multi-GPU: we split remaining rows and use multiple translators
        # For simplicity and reliability, we process sequentially per GPU
        # (multi-process within chunk was unreliable for checkpointing)
        if num_gpus > 1:
            # Round-robin GPU allocation for batches
            translators = []
            for gpu_id in range(num_gpus):
                print(f"  Loading translator on GPU:{gpu_id}...", flush=True)
                translators.append(Translator(device="cuda", device_index=gpu_id))
        else:
            translators = [Translator(device="cuda", device_index=0)]

        all_output = list(existing_output)
        t_chunk_start = time.time()
        total_sents_this_chunk = 0

        for batch_start in range(resume_from, total, CHECKPOINT_EVERY):
            batch_end = min(batch_start + CHECKPOINT_EVERY, total)
            batch_rows = rows[batch_start:batch_end]

            # Pick translator (round-robin across GPUs for each batch)
            gpu_idx = (batch_start // CHECKPOINT_EVERY) % len(translators)
            translator = translators[gpu_idx]

            print(f"\n  [chunk={chunk_idx}] Translating rows {batch_start}–{batch_end} on GPU:{gpu_idx}...", flush=True)
            output, sents_translated = process_rows(batch_rows, translator)
            all_output.extend(output)
            total_sents_this_chunk += sents_translated

            # Save intra-chunk checkpoint
            save_intra_checkpoint(chunk_idx, batch_end, all_output)

            elapsed = time.time() - t_chunk_start
            rows_done = batch_end - resume_from
            rows_left = total - batch_end
            speed = rows_done / elapsed if elapsed > 0 else 0
            eta = rows_left / speed if speed > 0 else 0
            print(f"  [chunk={chunk_idx}] {batch_end}/{total} rows | "
                  f"{speed:.0f} rows/sec | ETA {eta/60:.1f}min", flush=True)

        chunk_elapsed = time.time() - t_chunk_start
        print(f"\n  Chunk {chunk_idx} translated: {len(all_output)} rows, "
              f"{total_sents_this_chunk} sentences in {chunk_elapsed/60:.1f}min", flush=True)

    # Save final parquet locally
    local_path = os.path.join(BASE, f"chunk_{chunk_idx:05d}.parquet")
    table = pa.Table.from_pylist(all_output)
    pq.write_table(table, local_path)
    print(f"  Saved local: {local_path} ({len(all_output)} rows)", flush=True)

    # Upload to HF Hub
    remote_path = shard_filename(chunk_idx)
    print(f"  Uploading → {HF_REPO}/{remote_path}...", flush=True)

    max_upload_retries = 3
    for attempt in range(1, max_upload_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            print(f"  Upload complete (attempt {attempt}).", flush=True)
            break
        except Exception as e:
            print(f"  [UPLOAD ERROR] attempt {attempt}/{max_upload_retries}: {e}", flush=True)
            if attempt < max_upload_retries:
                time.sleep(30 * attempt)
            else:
                print(f"  [FATAL] Failed to upload chunk {chunk_idx} after {max_upload_retries} attempts!", flush=True)
                print(f"  Local file preserved at: {local_path}", flush=True)
                return False

    # Verify upload
    verified = verify_uploaded_shard(api, chunk_idx, len(all_output))
    if not verified:
        print(f"  [WARN] Verification failed for chunk {chunk_idx}. Local file preserved: {local_path}", flush=True)
        return False

    # Success — cleanup
    os.remove(local_path)
    cleanup_intra_checkpoint(chunk_idx)

    # Update progress
    progress["chunks_completed"].append(chunk_idx)
    progress["chunks_verified"].append(chunk_idx)
    progress["total_rows_translated"] += len(all_output)
    save_progress(progress)

    print(f"  Chunk {chunk_idx} DONE and verified.", flush=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu 100BT EN→KK pipeline")
    parser.add_argument("--start-chunk", type=str, default="0")
    parser.add_argument("--end-chunk", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if args.smoke_test:
        from translator import Translator
        from datasets import load_dataset

        print("Smoke test: 10 rows, CPU...", flush=True)
        ds = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)
        rows = list(islice(ds, 10))
        translator = Translator(device="cpu", device_index=0)
        results, sents = process_rows(rows, translator)

        print(f"\n{'='*60}")
        print(f"SMOKE TEST RESULTS: {len(results)} docs, {sents} sentences translated")
        print(f"{'='*60}")
        for r in results:
            print(f"\n--- {r['original_id']} (conf={r['confidence_mean']:.3f}/{r['confidence_min']:.3f}, "
                  f"translated={r['sentences_translated']}/{r['sentences_total']}) ---")
            print(f"EN: {r['text_en'][:200]}...")
            print(f"KK: {r['text_kk'][:200]}...")

        # Verify checkpoint save/load cycle
        print(f"\n{'='*60}")
        print("CHECKPOINT VERIFICATION")
        print(f"{'='*60}")
        save_intra_checkpoint(99999, len(results), results)
        loaded_rows_done, loaded_output = load_intra_checkpoint(99999)
        assert loaded_rows_done == len(results), f"rows_done mismatch: {loaded_rows_done} != {len(results)}"
        assert len(loaded_output) == len(results), f"output count mismatch: {len(loaded_output)} != {len(results)}"
        for i, (orig, loaded) in enumerate(zip(results, loaded_output)):
            assert orig["original_id"] == loaded["original_id"], f"Row {i} id mismatch"
            assert orig["text_kk"] == loaded["text_kk"], f"Row {i} text_kk mismatch"
            assert orig["confidence_mean"] == loaded["confidence_mean"], f"Row {i} confidence mismatch"
        cleanup_intra_checkpoint(99999)
        print("  [OK] Checkpoint save → load → verify → cleanup: ALL PASSED")

        # Verify progress tracking
        prog = load_progress()
        prog["total_rows_translated"] += len(results)
        save_progress(prog)
        prog2 = load_progress()
        assert prog2["total_rows_translated"] == prog["total_rows_translated"], "Progress save/load mismatch"
        # Reset
        os.remove(os.path.join(BASE, PROGRESS_FILE))
        print("  [OK] Progress save → load → verify: PASSED")

        print("\n  ALL SMOKE TESTS PASSED. Ready for production run.")
        return

    from huggingface_hub import HfApi
    from datasets import load_dataset

    api = HfApi()
    progress = load_progress()
    t_total = time.time()

    # Ensure HF repo exists
    try:
        api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"[WARN] Could not create/check repo: {e}", flush=True)

    if args.start_chunk == "auto":
        start_chunk = find_first_missing_chunk(api)
        print(f"Auto-detected start chunk: {start_chunk}", flush=True)
    else:
        start_chunk = int(args.start_chunk)

    # sample-100BT: ~96M rows → ~96 chunks of 1M
    end_chunk = args.end_chunk if args.end_chunk is not None else 100

    print(f"Pipeline 100BT: chunks {start_chunk}–{end_chunk - 1}", flush=True)
    print(f"Repo: {HF_REPO}", flush=True)
    print(f"Source: {SOURCE_DATASET} ({SOURCE_CONFIG})", flush=True)
    print(f"GPUs: {args.num_gpus}", flush=True)
    print(f"Checkpoint every: {CHECKPOINT_EVERY} rows", flush=True)
    print(f"Upload verification: {UPLOAD_VERIFY_ROWS}", flush=True)

    # Find which chunks still need processing
    chunks_todo = []
    for chunk_idx in range(start_chunk, end_chunk):
        if shard_exists(api, chunk_idx):
            print(f"  Chunk {chunk_idx}: exists on HF, skipping.", flush=True)
        else:
            chunks_todo.append(chunk_idx)

    if not chunks_todo:
        print("All chunks already uploaded!", flush=True)
        return

    print(f"\n{len(chunks_todo)} chunks to process: {chunks_todo[0]}–{chunks_todo[-1]}", flush=True)

    # Stream dataset
    first_offset = chunks_todo[0] * ROWS_PER_CHUNK
    last_end = (chunks_todo[-1] + 1) * ROWS_PER_CHUNK

    print(f"Streaming rows {first_offset}–{last_end} from {SOURCE_CONFIG}...", flush=True)
    ds_stream = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)
    stream_iter = iter(islice(ds_stream, first_offset, last_end))

    failed_chunks = []
    for chunk_idx in range(chunks_todo[0], chunks_todo[-1] + 1):
        rows = list(islice(stream_iter, ROWS_PER_CHUNK))
        if not rows:
            print(f"  No more rows at chunk {chunk_idx}, dataset exhausted.", flush=True)
            break
        if chunk_idx not in chunks_todo:
            continue

        success = process_chunk(chunk_idx, rows, args.num_gpus, api, progress)
        if not success:
            failed_chunks.append(chunk_idx)

        # Update overall progress
        elapsed = time.time() - t_total
        chunks_done = len(progress["chunks_completed"])
        chunks_left = len(chunks_todo) - chunks_done
        if chunks_done > 0:
            eta_h = (elapsed / chunks_done) * chunks_left / 3600
        else:
            eta_h = 0
        print(f"\n  OVERALL: {chunks_done}/{len(chunks_todo)} chunks | "
              f"{elapsed/3600:.1f}h elapsed | ETA {eta_h:.1f}h remaining", flush=True)

    total_elapsed = time.time() - t_total
    progress["total_elapsed_sec"] += total_elapsed
    save_progress(progress)

    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE in {total_elapsed / 3600:.1f}h", flush=True)
    print(f"Chunks completed: {len(progress['chunks_completed'])}", flush=True)
    print(f"Chunks verified: {len(progress['chunks_verified'])}", flush=True)
    print(f"Total rows: {progress['total_rows_translated']}", flush=True)
    if failed_chunks:
        print(f"FAILED chunks (retry manually): {failed_chunks}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
