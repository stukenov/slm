#!/usr/bin/env python3
"""
FineWeb-Edu sample-10BT EN→KK translation pipeline.

Produces two HF datasets:
  1. Documents: full translated texts with preserved formatting
  2. Sentence pairs: per-sentence EN↔KK pairs for multilingual reuse

Usage:
    python pipeline_10bt.py --num-gpus 2 --filter --start-chunk 0 --end-chunk 10
    python pipeline_10bt.py --num-gpus 2 --filter --start-chunk auto
"""

import argparse
import glob as globmod
import os
import re
import time
from itertools import islice
from multiprocessing import Process, Queue

import ctranslate2
import sentencepiece as spm
from datasets import Dataset
from huggingface_hub import HfApi

from filters import TextFilter

BASE = os.path.dirname(os.path.abspath(__file__))
CT2_DIR = os.path.join(BASE, "model_ct2")
SPM_PATH = os.path.join(BASE, "model_cache", "model.en-kk.spm")

HF_REPO_DOCS = "stukenov/sozkz-fineweb-edu-10bt-en-kk"
HF_REPO_SENTS = "stukenov/sozkz-fineweb-edu-10bt-parallel-en-kk"
ROWS_PER_CHUNK = 1_000_000
SOURCE_DATASET = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample-10BT"

SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ\d"])')


# ---------------------------------------------------------------------------
# Sentence splitting with paragraph preservation
# ---------------------------------------------------------------------------

def split_sentences_preserve_format(text):
    """Split text into sentences, preserving paragraph structure.

    Returns list of (sentence, paragraph_idx, is_empty_paragraph).
    Empty paragraphs are markers for blank lines in the original text.
    """
    paragraphs = text.split('\n')
    result = []
    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            result.append(("", para_idx, True))
            continue
        sents = SENT_RE.split(para)
        for sent in sents:
            sent = sent.strip()
            if sent:
                result.append((sent, para_idx, False))
    return result


def reassemble_text(sents_info, translated_sents):
    """Reassemble translated sentences preserving original paragraph breaks."""
    paragraphs = {}
    trans_idx = 0
    for sent_text, para_idx, is_empty in sents_info:
        if is_empty:
            paragraphs.setdefault(para_idx, [])
            continue
        paragraphs.setdefault(para_idx, []).append(translated_sents[trans_idx])
        trans_idx += 1
    result = []
    for para_idx in sorted(paragraphs.keys()):
        sents = paragraphs[para_idx]
        result.append(" ".join(sents))
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Translation (reused from pipeline.py logic)
# ---------------------------------------------------------------------------

def translate_batch_sentences(
    translator, sp, sentences,
    batch_size=4096, beam_size=1,
    max_input_length=128, max_decoding_length=200,
):
    if not sentences:
        return []

    all_tokens = []
    for s in sentences:
        toks = sp.encode(s, out_type=str)
        if len(toks) > max_input_length:
            toks = toks[:max_input_length]
        all_tokens.append(toks)

    translated = [""] * len(sentences)
    total_batches = (len(all_tokens) + batch_size - 1) // batch_size
    t0 = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_tokens))

        batch_indices = list(range(start, end))
        batch_indices.sort(key=lambda i: len(all_tokens[i]))
        batch_tokens = [all_tokens[i] for i in batch_indices]

        results = translator.translate_batch(
            batch_tokens,
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
        )

        for local_idx, global_idx in enumerate(batch_indices):
            translated[global_idx] = sp.decode(results[local_idx].hypotheses[0])

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - t0
            done = end
            sps = done / elapsed if elapsed > 0 else 0
            eta = (len(all_tokens) - done) / sps if sps > 0 else 0
            print(f"  [{done}/{len(all_tokens)}] {sps:.0f} sents/sec, ETA {eta:.0f}s", flush=True)

    return translated


# ---------------------------------------------------------------------------
# Worker: translate rows, produce doc + sent outputs
# ---------------------------------------------------------------------------

def find_completed_chunks(output_path, gpu_id):
    pattern = output_path.replace(".parquet", f"_gpu{gpu_id}_ckpt_*.parquet")
    completed = {}
    for path in globmod.glob(pattern):
        match = re.search(r'_ckpt_(\d+)\.parquet$', path)
        if match:
            abs_end = int(match.group(1))
            completed[abs_end] = path
    final = output_path.replace(".parquet", f"_gpu{gpu_id}.parquet")
    if os.path.exists(final):
        completed[-1] = final
    return completed


def worker_translate(
    gpu_id, rows, row_offset, output_path,
    batch_size, beam_size, max_input_length, max_decoding_length,
    compute_type, checkpoint_every, resume, result_queue,
):
    print(f"[GPU:{gpu_id}] Loading model (compute_type={compute_type})...", flush=True)
    translator = ctranslate2.Translator(
        CT2_DIR, device="cuda", device_index=gpu_id, compute_type=compute_type,
    )
    sp = spm.SentencePieceProcessor(SPM_PATH)

    chunk_size = checkpoint_every
    all_doc_rows = []
    all_sent_rows = []
    total_sents = 0
    t_start = time.time()
    skipped_chunks = 0

    # Checkpoint paths use _doc and _sent suffixes
    doc_output = output_path.replace(".parquet", "_doc.parquet")
    sent_output = output_path.replace(".parquet", "_sent.parquet")

    completed = {}
    if resume:
        completed = find_completed_chunks(output_path, gpu_id)
        if -1 in completed:
            print(f"[GPU:{gpu_id}] Already fully completed!", flush=True)
            doc_path = output_path.replace(".parquet", f"_gpu{gpu_id}_doc.parquet")
            sent_path = output_path.replace(".parquet", f"_gpu{gpu_id}_sent.parquet")
            doc_ds = Dataset.from_parquet(doc_path)
            result_queue.put((gpu_id, doc_path, sent_path, len(doc_ds)))
            return
        if completed:
            print(f"[GPU:{gpu_id}] Found {len(completed)} completed checkpoints", flush=True)

    for chunk_start in range(0, len(rows), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(rows))
        abs_start = row_offset + chunk_start
        abs_end = row_offset + chunk_end

        if resume and abs_end in completed:
            print(f"[GPU:{gpu_id}] Skipping chunk {abs_start}–{abs_end} (checkpoint exists)", flush=True)
            # Load doc checkpoint
            ckpt_doc = completed[abs_end]
            ckpt_sent = ckpt_doc.replace("_ckpt_", "_sent_ckpt_")
            doc_ds = Dataset.from_parquet(ckpt_doc)
            all_doc_rows.extend(doc_ds.to_list())
            if os.path.exists(ckpt_sent):
                sent_ds = Dataset.from_parquet(ckpt_sent)
                all_sent_rows.extend(sent_ds.to_list())
            skipped_chunks += 1
            continue

        chunk_rows = rows[chunk_start:chunk_end]
        print(f"\n[GPU:{gpu_id}] Chunk {abs_start}–{abs_end} ({len(chunk_rows)} rows)", flush=True)

        # Split all sentences with format preservation
        all_sentences = []
        doc_sent_info = []  # per-doc: list of (sentence, para_idx, is_empty)
        for row in chunk_rows:
            sents_info = split_sentences_preserve_format(row["text"])
            # Extract only non-empty sentences for translation
            real_sents = [(s, pi, ie) for s, pi, ie in sents_info if not ie]
            start_idx = len(all_sentences)
            all_sentences.extend([s for s, _, _ in real_sents])
            doc_sent_info.append((sents_info, start_idx, len(all_sentences)))

        chunk_sents = len(all_sentences)
        total_sents += chunk_sents
        print(f"[GPU:{gpu_id}] Sentences: {chunk_sents}", flush=True)

        t0 = time.time()
        translated_sents = translate_batch_sentences(
            translator, sp, all_sentences,
            batch_size=batch_size, beam_size=beam_size,
            max_input_length=max_input_length,
            max_decoding_length=max_decoding_length,
        )
        elapsed = time.time() - t0
        print(f"[GPU:{gpu_id}] Chunk done in {elapsed:.1f}s ({chunk_sents / elapsed:.0f} sents/sec)", flush=True)

        # Build doc and sent rows
        chunk_doc_rows = []
        chunk_sent_rows = []
        for i, row in enumerate(chunk_rows):
            sents_info, sent_start, sent_end = doc_sent_info[i]
            doc_translated = translated_sents[sent_start:sent_end]
            doc_id = row.get("id", str(abs_start + i))

            # Reassemble translated text with paragraph breaks
            text_kk = reassemble_text(sents_info, doc_translated)

            # Count real sentences
            real_sents_en = [s for s, _, ie in sents_info if not ie]
            num_sents = len(real_sents_en)

            chunk_doc_rows.append({
                "text_en": row["text"],
                "text_kk": text_kk,
                "id": doc_id,
                "num_sentences": num_sents,
            })

            # Sentence pairs
            for sent_idx, (sent_en, sent_kk) in enumerate(zip(real_sents_en, doc_translated)):
                chunk_sent_rows.append({
                    "sentence_en": sent_en,
                    "sentence_kk": sent_kk,
                    "doc_id": doc_id,
                    "sentence_idx": sent_idx,
                })

        all_doc_rows.extend(chunk_doc_rows)
        all_sent_rows.extend(chunk_sent_rows)

        # Checkpoint both formats
        if chunk_end < len(rows):
            ckpt_doc_path = output_path.replace(".parquet", f"_gpu{gpu_id}_ckpt_{abs_end}.parquet")
            ckpt_sent_path = output_path.replace(".parquet", f"_gpu{gpu_id}_sent_ckpt_{abs_end}.parquet")
            Dataset.from_list(all_doc_rows).to_parquet(ckpt_doc_path)
            Dataset.from_list(all_sent_rows).to_parquet(ckpt_sent_path)
            print(f"[GPU:{gpu_id}] Checkpoint: {len(all_doc_rows)} docs, {len(all_sent_rows)} sents", flush=True)

    total_elapsed = time.time() - t_start
    if total_sents > 0:
        print(f"\n[GPU:{gpu_id}] DONE: {len(all_doc_rows)} docs, {len(all_sent_rows)} sents, "
              f"{total_sents} new sents in {total_elapsed:.1f}s "
              f"({total_sents / total_elapsed:.0f} sents/sec, skipped {skipped_chunks} chunks)", flush=True)
    else:
        print(f"\n[GPU:{gpu_id}] DONE: {len(all_doc_rows)} docs (all from checkpoints)", flush=True)

    doc_path = output_path.replace(".parquet", f"_gpu{gpu_id}_doc.parquet")
    sent_path = output_path.replace(".parquet", f"_gpu{gpu_id}_sent.parquet")
    Dataset.from_list(all_doc_rows).to_parquet(doc_path)
    Dataset.from_list(all_sent_rows).to_parquet(sent_path)
    print(f"[GPU:{gpu_id}] Saved: {doc_path}, {sent_path}", flush=True)

    result_queue.put((gpu_id, doc_path, sent_path, len(all_doc_rows)))


# ---------------------------------------------------------------------------
# Shard management
# ---------------------------------------------------------------------------

def shard_filename(chunk_idx):
    return f"data/train-{chunk_idx:05d}.parquet"


def upload_shard(api, local_path, chunk_idx, repo_id):
    remote_path = shard_filename(chunk_idx)
    print(f"Uploading {local_path} → {repo_id}/{remote_path}...", flush=True)
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
    elapsed = time.time() - t0
    files = api.list_repo_files(repo_id, repo_type="dataset")
    if remote_path not in files:
        raise RuntimeError(f"Upload verification failed: {remote_path} not found in {repo_id}")
    print(f"Upload to {repo_id} verified in {elapsed:.1f}s", flush=True)


def shard_exists_both(api, chunk_idx):
    """Check if shard exists in BOTH repos."""
    remote_path = shard_filename(chunk_idx)
    try:
        docs_files = set(api.list_repo_files(HF_REPO_DOCS, repo_type="dataset"))
        sents_files = set(api.list_repo_files(HF_REPO_SENTS, repo_type="dataset"))
        return remote_path in docs_files and remote_path in sents_files
    except Exception:
        return False


def find_first_missing_chunk(api):
    try:
        docs_files = set(api.list_repo_files(HF_REPO_DOCS, repo_type="dataset"))
        sents_files = set(api.list_repo_files(HF_REPO_SENTS, repo_type="dataset"))
    except Exception:
        return 0
    idx = 0
    while shard_filename(idx) in docs_files and shard_filename(idx) in sents_files:
        idx += 1
    return idx


# ---------------------------------------------------------------------------
# Process one chunk
# ---------------------------------------------------------------------------

def process_chunk(chunk_idx, rows, args, api):
    print(f"\n{'='*60}", flush=True)
    print(f"CHUNK {chunk_idx}: {len(rows)} rows loaded", flush=True)
    print(f"{'='*60}", flush=True)

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
        _, doc_path, sent_path, count = result_queue.get()
        doc_output = chunk_output.replace(".parquet", "_doc.parquet")
        sent_output = chunk_output.replace(".parquet", "_sent.parquet")
        os.rename(doc_path, doc_output)
        os.rename(sent_path, sent_output)
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

        # Merge GPU parts for both doc and sent
        all_doc_rows = []
        all_sent_rows = []
        for gpu_id, doc_path, sent_path, count in sorted(results):
            doc_ds = Dataset.from_parquet(doc_path)
            sent_ds = Dataset.from_parquet(sent_path)
            all_doc_rows.extend(doc_ds.to_list())
            all_sent_rows.extend(sent_ds.to_list())
            print(f"  GPU:{gpu_id} → {count} docs, {len(sent_ds)} sents", flush=True)

        doc_output = chunk_output.replace(".parquet", "_doc.parquet")
        sent_output = chunk_output.replace(".parquet", "_sent.parquet")
        Dataset.from_list(all_doc_rows).to_parquet(doc_output)
        Dataset.from_list(all_sent_rows).to_parquet(sent_output)
        print(f"Merged: {len(all_doc_rows)} docs, {len(all_sent_rows)} sents", flush=True)

        # Cleanup GPU part files and checkpoints
        for gpu_id, doc_path, sent_path, _ in results:
            for p in [doc_path, sent_path]:
                if os.path.exists(p):
                    os.remove(p)
        for f in globmod.glob(chunk_output.replace(".parquet", "_gpu*")):
            os.remove(f)

    # Upload to both repos
    upload_shard(api, doc_output, chunk_idx, HF_REPO_DOCS)
    upload_shard(api, sent_output, chunk_idx, HF_REPO_SENTS)

    # Cleanup local files only after both uploads verified
    for p in [doc_output, sent_output]:
        if os.path.exists(p):
            os.remove(p)

    print(f"Chunk {chunk_idx} complete.\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu 10BT EN→KK dual-output pipeline")
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

    if args.start_chunk == "auto":
        start_chunk = find_first_missing_chunk(api)
        print(f"Auto-detected start chunk: {start_chunk}", flush=True)
    else:
        start_chunk = int(args.start_chunk)

    end_chunk = args.end_chunk if args.end_chunk is not None else start_chunk + 10

    print(f"Bulk 10BT pipeline: chunks {start_chunk}–{end_chunk - 1} × {ROWS_PER_CHUNK} rows", flush=True)
    print(f"Docs repo:  {HF_REPO_DOCS}", flush=True)
    print(f"Sents repo: {HF_REPO_SENTS}", flush=True)
    print(f"Source: {SOURCE_DATASET} ({SOURCE_CONFIG})", flush=True)
    print(f"GPUs: {args.num_gpus}, filter: {args.filter}", flush=True)

    chunks_todo = []
    for chunk_idx in range(start_chunk, end_chunk):
        if shard_exists_both(api, chunk_idx):
            print(f"Chunk {chunk_idx}: exists in both repos, skipping.", flush=True)
        else:
            chunks_todo.append(chunk_idx)

    if not chunks_todo:
        print("All chunks already uploaded!", flush=True)
    else:
        first_offset = chunks_todo[0] * ROWS_PER_CHUNK
        last_end = (chunks_todo[-1] + 1) * ROWS_PER_CHUNK
        total_to_stream = last_end - first_offset

        print(f"\nStreaming {SOURCE_DATASET}/{SOURCE_CONFIG} rows {first_offset}–{last_end} "
              f"({total_to_stream} rows)...", flush=True)
        from datasets import load_dataset
        ds_stream = load_dataset(
            SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True,
        )
        stream_iter = iter(islice(ds_stream, first_offset, last_end))

        for chunk_idx in range(chunks_todo[0], chunks_todo[-1] + 1):
            t_load = time.time()
            rows = list(islice(stream_iter, ROWS_PER_CHUNK))
            print(f"\nChunk {chunk_idx}: loaded {len(rows)} rows in {time.time() - t_load:.1f}s", flush=True)

            if chunk_idx not in chunks_todo:
                print(f"Chunk {chunk_idx}: already in both repos, skipping.", flush=True)
                continue

            process_chunk(chunk_idx, rows, args, api)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE in {elapsed / 3600:.1f}h", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
