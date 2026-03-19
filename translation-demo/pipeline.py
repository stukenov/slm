#!/usr/bin/env python3
"""
FineWeb-Edu EN→KK translation pipeline via CTranslate2.

Optimized: dual-GPU, FP16, greedy, length limiting, resume from checkpoints.

Usage:
    python pipeline.py --smoke-test
    python pipeline.py --num-rows 1000000 --num-gpus 2
    python pipeline.py --num-rows 1000000 --num-gpus 2 --resume   # resume from checkpoints
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

BASE = os.path.dirname(os.path.abspath(__file__))
CT2_DIR = os.path.join(BASE, "model_ct2")
SPM_PATH = os.path.join(BASE, "model_cache", "model.en-kk.spm")
DEFAULT_OUTPUT = os.path.join(BASE, "fineweb_edu_kk.parquet")

SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ\d"])')


def split_sentences(text: str) -> list[str]:
    sents = SENT_RE.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


def find_completed_chunks(output_path: str, gpu_id: int) -> dict[int, str]:
    """Find already-completed checkpoint files for a GPU. Returns {abs_end: path}."""
    pattern = output_path.replace(".parquet", f"_gpu{gpu_id}_ckpt_*.parquet")
    completed = {}
    for path in globmod.glob(pattern):
        match = re.search(r'_ckpt_(\d+)\.parquet$', path)
        if match:
            abs_end = int(match.group(1))
            completed[abs_end] = path
    # Also check for final gpu part
    final = output_path.replace(".parquet", f"_gpu{gpu_id}.parquet")
    if os.path.exists(final):
        completed[-1] = final  # -1 means fully done
    return completed


def translate_batch_sentences(
    translator: ctranslate2.Translator,
    sp: spm.SentencePieceProcessor,
    sentences: list[str],
    batch_size: int = 4096,
    beam_size: int = 1,
    max_input_length: int = 128,
    max_decoding_length: int = 200,
) -> list[str]:
    if not sentences:
        return []

    # Tokenize, truncate
    all_tokens = []
    for s in sentences:
        toks = sp.encode(s, out_type=str)
        if len(toks) > max_input_length:
            toks = toks[:max_input_length]
        all_tokens.append(toks)

    # Translate with local sorting per batch
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


def worker_translate(
    gpu_id: int,
    rows: list,
    row_offset: int,
    output_path: str,
    batch_size: int,
    beam_size: int,
    max_input_length: int,
    max_decoding_length: int,
    compute_type: str,
    checkpoint_every: int,
    resume: bool,
    result_queue: Queue,
):
    print(f"[GPU:{gpu_id}] Loading model (compute_type={compute_type})...", flush=True)
    translator = ctranslate2.Translator(
        CT2_DIR, device="cuda", device_index=gpu_id, compute_type=compute_type,
    )
    sp = spm.SentencePieceProcessor(SPM_PATH)

    chunk_size = checkpoint_every
    all_output_rows = []
    total_sents = 0
    t_start = time.time()
    skipped_chunks = 0

    # Find completed checkpoints for resume
    completed = {}
    if resume:
        completed = find_completed_chunks(output_path, gpu_id)
        if -1 in completed:
            print(f"[GPU:{gpu_id}] Already fully completed! Loading final file.", flush=True)
            part_ds = Dataset.from_parquet(completed[-1])
            result_queue.put((gpu_id, completed[-1], len(part_ds)))
            return
        if completed:
            print(f"[GPU:{gpu_id}] Found {len(completed)} completed checkpoints", flush=True)

    for chunk_start in range(0, len(rows), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(rows))
        abs_start = row_offset + chunk_start
        abs_end = row_offset + chunk_end

        # Check if this chunk is already done
        if resume and abs_end in completed:
            print(f"[GPU:{gpu_id}] Skipping chunk {abs_start}–{abs_end} (checkpoint exists)", flush=True)
            ckpt_ds = Dataset.from_parquet(completed[abs_end])
            all_output_rows.extend(ckpt_ds.to_list())
            skipped_chunks += 1
            continue

        chunk_rows = rows[chunk_start:chunk_end]
        print(f"\n[GPU:{gpu_id}] Chunk {abs_start}–{abs_end} ({len(chunk_rows)} rows)", flush=True)

        all_sentences = []
        boundaries = []
        for row in chunk_rows:
            sents = split_sentences(row["text"])
            start = len(all_sentences)
            all_sentences.extend(sents)
            boundaries.append((start, len(all_sentences)))

        chunk_sents = len(all_sentences)
        total_sents += chunk_sents
        print(f"[GPU:{gpu_id}] Sentences: {chunk_sents}", flush=True)

        t0 = time.time()
        translated_sents = translate_batch_sentences(
            translator, sp, all_sentences,
            batch_size=batch_size,
            beam_size=beam_size,
            max_input_length=max_input_length,
            max_decoding_length=max_decoding_length,
        )
        elapsed = time.time() - t0
        print(f"[GPU:{gpu_id}] Chunk done in {elapsed:.1f}s ({chunk_sents / elapsed:.0f} sents/sec)", flush=True)

        for i, row in enumerate(chunk_rows):
            start, end = boundaries[i]
            trans_sents = translated_sents[start:end]
            all_output_rows.append({
                "text_en": row["text"],
                "text_kk": " ".join(trans_sents),
                "id": row.get("id", str(abs_start + i)),
                "num_sentences": end - start,
            })

        # Checkpoint
        if chunk_end < len(rows):
            ckpt_path = output_path.replace(".parquet", f"_gpu{gpu_id}_ckpt_{abs_end}.parquet")
            Dataset.from_list(all_output_rows).to_parquet(ckpt_path)
            print(f"[GPU:{gpu_id}] Checkpoint: {ckpt_path} ({len(all_output_rows)} rows)", flush=True)

    total_elapsed = time.time() - t_start
    if total_sents > 0:
        print(f"\n[GPU:{gpu_id}] DONE: {len(all_output_rows)} rows, {total_sents} new sents in {total_elapsed:.1f}s "
              f"({total_sents / total_elapsed:.0f} sents/sec, skipped {skipped_chunks} chunks)", flush=True)
    else:
        print(f"\n[GPU:{gpu_id}] DONE: {len(all_output_rows)} rows (all from checkpoints)", flush=True)

    part_path = output_path.replace(".parquet", f"_gpu{gpu_id}.parquet")
    Dataset.from_list(all_output_rows).to_parquet(part_path)
    print(f"[GPU:{gpu_id}] Saved: {part_path}", flush=True)

    result_queue.put((gpu_id, part_path, len(all_output_rows)))


def run_pipeline(args):
    num_rows = args.num_rows
    num_gpus = args.num_gpus
    output_path = args.output

    print(f"Config: {num_rows} rows, {num_gpus} GPU(s), compute={args.compute_type}, "
          f"beam={args.beam_size}, batch={args.batch_size}, "
          f"max_input={args.max_input_length}, max_decode={args.max_decoding_length}, "
          f"resume={args.resume}", flush=True)

    print(f"Loading FineWeb-Edu (streaming, {num_rows} rows)...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu-score-2", split="train", streaming=True)
    t_load = time.time()
    rows = list(islice(ds, num_rows))
    print(f"Collected {len(rows)} rows in {time.time() - t_load:.1f}s", flush=True)

    # Pre-translation filtering
    if args.filter:
        from filters import TextFilter
        print("Running pre-translation filters...", flush=True)
        tf = TextFilter(fuzzy_dedup=args.fuzzy_dedup)
        t_filt = time.time()
        filtered = []
        for i, row in enumerate(rows):
            keep, reason = tf.filter(row.get("text", ""), doc_id=str(i))
            if keep:
                filtered.append(row)
        print(f"Filtering done in {time.time() - t_filt:.1f}s", flush=True)
        print(tf.summary(), flush=True)
        rows = filtered

    if num_gpus == 1:
        result_queue = Queue()
        worker_translate(
            gpu_id=args.device_index, rows=rows, row_offset=0,
            output_path=output_path,
            batch_size=args.batch_size, beam_size=args.beam_size,
            max_input_length=args.max_input_length,
            max_decoding_length=args.max_decoding_length,
            compute_type=args.compute_type,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume, result_queue=result_queue,
        )
        _, part_path, count = result_queue.get()
        os.rename(part_path, output_path)
        print(f"Final output: {output_path} ({count} rows)")
    else:
        chunk = len(rows) // num_gpus
        result_queue = Queue()
        processes = []

        for gpu_id in range(num_gpus):
            start = gpu_id * chunk
            end = start + chunk if gpu_id < num_gpus - 1 else len(rows)
            gpu_rows = rows[start:end]
            print(f"[GPU:{gpu_id}] rows {start}–{end} ({len(gpu_rows)} rows)", flush=True)

            p = Process(
                target=worker_translate,
                args=(
                    gpu_id, gpu_rows, start, output_path,
                    args.batch_size, args.beam_size,
                    args.max_input_length, args.max_decoding_length,
                    args.compute_type, args.checkpoint_every,
                    args.resume, result_queue,
                ),
            )
            p.start()
            processes.append(p)

        results = []
        for _ in processes:
            results.append(result_queue.get())
        for p in processes:
            p.join()

        print(f"\nMerging {len(results)} parts...", flush=True)
        all_rows = []
        for gpu_id, part_path, count in sorted(results):
            part_ds = Dataset.from_parquet(part_path)
            all_rows.extend(part_ds.to_list())
            print(f"  GPU:{gpu_id} → {count} rows from {part_path}")

        merged = Dataset.from_list(all_rows)
        merged.to_parquet(output_path)
        print(f"Merged output: {output_path} ({len(all_rows)} rows)")

    if args.smoke_test:
        out_ds = Dataset.from_parquet(output_path)
        errors = sum(1 for r in out_ds if not r["text_kk"].strip())
        print(f"\n--- Smoke Test ---")
        print(f"  Rows: {len(out_ds)}, Empty: {errors}")
        print(f"  [0] EN: {out_ds[0]['text_en'][:120]}...")
        print(f"  [0] KK: {out_ds[0]['text_kk'][:120]}...")
        print("--- PASSED ---" if errors == 0 else f"--- {errors} warnings ---")


def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu EN→KK translation pipeline (optimized v3)")
    parser.add_argument("--num-rows", type=int, default=100_000)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4096, help="Sentences per batch")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size (1=greedy)")
    parser.add_argument("--max-input-length", type=int, default=128, help="Max input tokens")
    parser.add_argument("--max-decoding-length", type=int, default=200, help="Max output tokens")
    parser.add_argument("--compute-type", type=str, default="float16", choices=["float32", "float16", "int8"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints")
    parser.add_argument("--filter", action="store_true", help="Enable pre-translation filtering")
    parser.add_argument("--fuzzy-dedup", action="store_true", help="Enable MinHash near-dedup (slow, off by default)")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.num_rows = 50
        args.num_gpus = 1
        args.compute_type = "float32"
        if args.output == DEFAULT_OUTPUT:
            args.output = os.path.join(BASE, "smoke_test_output.parquet")

    run_pipeline(args)


if __name__ == "__main__":
    main()
