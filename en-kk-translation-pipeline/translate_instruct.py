#!/usr/bin/env python3
"""
Translate instruct dataset EN→KK using CTranslate2 (HPLT Marian model).

Optimized: collects all sentences from a chunk of rows, translates in large
batches (4096 sentences), then reassembles per-row.

Usage:
    python scripts/translate_instruct.py --smoke-test
    python scripts/translate_instruct.py --num-gpus 2
    python scripts/translate_instruct.py --num-gpus 2 --resume
    python scripts/translate_instruct.py --upload --repo stukenov/sozkz-instruct-chatml-kk-v1
"""

import argparse
import json
import os
import re
import time
from multiprocessing import Process, Queue

import ctranslate2
import sentencepiece as spm
from datasets import Dataset, DatasetDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

CT2_DIR = os.path.join(PROJECT_DIR, "translation-demo", "model_ct2")
SPM_PATH = os.path.join(PROJECT_DIR, "translation-demo", "model_cache", "model.en-kk.spm")

DEFAULT_INPUT = os.path.join(PROJECT_DIR, "data", "instruct_chatml_en.parquet")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "data", "instruct_chatml_kk.parquet")

SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ\d"])')
CODE_BLOCK_RE = re.compile(r'(```[\s\S]*?```)')
MD_PREFIX_RE = re.compile(r'^(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s+)')


def split_sentences(text: str) -> list[str]:
    sents = SENT_RE.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


def messages_to_chatml(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|{m['role']}|>{m['content']}<|end|>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Segment extraction: decompose all messages into translatable segments
# ---------------------------------------------------------------------------

def extract_segments(messages: list[dict]) -> tuple[list[str], list]:
    """Extract translatable text segments from messages.

    Returns:
        sentences: flat list of sentences to translate
        blueprint: structure to reassemble translated sentences back into messages
    """
    sentences = []
    blueprint = []  # per-message reconstruction info

    for m in messages:
        role = m["role"]
        content = m["content"]

        # Skip system messages with code blocks entirely
        if role == "system" and "```" in content:
            blueprint.append({"role": role, "type": "passthrough", "content": content})
            continue

        # Split by code blocks
        parts = CODE_BLOCK_RE.split(content)
        msg_parts = []

        for part in parts:
            if part.startswith("```"):
                msg_parts.append({"type": "code", "text": part})
            else:
                # Process lines, extract sentences
                lines = part.split("\n")
                line_infos = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        line_infos.append({"type": "empty"})
                        continue

                    prefix = ""
                    text_to_translate = stripped
                    md_match = MD_PREFIX_RE.match(stripped)
                    if md_match:
                        prefix = md_match.group(0)
                        text_to_translate = stripped[len(prefix):]

                    if not text_to_translate.strip():
                        line_infos.append({"type": "prefix_only", "prefix": prefix})
                        continue

                    sents = split_sentences(text_to_translate)
                    start_idx = len(sentences)
                    sentences.extend(sents)
                    end_idx = len(sentences)
                    line_infos.append({
                        "type": "text",
                        "prefix": prefix,
                        "sent_start": start_idx,
                        "sent_end": end_idx,
                    })

                msg_parts.append({"type": "lines", "lines": line_infos})

        blueprint.append({"role": role, "type": "translate", "parts": msg_parts})

    return sentences, blueprint


def reassemble(translated_sents: list[str], blueprint: list) -> list[dict]:
    """Reassemble translated sentences back into messages using the blueprint."""
    messages = []
    for entry in blueprint:
        role = entry["role"]
        if entry["type"] == "passthrough":
            messages.append({"role": role, "content": entry["content"]})
            continue

        content_parts = []
        for part in entry["parts"]:
            if part["type"] == "code":
                content_parts.append(part["text"])
            elif part["type"] == "lines":
                rebuilt_lines = []
                for line_info in part["lines"]:
                    if line_info["type"] == "empty":
                        rebuilt_lines.append("")
                    elif line_info["type"] == "prefix_only":
                        rebuilt_lines.append(line_info["prefix"])
                    else:
                        prefix = line_info["prefix"]
                        sents = translated_sents[line_info["sent_start"]:line_info["sent_end"]]
                        rebuilt_lines.append(prefix + " ".join(sents))
                content_parts.append("\n".join(rebuilt_lines))

        messages.append({"role": role, "content": "".join(content_parts)})

    return messages


# ---------------------------------------------------------------------------
# Batch translation
# ---------------------------------------------------------------------------

def translate_sentences_batch(
    translator: ctranslate2.Translator,
    sp: spm.SentencePieceProcessor,
    sentences: list[str],
    batch_size: int = 16384,
    beam_size: int = 1,
    max_input_length: int = 72,
    max_decoding_length: int = 80,
) -> list[str]:
    """Translate a large list of sentences efficiently in batches.

    TURBO: median=15 tokens, p99=64. 21GB free VRAM per 5090.
    - batch_size=32768: saturate 32GB VRAM
    - max_decoding_length=80: 1.1x input (KK ~same length as EN)
    """
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
        # Sort by length for minimal padding waste
        batch_indices.sort(key=lambda i: len(all_tokens[i]))
        batch_tokens = [all_tokens[i] for i in batch_indices]

        results = translator.translate_batch(
            batch_tokens,
            beam_size=beam_size,
            max_decoding_length=max_decoding_length,
            max_input_length=0,  # already truncated above
            replace_unknowns=True,
        )
        for local_idx, global_idx in enumerate(batch_indices):
            translated[global_idx] = sp.decode(results[local_idx].hypotheses[0])

        if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - t0
            done = end
            sps = done / elapsed if elapsed > 0 else 0
            print(f"    [{done}/{len(all_tokens)}] {sps:.0f} sents/sec", flush=True)

    return translated


# ---------------------------------------------------------------------------
# Worker: process chunks of rows
# ---------------------------------------------------------------------------

CHUNK_SIZE = 50000  # rows per chunk — minimize overhead between chunks


def worker_translate(
    gpu_id: int,
    rows: list[dict],
    row_offset: int,
    output_path: str,
    batch_size: int,
    compute_type: str,
    checkpoint_every: int,
    resume: bool,
    result_queue: Queue,
):
    print(f"[GPU:{gpu_id}] Loading CTranslate2 model (compute_type={compute_type})...", flush=True)
    translator = ctranslate2.Translator(
        CT2_DIR, device="cuda", device_index=gpu_id, compute_type=compute_type,
        inter_threads=2,  # overlap CPU↔GPU without OOM
    )
    sp = spm.SentencePieceProcessor(SPM_PATH)

    all_output = []
    t_start = time.time()
    total_sents = 0

    # Resume from checkpoint
    ckpt_path = output_path.replace(".parquet", f"_gpu{gpu_id}_ckpt.parquet")
    start_row = 0
    if resume and os.path.exists(ckpt_path):
        ckpt_ds = Dataset.from_parquet(ckpt_path)
        all_output = ckpt_ds.to_list()
        start_row = len(all_output)
        print(f"[GPU:{gpu_id}] Resumed from checkpoint: {start_row} rows done", flush=True)

    for chunk_start in range(start_row, len(rows), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(rows))
        chunk_rows = rows[chunk_start:chunk_end]

        print(f"\n[GPU:{gpu_id}] Chunk {chunk_start}–{chunk_end} ({len(chunk_rows)} rows)", flush=True)

        # Phase 1: Extract all sentences from all rows in chunk
        t0 = time.time()
        all_sentences = []
        blueprints = []
        for row in chunk_rows:
            messages = json.loads(row["messages"])
            sents, bp = extract_segments(messages)
            all_sentences.extend(sents)
            blueprints.append(bp)

        num_sents = len(all_sentences)
        total_sents += num_sents
        print(f"[GPU:{gpu_id}] Extracted {num_sents} sentences, translating...", flush=True)

        # Phase 2: Translate all sentences in one big batch
        translated_sents = translate_sentences_batch(
            translator, sp, all_sentences,
            batch_size=batch_size,
        )

        # Phase 3: Reassemble per row
        sent_offset = 0
        for i, (row, bp) in enumerate(zip(chunk_rows, blueprints)):
            # Count sentences for this row
            row_sent_count = 0
            for entry in bp:
                if entry["type"] == "translate":
                    for part in entry["parts"]:
                        if part["type"] == "lines":
                            for li in part["lines"]:
                                if li["type"] == "text":
                                    row_sent_count += li["sent_end"] - li["sent_start"]

            translated_messages = reassemble(translated_sents, bp)
            chatml_text = messages_to_chatml(translated_messages)
            all_output.append({
                "id": row["id"],
                "source": row["source"],
                "messages": json.dumps(translated_messages, ensure_ascii=False),
                "num_turns": row["num_turns"],
                "text": chatml_text,
            })

        elapsed = time.time() - t0
        rps = len(chunk_rows) / elapsed if elapsed > 0 else 0
        total_done = len(all_output)
        total_elapsed = time.time() - t_start
        total_rps = (total_done - start_row) / total_elapsed if total_elapsed > 0 else 0
        remaining = len(rows) - total_done
        eta = remaining / total_rps if total_rps > 0 else 0
        print(f"[GPU:{gpu_id}] Chunk done: {rps:.0f} rows/sec ({num_sents/elapsed:.0f} sents/sec), "
              f"total {total_done}/{len(rows)}, ETA {eta/3600:.1f}h", flush=True)

        # Checkpoint
        if total_done % checkpoint_every < CHUNK_SIZE:
            Dataset.from_list(all_output).to_parquet(ckpt_path)
            print(f"[GPU:{gpu_id}] Checkpoint: {len(all_output)} rows", flush=True)

    # Save final
    part_path = output_path.replace(".parquet", f"_gpu{gpu_id}.parquet")
    Dataset.from_list(all_output).to_parquet(part_path)
    total_elapsed = time.time() - t_start
    print(f"\n[GPU:{gpu_id}] DONE: {len(all_output)} rows, {total_sents} sents in {total_elapsed:.1f}s "
          f"({total_sents/total_elapsed:.0f} sents/sec)", flush=True)
    result_queue.put((gpu_id, part_path, len(all_output)))


def main():
    parser = argparse.ArgumentParser(description="Translate instruct dataset EN→KK (batched)")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--compute-type", default="float16", choices=["float32", "float16", "int8"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo", default="stukenov/sozkz-instruct-chatml-kk-v1")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    args = parser.parse_args()

    if args.smoke_test:
        args.num_rows = 100
        args.num_gpus = 1
        args.compute_type = "float32"

    print(f"Loading input: {args.input}", flush=True)
    ds = Dataset.from_parquet(args.input)
    if args.num_rows:
        ds = ds.select(range(min(args.num_rows, len(ds))))
    rows = ds.to_list()
    print(f"Loaded {len(rows)} rows", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.num_gpus == 1:
        result_queue = Queue()
        worker_translate(0, rows, 0, args.output, args.batch_size, args.compute_type,
                         args.checkpoint_every, args.resume, result_queue)
        _, part_path, count = result_queue.get()
        os.rename(part_path, args.output)
    else:
        chunk = len(rows) // args.num_gpus
        result_queue = Queue()
        processes = []
        for gpu_id in range(args.num_gpus):
            start = gpu_id * chunk
            end = start + chunk if gpu_id < args.num_gpus - 1 else len(rows)
            p = Process(target=worker_translate,
                        args=(gpu_id, rows[start:end], start, args.output,
                              args.batch_size, args.compute_type,
                              args.checkpoint_every, args.resume, result_queue))
            p.start()
            processes.append(p)

        results = [result_queue.get() for _ in processes]
        for p in processes:
            p.join()

        print("Merging GPU parts...", flush=True)
        all_output = []
        for gpu_id, part_path, count in sorted(results):
            part = Dataset.from_parquet(part_path)
            all_output.extend(part.to_list())
            print(f"  GPU:{gpu_id} → {count} rows")
        Dataset.from_list(all_output).to_parquet(args.output)

    print(f"Output: {args.output}", flush=True)

    if args.upload:
        print(f"Preparing train/val split (val_ratio={args.val_ratio})...", flush=True)
        final_ds = Dataset.from_parquet(args.output)
        split = final_ds.train_test_split(test_size=args.val_ratio, seed=42)
        dsd = DatasetDict({"train": split["train"], "validation": split["test"]})
        t = len(dsd["train"])
        v = len(dsd["validation"])
        print(f"Train: {t}, Val: {v}", flush=True)
        print(f"Uploading to {args.repo}...", flush=True)
        dsd.push_to_hub(args.repo, private=False)
        print(f"Uploaded: https://huggingface.co/datasets/{args.repo}", flush=True)

    if args.smoke_test:
        out_ds = Dataset.from_parquet(args.output)
        print(f"\n--- Smoke Test ---")
        print(f"Rows: {len(out_ds)}")
        sample = out_ds[0]
        print(f"ID: {sample['id']}")
        print(f"Text preview:\n{sample['text'][:300]}...")
        print("--- PASSED ---")


if __name__ == "__main__":
    main()
