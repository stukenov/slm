#!/usr/bin/env python3
"""Benchmark EN→KK translation: CPU vs GPU speed comparison."""

import os
import time

import ctranslate2
import sentencepiece as spm

BASE = os.path.dirname(os.path.abspath(__file__))
CT2_DIR = os.path.join(BASE, "model_ct2")
SPM_PATH = os.path.join(BASE, "model_cache", "model.en-kk.spm")

# Test sentences (repeat to get ~1000 sentences)
SAMPLE_TEXTS = [
    "The weather is beautiful today, and I want to go for a walk in the park.",
    "Artificial intelligence is transforming the way we live and work.",
    "Kazakhstan is the largest landlocked country in the world.",
    "My grandmother makes the best beshbarmak in the whole village.",
    "Students should read more books to expand their knowledge.",
    "The space program has achieved remarkable progress in recent years.",
    "Clean water is essential for the health of every person on the planet.",
    "Learning a new language opens doors to different cultures and opportunities.",
    "The economy of Central Asia is growing rapidly due to new investments.",
    "Children love playing football in the schoolyard after classes.",
]

SENTENCES = SAMPLE_TEXTS * 100  # 1000 sentences


def bench(device, device_index=0, batch_size=64, beam_size=2):
    print(f"\n{'='*60}")
    print(f"Device: {device} (index={device_index}), batch={batch_size}, beam={beam_size}")
    print(f"{'='*60}")

    translator = ctranslate2.Translator(
        CT2_DIR,
        device=device,
        device_index=device_index,
        inter_threads=1 if device == "cuda" else 2,
        intra_threads=1 if device == "cuda" else 8,
    )
    sp = spm.SentencePieceProcessor(SPM_PATH)

    all_tokens = [sp.encode(s, out_type=str) for s in SENTENCES]

    # Warmup
    translator.translate_batch(all_tokens[:10], beam_size=beam_size)

    # Benchmark
    t0 = time.time()
    for i in range(0, len(all_tokens), batch_size):
        batch = all_tokens[i : i + batch_size]
        translator.translate_batch(batch, beam_size=beam_size)
    elapsed = time.time() - t0

    sps = len(SENTENCES) / elapsed
    print(f"  {len(SENTENCES)} sentences in {elapsed:.2f}s = {sps:.0f} sents/sec")
    return sps


if __name__ == "__main__":
    results = {}

    n_gpu = ctranslate2.get_cuda_device_count()
    print(f"Found {n_gpu} CUDA device(s)")
    for i in range(n_gpu):
        for bs in [64, 128, 256]:
            label = f"GPU:{i} batch={bs}"
            results[label] = bench("cuda", device_index=i, batch_size=bs)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, sps in results.items():
        print(f"  {label:30s} → {sps:6.0f} sents/sec")
