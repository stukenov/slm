#!/usr/bin/env python3
"""Encode all audio from kzcalm-tts-kk-v1 through Mimi codec, push codes to HF.

Produces a dataset with columns: text, speaker_id, source, emotion, duration, codes, num_frames
where `codes` is a list[list[int]] (num_codebooks x T_frames) of discrete codec tokens.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _encode_batch(
    mimi,
    audio_list: list[np.ndarray],
    meta_list: list[dict],
    results: list[dict],
    device: str,
):
    """Encode a batch of audio arrays through Mimi, append results."""
    max_len = max(len(a) for a in audio_list)
    padded = np.zeros((len(audio_list), 1, max_len), dtype=np.float32)
    lengths = []
    for i, a in enumerate(audio_list):
        padded[i, 0, :len(a)] = a
        lengths.append(len(a))

    waveform = torch.from_numpy(padded).to(device)

    with torch.no_grad():
        codes = mimi.encode(waveform)  # (B, num_codebooks, T_frames)

    codes_np = codes.cpu().numpy()

    for i, meta in enumerate(meta_list):
        actual_frames = int(np.ceil(lengths[i] / 24000 * 12.5))
        sample_codes = codes_np[i, :, :actual_frames]  # (num_codebooks, T)

        results.append({
            **meta,
            "codes": sample_codes.tolist(),
            "num_frames": actual_frames,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset", default="stukenov/kzcalm-tts-kk-v1")
    parser.add_argument("--hf-repo", default="stukenov/kzcalm-mimi-codes-kk-v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config", help="Ignored")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from datasets import load_dataset, Dataset
    from moshi.models import loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load Mimi
    logger.info("Loading Mimi codec...")
    mimi = loaders.get_mimi("kyutai/mimi", device=device)
    mimi.set_num_codebooks(args.num_codebooks)
    for p in mimi.parameters():
        p.requires_grad = False
    logger.info(f"Mimi loaded: {args.num_codebooks} codebooks, 24kHz, 12.5 Hz frame rate")

    # Load dataset (streaming)
    logger.info("Loading dataset (streaming)...")
    ds = load_dataset(args.hf_dataset, split="train", streaming=True)

    results = []
    batch_audio = []
    batch_meta = []
    count = 0
    errors = 0

    for sample in ds:
        audio_arr = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]

        if sr != 24000:
            logger.warning(f"Unexpected sample rate {sr}, skipping")
            errors += 1
            continue

        batch_audio.append(audio_arr)
        batch_meta.append({
            "text": sample["text"],
            "speaker_id": sample["speaker_id"],
            "source": sample["source"],
            "emotion": sample["emotion"],
            "duration": sample["duration"],
        })

        if len(batch_audio) >= args.batch_size:
            _encode_batch(mimi, batch_audio, batch_meta, results, device)
            count += len(batch_audio)
            batch_audio.clear()
            batch_meta.clear()

            if count % 5000 < args.batch_size:
                logger.info(f"  {count} samples encoded, {errors} errors")

            if args.max_samples and count >= args.max_samples:
                break

    # Remaining
    if batch_audio:
        _encode_batch(mimi, batch_audio, batch_meta, results, device)
        count += len(batch_audio)

    logger.info(f"Encoded {count} samples total ({errors} errors)")

    # Build and push — in shards to avoid OOM
    logger.info("Building HF dataset...")
    out_ds = Dataset.from_list(results)
    logger.info(f"Dataset: {out_ds}")

    logger.info(f"Pushing to {args.hf_repo}...")
    out_ds.push_to_hub(args.hf_repo, private=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()
