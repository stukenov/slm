"""Preprocess KSC2 from extracted local files (flac+txt pairs) into mel spectrograms.

KSC2 is stored as split tar.gz on HF, which HF datasets can't load.
This script processes the extracted ISSAI_KSC2/ directory directly.

Usage:
    python scripts/preprocess_ksc2_local.py \
        --ksc2_dir /root/slm/data/ISSAI_KSC2 \
        --output_dir /root/slm/data/asr_mels
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── Mel config (must match other datasets) ────────────────────
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
MAX_AUDIO_SEC = 30
SHARD_SIZE = 5000


def compute_mel(audio_path):
    """Load audio and compute log-mel spectrogram."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    duration = waveform.shape[1] / SAMPLE_RATE
    if duration > MAX_AUDIO_SEC or duration < 0.1:
        return None, duration

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-10))
    mel = mel.squeeze(0).numpy().astype(np.float16)
    return mel, duration


def find_pairs(ksc2_dir):
    """Find all flac+txt pairs in KSC2 directory."""
    pairs = []
    ksc2_path = Path(ksc2_dir)
    for split_dir in ["Train", "Dev", "Test"]:
        split_path = ksc2_path / split_dir
        if not split_path.exists():
            continue
        for subdir in split_path.iterdir():
            if not subdir.is_dir():
                continue
            flac_files = sorted(subdir.glob("*.flac"))
            for flac in flac_files:
                txt = flac.with_suffix(".txt")
                if txt.exists():
                    pairs.append((str(flac), str(txt), "ksc2"))
    return pairs


def save_shard(mels, texts, durations, sources, shard_idx, output_dir):
    """Save a shard of mel spectrograms using npy with pickle (required for ragged arrays)."""
    shard_dir = Path(output_dir) / "ksc2"
    shard_dir.mkdir(parents=True, exist_ok=True)

    mel_path = shard_dir / f"shard_{shard_idx:05d}_mels.npy"
    meta_path = shard_dir / f"shard_{shard_idx:05d}_meta.json"

    # Save ragged mel arrays - create object array explicitly to avoid broadcasting
    arr = np.empty(len(mels), dtype=object)
    for idx, m in enumerate(mels):
        arr[idx] = m
    np.save(mel_path, arr, allow_pickle=True)

    meta = [
        {"text": t, "duration": d, "source": s, "mel_shape": list(m.shape)}
        for t, d, s, m in zip(texts, durations, sources, mels)
    ]
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    size_mb = mel_path.stat().st_size / 1e6
    print(f"  Saved shard_{shard_idx:05d}: {len(mels)} samples, {size_mb:.1f} MB mels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ksc2_dir", type=str, default="/root/slm/data/ISSAI_KSC2")
    parser.add_argument("--output_dir", type=str, default="/root/slm/data/asr_mels")
    args = parser.parse_args()

    print(f"Finding flac+txt pairs in {args.ksc2_dir}...")
    pairs = find_pairs(args.ksc2_dir)
    print(f"Found {len(pairs)} pairs")

    if not pairs:
        print("No pairs found! Check directory structure.")
        return

    # Check for existing shards to resume
    shard_dir = Path(args.output_dir) / "ksc2"
    existing_shards = list(shard_dir.glob("shard_*_meta.json")) if shard_dir.exists() else []
    existing_samples = 0
    if existing_shards:
        for meta_file in existing_shards:
            with open(meta_file) as f:
                existing_samples += len(json.load(f))
        print(f"Found {len(existing_shards)} existing shards ({existing_samples} samples), resuming...")

    start_idx = existing_samples
    shard_idx = len(existing_shards)

    mels, texts, durations, sources = [], [], [], []
    processed = 0
    skipped = 0
    total_hours = 0
    t0 = time.time()

    for i, (flac_path, txt_path, source) in enumerate(pairs):
        if i < start_idx:
            continue

        try:
            text = open(txt_path, "r", encoding="utf-8").read().strip()
            if not text:
                skipped += 1
                continue

            mel, duration = compute_mel(flac_path)
            if mel is None:
                skipped += 1
                continue

            mels.append(mel)
            texts.append(text)
            durations.append(round(duration, 2))
            sources.append(source)
            processed += 1
            total_hours += duration / 3600

            if len(mels) >= SHARD_SIZE:
                save_shard(mels, texts, durations, sources, shard_idx, args.output_dir)
                mels, texts, durations, sources = [], [], [], []
                shard_idx += 1

            if processed % 1000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  ksc2: {processed} samples, {total_hours:.1f}h, {skipped} skipped, {rate:.0f} samples/s")

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Error at {flac_path}: {e}")
            continue

    # Save remaining
    if mels:
        save_shard(mels, texts, durations, sources, shard_idx, args.output_dir)

    elapsed = time.time() - t0
    print(f"\n  ksc2 done: {processed} samples, {total_hours:.1f}h, {skipped} skipped, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
