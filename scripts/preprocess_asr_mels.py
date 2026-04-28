"""Preprocess Kazakh ASR datasets into mel spectrograms.

Processes datasets from small to large, computes log-mel spectrograms,
saves as compact float16 numpy arrays + text, then combines everything
into one shuffled HF dataset.

Usage:
    python scripts/preprocess_asr_mels.py --output_dir /root/slm/data/asr_mels
    python scripts/preprocess_asr_mels.py --output_dir /root/slm/data/asr_mels --push_to_hub
    python scripts/preprocess_asr_mels.py --datasets kazemotts,openslr140  # specific datasets only
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

# ── Mel config (must match training) ────────────────────────────
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
MAX_AUDIO_SEC = 30  # skip longer clips

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
)


def compute_mel(waveform, sr):
    """Compute log-mel spectrogram from waveform tensor. Returns (n_mels, T) float16 numpy."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    duration = waveform.shape[1] / SAMPLE_RATE
    if duration > MAX_AUDIO_SEC:
        return None, duration

    mel = mel_transform(waveform)  # (1, n_mels, T)
    mel = torch.log(torch.clamp(mel, min=1e-10))
    mel = mel.squeeze(0).numpy().astype(np.float16)  # (n_mels, T)
    return mel, duration


def mel_to_bytes(mel):
    """Serialize mel (n_mels, T) float16 numpy to bytes for compact storage."""
    buf = io.BytesIO()
    np.save(buf, mel)
    return buf.getvalue()


def bytes_to_mel(b):
    """Deserialize bytes back to mel numpy array."""
    return np.load(io.BytesIO(b))


# ── Dataset processors ──────────────────────────────────────────

def process_openslr140(output_dir, max_samples=None):
    """OpenSLR-140: 554h, HF-native. Download mode for random access.

    Streaming dies on corrupt audio (soundfile error kills iterator).
    Download mode allows per-sample try/except via ds[i] indexing.
    Needs ~8GB HF cache (parquet), not the full 56GB raw audio.
    """
    from datasets import load_dataset
    print("\n[openslr140] Processing OpenSLR-140 (554h, download mode)...")

    ds = load_dataset(
        "voice-biomarkers/openslr-140-hq-Kazakh",
        split="train",
        trust_remote_code=True,
    )
    return _process_downloaded(ds, "transcription", output_dir, "openslr140", max_samples)


def process_kazemotts(output_dir, max_samples=None):
    """KazEmoTTS: 75h, HF dataset. Parquet version works."""
    from datasets import load_dataset
    print("\n[kazemotts] Processing KazEmoTTS (75h)...")

    # akuzdeuov version is parquet-formatted and actually loadable
    sources = [
        ("akuzdeuov/kazakh-emotional-tts", {}),
        ("issai/KazEmoTTS", {}),
        ("ai4kazakh/ISSAI_KazEmoTTS", {}),
    ]
    for hf_id, kwargs in sources:
        for mode in ["streaming", "download"]:
            try:
                streaming = mode == "streaming"
                print(f"  Trying {hf_id} ({mode})...")
                ds = load_dataset(hf_id, split="train", streaming=streaming,
                                  trust_remote_code=True, **kwargs)
                if streaming:
                    return _process_streaming(ds, None, output_dir, "kazemotts", max_samples)
                else:
                    return _process_downloaded(ds, None, output_dir, "kazemotts", max_samples)
            except Exception as e:
                print(f"  Failed: {e}")

    print("  All KazEmoTTS sources failed. Skipping.")
    return 0


def process_kazakhtts2(output_dir, max_samples=None):
    """KazakhTTS2: 271h, 5 speakers, parquet on ai4kazakh. Loads per-speaker configs."""
    from datasets import load_dataset, concatenate_datasets
    print("\n[kazakhtts2] Processing KazakhTTS2 (271h, 5 speakers)...")

    speakers = ["f1", "f2", "f3", "m1", "m2"]
    total = 0

    # Try loading all speakers and concatenating
    try:
        print("  Trying ai4kazakh/ISSAI_KazakhTTS2 (streaming, all speakers)...")
        for spk in speakers:
            print(f"  Speaker {spk}...")
            ds = load_dataset("ai4kazakh/ISSAI_KazakhTTS2", spk, split="train",
                              streaming=True, trust_remote_code=True)
            count = _process_streaming(ds, None, output_dir, f"kazakhtts2_{spk}", max_samples)
            total += count
        return total
    except Exception as e:
        print(f"  Per-speaker streaming failed: {e}")

    # Fallback: try without config name
    try:
        print("  Trying ai4kazakh/ISSAI_KazakhTTS2 (streaming, no config)...")
        ds = load_dataset("ai4kazakh/ISSAI_KazakhTTS2", split="train",
                          streaming=True, trust_remote_code=True)
        return _process_streaming(ds, None, output_dir, "kazakhtts2", max_samples)
    except Exception as e:
        print(f"  Failed: {e}")

    # Fallback: issai/KazakhTTS
    try:
        print("  Trying issai/KazakhTTS (streaming)...")
        ds = load_dataset("issai/KazakhTTS", split="train",
                          streaming=True, trust_remote_code=True)
        return _process_streaming(ds, None, output_dir, "kazakhtts2", max_samples)
    except Exception as e:
        print(f"  Failed: {e}. Skipping.")
        return 0


def process_openslr102(output_dir, max_samples=None):
    """OpenSLR-102: 332h. Try HF mirrors."""
    from datasets import load_dataset
    print("\n[openslr102] Processing OpenSLR-102 (332h)...")

    sources = [
        ("openslr/openslr", {"name": "SLR102"}),
        ("issai/kazakh_speech_corpus", {}),
    ]
    for hf_id, kwargs in sources:
        try:
            print(f"  Trying {hf_id} (streaming)...")
            ds = load_dataset(hf_id, split="train", streaming=True, trust_remote_code=True, **kwargs)
            return _process_streaming(ds, None, output_dir, "openslr102", max_samples)
        except Exception as e:
            print(f"  Failed: {e}")

    print("  All OpenSLR-102 sources failed. Skipping.")
    return 0


def process_ksc2(output_dir, max_samples=None):
    """KSC2: 1,200h. Largest dataset."""
    from datasets import load_dataset
    print("\n[ksc2] Processing KSC2 (1,200h)...")

    try:
        print("  Trying streaming...")
        ds = load_dataset("issai/Kazakh_Speech_Corpus_2", split="train",
                          streaming=True, trust_remote_code=True)
        return _process_streaming(ds, None, output_dir, "ksc2", max_samples)
    except Exception as e:
        print(f"  Streaming failed: {e}")

    try:
        print("  Trying download...")
        ds = load_dataset("issai/Kazakh_Speech_Corpus_2", split="train",
                          trust_remote_code=True)
        return _process_downloaded(ds, None, output_dir, "ksc2", max_samples)
    except Exception as e:
        print(f"  Download failed: {e}. Skipping.")
        return 0


def _find_text_column(example):
    """Auto-detect text column name."""
    for col in ["sentence", "transcription", "text", "txt", "normalized_text"]:
        if col in example:
            return col
    raise ValueError(f"No text column found in: {list(example.keys())}")


def _find_audio_column(example):
    """Auto-detect audio column name."""
    for col in ["audio", "wav", "speech", "input_values"]:
        if col in example:
            return col
    raise ValueError(f"No audio column found in: {list(example.keys())}")


def _process_streaming_raw_audio(ds, text_col, output_dir, source_name, max_samples=None):
    """Process streaming dataset where audio is raw bytes (decode=False).
    Uses torchaudio to decode, which handles more formats than soundfile."""
    shard_dir = Path(output_dir) / source_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    shard_size = 5000
    buffer_mels, buffer_texts, buffer_durations = [], [], []
    total = 0
    skipped = 0
    total_hours = 0.0
    t0 = time.time()

    ds_iter = iter(ds)
    i = 0
    while True:
        if max_samples and total >= max_samples:
            break

        try:
            example = next(ds_iter)
        except StopIteration:
            break
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Iter error on sample {i}: {type(e).__name__}")
            i += 1
            continue

        try:
            text = example.get(text_col, "")
            if not text or not text.strip():
                skipped += 1
                i += 1
                continue

            audio = example.get("audio", {})
            audio_bytes = audio.get("bytes")
            if audio_bytes is None:
                skipped += 1
                i += 1
                continue

            # Decode with torchaudio from bytes
            buf = io.BytesIO(audio_bytes)
            try:
                waveform, sr = torchaudio.load(buf)
            except Exception:
                skipped += 1
                if skipped <= 10:
                    print(f"  Audio decode error on sample {i}, skipping")
                i += 1
                continue

            mel, duration = compute_mel(waveform, sr)
            if mel is None:
                skipped += 1
                i += 1
                continue

            buffer_mels.append(mel_to_bytes(mel))
            buffer_texts.append(text.strip())
            buffer_durations.append(duration)
            total += 1
            total_hours += duration / 3600

            if len(buffer_mels) >= shard_size:
                _save_shard(shard_dir, shard_idx, buffer_mels, buffer_texts, buffer_durations, source_name)
                shard_idx += 1
                buffer_mels, buffer_texts, buffer_durations = [], [], []

            if total % 1000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  {source_name}: {total} samples, {total_hours:.1f}h, "
                      f"{skipped} skipped, {rate:.0f} samples/s")

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Error on sample {i}: {e}")
        i += 1

    if buffer_mels:
        _save_shard(shard_dir, shard_idx, buffer_mels, buffer_texts, buffer_durations, source_name)

    elapsed = time.time() - t0
    print(f"  {source_name} done: {total} samples, {total_hours:.1f}h, "
          f"{skipped} skipped, {elapsed:.0f}s")
    return total


def _process_streaming(ds, text_col, output_dir, source_name, max_samples=None):
    """Process a streaming HF dataset, save sharded npz files."""
    shard_dir = Path(output_dir) / source_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    shard_size = 5000
    buffer_mels = []
    buffer_texts = []
    buffer_durations = []
    total = 0
    skipped = 0
    total_hours = 0.0
    text_col_resolved = text_col
    audio_col_resolved = None
    t0 = time.time()

    ds_iter = iter(ds)
    i = 0
    while True:
        if max_samples and total >= max_samples:
            break

        # Wrap iteration itself — HF datasets can throw on audio decode
        try:
            example = next(ds_iter)
        except StopIteration:
            break
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Decode error on sample {i}: {type(e).__name__}: {e}")
            i += 1
            continue

        # Auto-detect columns on first sample
        if text_col_resolved is None:
            text_col_resolved = _find_text_column(example)
            print(f"  Text column: {text_col_resolved}")
        if audio_col_resolved is None:
            audio_col_resolved = _find_audio_column(example)
            print(f"  Audio column: {audio_col_resolved}")

        try:
            audio = example[audio_col_resolved]
            text = example[text_col_resolved]

            if not text or not text.strip():
                skipped += 1
                i += 1
                continue

            if isinstance(audio, dict):
                waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
                sr = audio["sampling_rate"]
            else:
                skipped += 1
                i += 1
                continue

            mel, duration = compute_mel(waveform, sr)
            if mel is None:
                skipped += 1
                i += 1
                continue

            buffer_mels.append(mel_to_bytes(mel))
            buffer_texts.append(text.strip())
            buffer_durations.append(duration)
            total += 1
            total_hours += duration / 3600

            if len(buffer_mels) >= shard_size:
                _save_shard(shard_dir, shard_idx, buffer_mels, buffer_texts, buffer_durations, source_name)
                shard_idx += 1
                buffer_mels, buffer_texts, buffer_durations = [], [], []

            if total % 1000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  {source_name}: {total} samples, {total_hours:.1f}h, "
                      f"{skipped} skipped, {rate:.0f} samples/s")

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Error on sample {i}: {e}")

        i += 1

    if buffer_mels:
        _save_shard(shard_dir, shard_idx, buffer_mels, buffer_texts, buffer_durations, source_name)

    elapsed = time.time() - t0
    print(f"  {source_name} done: {total} samples, {total_hours:.1f}h, "
          f"{skipped} skipped, {elapsed:.0f}s")
    return total


def _process_downloaded(ds, text_col, output_dir, source_name, max_samples=None):
    """Process a fully-downloaded HF dataset."""
    shard_dir = Path(output_dir) / source_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    cols = ds.column_names
    if text_col is None:
        for c in ["sentence", "transcription", "text", "txt", "normalized_text"]:
            if c in cols:
                text_col = c
                break
    audio_col = None
    for c in ["audio", "wav", "speech"]:
        if c in cols:
            audio_col = c
            break

    if text_col is None or audio_col is None:
        print(f"  Cannot find columns. Available: {cols}")
        return 0

    print(f"  Columns: audio={audio_col}, text={text_col}")

    shard_idx = 0
    shard_size = 5000
    buffer_mels, buffer_texts, buffer_durations = [], [], []
    total = 0
    skipped = 0
    total_hours = 0.0
    t0 = time.time()

    for i in range(len(ds)):
        if max_samples and total >= max_samples:
            break

        try:
            example = ds[i]
            audio = example[audio_col]
            text = example[text_col]

            if not text or not text.strip():
                skipped += 1
                continue

            if isinstance(audio, dict):
                waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
                sr = audio["sampling_rate"]
            else:
                skipped += 1
                continue

            mel, duration = compute_mel(waveform, sr)
            if mel is None:
                skipped += 1
                continue

            buffer_mels.append(mel_to_bytes(mel))
            buffer_texts.append(text.strip())
            buffer_durations.append(duration)
            total += 1
            total_hours += duration / 3600

            if len(buffer_mels) >= shard_size:
                _save_shard(shard_dir, shard_idx, buffer_mels, buffer_texts, buffer_durations, source_name)
                shard_idx += 1
                buffer_mels, buffer_texts, buffer_durations = [], [], []

            if total % 1000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  {source_name}: {total} samples, {total_hours:.1f}h, "
                      f"{skipped} skipped, {rate:.0f} samples/s")

        except Exception as e:
            skipped += 1
            continue

    if buffer_mels:
        _save_shard(shard_dir, shard_idx, buffer_mels, buffer_texts, buffer_durations, source_name)

    elapsed = time.time() - t0
    print(f"  {source_name} done: {total} samples, {total_hours:.1f}h, "
          f"{skipped} skipped, {elapsed:.0f}s")
    return total


def _save_shard(shard_dir, shard_idx, mels, texts, durations, source):
    """Save a shard as a compact JSON + binary file pair."""
    # Save metadata as JSON
    meta_path = shard_dir / f"shard_{shard_idx:05d}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "texts": texts,
            "durations": [float(d) for d in durations],
            "source": source,
            "count": len(mels),
        }, f, ensure_ascii=False)

    # Save mel bytes as individual .npy files in a subdirectory
    mel_dir = shard_dir / f"shard_{shard_idx:05d}_mels"
    mel_dir.mkdir(exist_ok=True)
    for j, mel_bytes in enumerate(mels):
        mel = bytes_to_mel(mel_bytes)
        np.save(mel_dir / f"{j:05d}.npy", mel)

    total_size = sum(f.stat().st_size for f in mel_dir.glob("*.npy")) / 1e6
    print(f"  Saved shard_{shard_idx:05d}: {len(mels)} samples, {total_size:.1f} MB mels")


# ── Combine & push ──────────────────────────────────────────────

def combine_and_push(output_dir, repo_id, token):
    """Load all shards, shuffle, create HF dataset, push."""
    from datasets import Dataset

    output_dir = Path(output_dir)
    all_mels = []
    all_texts = []
    all_durations = []
    all_sources = []

    for source_dir in sorted(output_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        for meta_path in sorted(source_dir.glob("shard_*_meta.json")):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            shard_name = meta_path.stem.replace("_meta", "")
            mel_dir = source_dir / f"{shard_name}_mels"

            for j in range(meta["count"]):
                mel_path = mel_dir / f"{j:05d}.npy"
                mel = np.load(mel_path)
                all_mels.append(mel_to_bytes(mel))

            all_texts.extend(meta["texts"])
            all_durations.extend(meta["durations"])
            all_sources.extend([meta["source"]] * meta["count"])

    print(f"\nTotal: {len(all_mels)} samples from {len(set(all_sources))} sources")
    for src in sorted(set(all_sources)):
        cnt = all_sources.count(src)
        print(f"  {src}: {cnt} samples")

    # Shuffle
    print("Shuffling...")
    indices = list(range(len(all_mels)))
    import random
    random.seed(42)
    random.shuffle(indices)
    all_mels = [all_mels[i] for i in indices]
    all_texts = [all_texts[i] for i in indices]
    all_durations = [all_durations[i] for i in indices]
    all_sources = [all_sources[i] for i in indices]

    # Split: 94% train, 5% val, 1% test
    n = len(all_mels)
    n_train = int(n * 0.94)
    n_val = int(n * 0.05)

    splits = {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }

    for split_name, (start, end) in splits.items():
        print(f"\nCreating {split_name} split: {end - start} samples...")
        ds = Dataset.from_dict({
            "mel": all_mels[start:end],
            "text": all_texts[start:end],
            "duration": all_durations[start:end],
            "source": all_sources[start:end],
        })

        if repo_id and token:
            print(f"Pushing {split_name} to {repo_id}...")
            ds.push_to_hub(repo_id, split=split_name, token=token)
        else:
            save_path = output_dir / split_name
            ds.save_to_disk(str(save_path))
            print(f"Saved to {save_path}")

    # Save metadata
    meta = {
        "total_samples": n,
        "splits": {k: v[1] - v[0] for k, v in splits.items()},
        "sources": {s: all_sources.count(s) for s in sorted(set(all_sources))},
        "mel_config": {
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "dtype": "float16",
        },
        "max_audio_sec": MAX_AUDIO_SEC,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {meta_path}")


# ── Main ────────────────────────────────────────────────────────

DATASETS = {
    "kazemotts": (process_kazemotts, "KazEmoTTS (75h)"),
    "kazakhtts2": (process_kazakhtts2, "KazakhTTS2 (271h)"),
    "openslr102": (process_openslr102, "OpenSLR-102 (332h)"),
    "openslr140": (process_openslr140, "OpenSLR-140 (554h)"),
    "ksc2": (process_ksc2, "KSC2 (1,200h)"),
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess Kazakh ASR datasets to mel spectrograms")
    parser.add_argument("--output_dir", type=str, default="./data/asr_mels",
                        help="Where to save processed mels")
    parser.add_argument("--datasets", type=str, default="kazemotts,kazakhtts2,openslr102,openslr140,ksc2",
                        help="Comma-separated dataset names to process (order: small to large)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per dataset (for testing)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push combined dataset to HuggingFace Hub")
    parser.add_argument("--repo_id", type=str, default="stukenov/sozkz-asr-mels-kk-v1",
                        help="HF repo ID for push")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    print(f"Datasets to process: {dataset_names}")
    print(f"Output: {output_dir}")
    print(f"Mel config: sr={SAMPLE_RATE}, n_mels={N_MELS}, n_fft={N_FFT}, hop={HOP_LENGTH}")
    print(f"Max audio: {MAX_AUDIO_SEC}s")

    # Get HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            token = open(token_path).read().strip()

    results = {}
    for name in dataset_names:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
            continue
        proc_fn, desc = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"Processing: {desc}")
        print(f"{'='*60}")
        count = proc_fn(args.output_dir, max_samples=args.max_samples)
        results[name] = count

    print(f"\n{'='*60}")
    print("RESULTS:")
    for name, count in results.items():
        print(f"  {name}: {count} samples")
    print(f"  TOTAL: {sum(results.values())} samples")
    print(f"{'='*60}")

    if args.push_to_hub:
        combine_and_push(args.output_dir, args.repo_id, token)
    else:
        combine_and_push(args.output_dir, None, None)
        print(f"\nTo push to HF Hub, re-run with --push_to_hub")


if __name__ == "__main__":
    main()
