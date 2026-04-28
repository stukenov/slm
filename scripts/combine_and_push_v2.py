"""Combine v1 (from HF) + KSC2 (local shards) — memory-efficient version.

Instead of loading all mels into RAM, converts KSC2 shards to Arrow dataset
on disk shard-by-shard, then concatenates with v1 and pushes.

Usage:
    python scripts/combine_and_push_v2.py
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi

REPO_ID = "stukenov/sozkz-asr-mels-kk-v1"
KSC2_SHARDS_DIR = "/root/slm/data/asr_mels/ksc2"
KSC2_ARROW_DIR = "/root/slm/data/asr_mels/ksc2_arrow"
SEED = 42
BATCH_SIZE = 10  # shards per batch


def convert_ksc2_to_arrow():
    """Convert KSC2 npy shards to HF Arrow dataset on disk, batch by batch."""
    shard_dir = Path(KSC2_SHARDS_DIR)
    arrow_dir = Path(KSC2_ARROW_DIR)

    # Check if already converted
    if arrow_dir.exists() and (arrow_dir / "dataset_info.json").exists():
        print("KSC2 Arrow dataset already exists, loading...")
        return load_from_disk(str(arrow_dir))

    meta_files = sorted(shard_dir.glob("shard_*_meta.json"))
    mel_files = sorted(shard_dir.glob("shard_*_mels.npy"))
    print(f"Converting {len(meta_files)} KSC2 shards to Arrow...")

    # Process in batches to limit RAM
    all_datasets = []
    for batch_start in range(0, len(meta_files), BATCH_SIZE):
        batch_meta = meta_files[batch_start:batch_start + BATCH_SIZE]
        batch_mels = mel_files[batch_start:batch_start + BATCH_SIZE]

        mels_list = []
        texts = []
        durations = []
        sources = []

        for meta_f, mel_f in zip(batch_meta, batch_mels):
            meta = json.loads(meta_f.read_text())
            # allow_pickle needed for ragged object arrays (our own generated data)
            mels = np.load(str(mel_f), allow_pickle=True)

            for m, entry in zip(mels, meta):
                # Serialize mel as npy bytes to match v1 format (binary)
                buf = io.BytesIO()
                np.save(buf, m)
                mels_list.append(buf.getvalue())
                texts.append(entry["text"])
                durations.append(entry["duration"])
                sources.append(entry["source"])

        ds = Dataset.from_dict({
            "mel": mels_list,
            "text": texts,
            "duration": durations,
            "source": sources,
        })
        all_datasets.append(ds)

        total = sum(len(d) for d in all_datasets)
        print(f"  Batch {batch_start//BATCH_SIZE + 1}: {len(mels_list)} samples (total: {total})")

        # Free memory
        del mels_list, texts, durations, sources

    print("Concatenating batches...")
    combined = concatenate_datasets(all_datasets)
    del all_datasets

    print(f"Saving KSC2 Arrow to {arrow_dir}...")
    combined.save_to_disk(str(arrow_dir))
    print(f"  Saved: {len(combined)} samples")
    return combined


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            token = open(token_path).read().strip()
    if not token:
        print("ERROR: No HF token found")
        sys.exit(1)

    # Step 1: Convert KSC2 shards to Arrow (memory-efficient)
    print("Step 1: Converting KSC2 shards to Arrow...")
    ksc2 = convert_ksc2_to_arrow()
    print(f"  KSC2: {len(ksc2)} samples")

    # Step 2: Load v1 from HF
    print("\nStep 2: Loading v1 dataset from HF...")
    v1 = load_dataset(REPO_ID, token=token)
    v1_total = len(v1["train"]) + len(v1["validation"]) + len(v1["test"])
    print(f"  v1: {v1_total} samples")

    # Combine all v1 splits
    v1_all = concatenate_datasets([v1["train"], v1["validation"], v1["test"]])
    del v1

    # Step 3: Combine
    n_combined = len(v1_all) + len(ksc2)
    print(f"\nStep 3: Combining {len(v1_all)} + {len(ksc2)} = {n_combined} samples...")
    combined = concatenate_datasets([v1_all, ksc2])
    del v1_all, ksc2

    # Step 4: Shuffle
    print("Step 4: Shuffling...")
    combined = combined.shuffle(seed=SEED)

    # Step 5: Split 94/5/1
    n = len(combined)
    n_val = int(n * 0.05)
    n_test = int(n * 0.01)
    n_train = n - n_val - n_test

    print(f"Step 5: Splitting — train: {n_train}, val: {n_val}, test: {n_test}")

    ds_dict = DatasetDict({
        "train": combined.select(range(n_train)),
        "validation": combined.select(range(n_train, n_train + n_val)),
        "test": combined.select(range(n_train + n_val, n)),
    })
    del combined

    # Step 6: Push to HF
    print(f"\nStep 6: Pushing to {REPO_ID}...")
    ds_dict.push_to_hub(REPO_ID, token=token)
    print("Dataset pushed!")

    # Step 7: Update README
    print("Step 7: Updating README...")
    sources = {}
    for split in ds_dict.values():
        for s in split["source"]:
            sources[s] = sources.get(s, 0) + 1

    kazakhtts2_total = sum(v for k, v in sources.items() if k.startswith("kazakhtts2"))

    readme = f"""---
language:
- kk
license: mit
task_categories:
- automatic-speech-recognition
tags:
- speech
- kazakh
- mel-spectrogram
- asr
- sozkz
size_categories:
- 1M<n<10M
---

# SozKZ ASR Mel Spectrograms (Kazakh) v1

Pre-computed **log-mel spectrograms** for Kazakh automatic speech recognition, ready for training encoder-decoder ASR models.

## Dataset Details

| | Train | Validation | Test | Total |
|---|---|---|---|---|
| **Samples** | {n_train:,} | {n_val:,} | {n_test:,} | {n:,} |

### Sources

| Dataset | Samples | Description |
|---|---|---|
| **KSC2** | {sources.get('ksc2', 0):,} | Kazakh Speech Corpus 2 (~1,200h, crowdsourced + podcasts) |
| **OpenSLR-140** | {sources.get('openslr140', 0):,} | Read speech corpus (~554h) |
| **KazEmoTTS** | {sources.get('kazemotts', 0):,} | Emotional Kazakh TTS corpus (~75h) |
| **KazakhTTS2** | {kazakhtts2_total:,} | Multi-speaker TTS corpus (~271h, 5 speakers) |

**Total audio:** ~2,100 hours of Kazakh speech

### Mel Configuration

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Mel bins | 80 |
| FFT size | 400 |
| Hop length | 160 |
| Max audio | 30 seconds |
| Storage dtype | float16 |

### Features

- `mel` — Log-mel spectrogram as a list of floats (shape: `[n_mels, time_frames]`, stored flat, dtype float16)
- `text` — Transcription text (Kazakh)
- `duration` — Audio duration in seconds
- `source` — Source dataset name

### Split Strategy

- **Train:** 94%
- **Validation:** 5%
- **Test:** 1%
- All splits are **shuffled** across all source datasets (seed=42)

## Usage

```python
from datasets import load_dataset

ds = load_dataset("stukenov/sozkz-asr-mels-kk-v1")

# Access a sample
sample = ds["train"][0]
mel = sample["mel"]       # flat list -> reshape to (80, T)
text = sample["text"]     # kazakh transcription
duration = sample["duration"]
```

## License

MIT

## Citation

Part of the [SozKZ](https://huggingface.co/stukenov) project — building open Kazakh language technology.
"""

    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
    )
    print("README updated!")

    meta = {
        "total_samples": n,
        "splits": {"train": n_train, "validation": n_val, "test": n_test},
        "sources": sources,
        "mel_config": {
            "sample_rate": 16000, "n_mels": 80, "n_fft": 400,
            "hop_length": 160, "dtype": "float16",
        },
        "max_audio_sec": 30,
    }
    api.upload_file(
        path_or_fileobj=json.dumps(meta, indent=2).encode(),
        path_in_repo="metadata.json",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
    )

    print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")
    print(f"Total: {n:,} samples from {len(sources)} sources")


if __name__ == "__main__":
    main()
