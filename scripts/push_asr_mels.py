"""Push preprocessed ASR mel dataset to HuggingFace Hub.

Loads the already-combined Arrow splits from disk and pushes them.

Usage:
    python scripts/push_asr_mels.py --data_dir /root/slm/data/asr_mels
"""
import argparse
import json
import os
import sys

from datasets import load_from_disk, DatasetDict
from huggingface_hub import HfApi, create_repo

REPO_ID = "stukenov/sozkz-asr-mels-kk-v1"

README_TEMPLATE = """---
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
- 100K<n<1M
---

# SozKZ ASR Mel Spectrograms (Kazakh) v1

Pre-computed **log-mel spectrograms** for Kazakh automatic speech recognition, ready for training encoder-decoder ASR models.

## Dataset Details

| | Train | Validation | Test | Total |
|---|---|---|---|---|
| **Samples** | {train_samples:,} | {val_samples:,} | {test_samples:,} | {total_samples:,} |

### Sources

| Dataset | Samples | Description |
|---|---|---|
| **KazEmoTTS** | {kazemotts:,} | Emotional Kazakh TTS corpus (~75h) |
| **KazakhTTS2** | {kazakhtts2:,} | Multi-speaker TTS corpus (~271h, 5 speakers) |
| **OpenSLR-140** | {openslr140:,} | Read speech corpus (~554h) |

**Total audio:** ~900 hours of Kazakh speech

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
mel = sample["mel"]       # flat list → reshape to (80, T)
text = sample["text"]     # kazakh transcription
duration = sample["duration"]
```

## License

MIT

## Citation

Part of the [SozKZ](https://huggingface.co/stukenov) project — building open Kazakh language technology.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/slm/data/asr_mels")
    parser.add_argument("--repo_id", type=str, default=REPO_ID)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            token = open(token_path).read().strip()
    if not token:
        print("ERROR: No HF token found")
        sys.exit(1)

    # Load metadata
    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Dataset: {meta['total_samples']:,} samples")
    print(f"Splits: {meta['splits']}")
    print(f"Sources: {meta['sources']}")

    # Create repo
    print(f"\nCreating repo: {args.repo_id}")
    create_repo(args.repo_id, token=token, exist_ok=True, repo_type="dataset")

    # Load splits
    print("\nLoading splits from disk...")
    ds_dict = DatasetDict({
        "train": load_from_disk(os.path.join(args.data_dir, "train")),
        "validation": load_from_disk(os.path.join(args.data_dir, "validation")),
        "test": load_from_disk(os.path.join(args.data_dir, "test")),
    })
    print(ds_dict)

    # Push to hub
    print(f"\nPushing to {args.repo_id}...")
    ds_dict.push_to_hub(args.repo_id, token=token)
    print("Dataset pushed!")

    # Generate and upload README
    sources = meta["sources"]
    kazakhtts2_total = sum(v for k, v in sources.items() if k.startswith("kazakhtts2"))

    readme = README_TEMPLATE.format(
        train_samples=meta["splits"]["train"],
        val_samples=meta["splits"]["validation"],
        test_samples=meta["splits"]["test"],
        total_samples=meta["total_samples"],
        kazemotts=sources.get("kazemotts", 0),
        kazakhtts2=kazakhtts2_total,
        openslr140=sources.get("openslr140", 0),
    )

    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )
    print("README uploaded!")

    # Upload metadata
    api.upload_file(
        path_or_fileobj=json.dumps(meta, indent=2).encode(),
        path_in_repo="metadata.json",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=token,
    )
    print(f"\nDone! https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
