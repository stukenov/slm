#!/usr/bin/env python3
"""Upload README for kzcalm-mimi-codes-kk-v1."""
from huggingface_hub import HfApi

README = r'''---
dataset_info:
  features:
    - name: text
      dtype: string
    - name: speaker_id
      dtype: string
    - name: source
      dtype: string
    - name: emotion
      dtype: string
    - name: duration
      dtype: float64
    - name: codes
      sequence:
        sequence: int64
    - name: num_frames
      dtype: int64
  splits:
    - name: train
      num_examples: 232350
language:
  - kk
license: cc-by-4.0
tags:
  - audio-codes
  - mimi
  - codec
  - tts
  - kazakh
pretty_name: KZ-CALM Mimi Codec Codes (Kazakh) v1
size_categories:
  - 100K<n<1M
---

# KZ-CALM Mimi Codec Codes — Kazakh v1

Discrete codec tokens extracted from **232,350** Kazakh speech utterances (~439 hours) using the [Mimi neural audio codec](https://huggingface.co/kyutai/mimi) (Kyutai).

This dataset is the **B2 output** of the [KZ-CALM](https://github.com/saken-tukenov/slm) TTS pipeline — it bridges raw audio and the latent-space generative model.

## Source

All audio comes from [`stukenov/kzcalm-tts-kk-v1`](https://huggingface.co/datasets/stukenov/kzcalm-tts-kk-v1), which merges:

| Source | Samples | Hours | Speakers |
|--------|---------|-------|----------|
| KazakhTTS (ISSAI) | 91,424 | 177.7 | 5 professional |
| KazEmoTTS (ISSAI) | 140,926 | 261.1 | 3, 6 emotions |
| **Total** | **232,350** | **438.8** | **8 unique** |

## Codec Details

| Parameter | Value |
|-----------|-------|
| Codec | Mimi (Kyutai) via `transformers.MimiModel` |
| Weights | [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) |
| Sample rate | 24,000 Hz |
| Frame rate | 12.5 Hz (1 frame = 80 ms) |
| Codebooks | 8 (RVQ) |
| Codebook size | 2,048 entries each |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `text` | `string` | Kazakh utterance text |
| `speaker_id` | `string` | Speaker identifier (e.g. `ISSAI_KazakhTTS_M01`) |
| `source` | `string` | `KazakhTTS` or `KazEmoTTS` |
| `emotion` | `string` | Emotion label (`neutral`, `angry`, `happy`, `sad`, `surprised`, `scared`, or empty) |
| `duration` | `float` | Audio duration in seconds |
| `codes` | `list[list[int]]` | Codec tokens — shape `(8, T)` where T = `num_frames` |
| `num_frames` | `int` | Number of codec frames (`ceil(duration * 12.5)`) |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("stukenov/kzcalm-mimi-codes-kk-v1", split="train")
sample = ds[0]

codes = sample["codes"]   # list of 8 lists (one per codebook)
print(f"Text: {sample['text']}")
print(f"Codebooks: {len(codes)}, Frames: {sample['num_frames']}")
print(f"Duration: {sample['duration']:.1f}s")
```

### Reconstruct audio

```python
import torch
from transformers import MimiModel

model = MimiModel.from_pretrained("kyutai/mimi").cuda()
codes_tensor = torch.tensor([sample["codes"]], device="cuda")  # (1, 8, T)
waveform = model.decode(codes_tensor).audio_values  # (1, 1, T_samples)
```

## Intended Use

- Training latent-space TTS models (flow matching, consistency models)
- Kazakh speech synthesis research
- Codec token language modeling

## Encoding Process

1. Loaded audio via HuggingFace `datasets` streaming (24 kHz mono)
2. Batched (batch_size=16), padded to max length in batch
3. Encoded through `MimiModel.encode()` on GPU (RTX 3060)
4. Trimmed output codes to actual frame count per utterance
5. Total encoding time: ~5 hours on a single RTX 3060

## License

CC-BY-4.0 (following the source datasets from ISSAI).

## Citation

If you use this dataset, please cite the original corpora:

```bibtex
@inproceedings{mussakhojayeva2022kazakhtts,
  title={KazakhTTS: An Open-Source Kazakh Text-to-Speech Synthesis Dataset},
  author={Mussakhojayeva, Saida and Khassanov, Yerbolat and Varol, Huseyin Atakan},
  booktitle={Proc. LREC},
  year={2022}
}

@inproceedings{razakhan2024kazemotts,
  title={KazEmoTTS: A Dataset for Kazakh Emotional Text-to-Speech Synthesis},
  author={Razakhan, Adal and Mussakhojayeva, Saida and Khassanov, Yerbolat},
  booktitle={Proc. LREC-COLING},
  year={2024}
}
```
'''

api = HfApi()
api.upload_file(
    path_or_fileobj=README.encode(),
    path_in_repo="README.md",
    repo_id="stukenov/kzcalm-mimi-codes-kk-v1",
    repo_type="dataset",
)
print("README uploaded")
