# OmniAudio — Казахский ASR Decode-Only Микромодель

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Обучить с нуля decode-only омнимодель (~68M params) для распознавания казахской речи, по архитектуре Qwen-Audio (audio encoder + projector + LLM decoder).

**Architecture:** Audio Encoder (CNN + Transformer, ~18M) извлекает аудио-эмбеддинги из mel-спектрограмм. Linear Projector (~0.2M) проецирует их в embedding space LLM. Decode-only LLM (~50M, Llama-архитектура) авторегрессивно генерирует текст транскрипции. Обучение на Common Voice kk (~200ч).

**Tech Stack:** PyTorch, transformers, datasets, torchaudio, librosa, vast.ai (1x RTX 4090)

**Infra:** Существующий vast.ai cloud pipeline (`slm.cloud`) с кастомным `--train-module omniaudio.train`

---

## Сводка параметров модели

| Компонент | Params | Детали |
|-----------|--------|--------|
| Audio Encoder | ~18M | 2 Conv1d + 4 Transformer layers (384d, 6h) |
| Projector | ~0.2M | Linear 384 to 576 + LayerNorm |
| LLM Decoder | ~50M | 8 layers, 576d, 8h, SwiGLU (1536 intermediate) |
| **Итого** | **~68M** | |

## План обучения на vast.ai (1x RTX 4090)

| Этап | Что обучаем | Trainable | Время | Стоимость |
|------|------------|-----------|-------|-----------|
| Stage 1: Alignment | Projector only | ~0.2M | ~2-3ч | ~$1.5 |
| Stage 2: Finetune | Всё | ~68M | ~6-8ч | ~$4-5 |
| **Итого** | | | **~10ч** | **~$6** |

## Датасет

- **Train:** Common Voice kk train split (~200ч аудио)
- **Val:** Common Voice kk validation
- **Test:** Common Voice kk test + FLEURS kk (benchmark)

---

### Task 1: Создать структуру подпроекта omniaudio

**Files:**
- Create: `omniaudio/README.md`
- Create: `omniaudio/pyproject.toml`
- Create: `omniaudio/src/omniaudio/__init__.py`

**Step 1: Создать директории**

```bash
mkdir -p omniaudio/src/omniaudio omniaudio/configs omniaudio/tests
```

**Step 2: Создать pyproject.toml**

```toml
[project]
name = "omniaudio"
version = "0.1.0"
description = "Kazakh ASR omni-model: decode-only audio-to-text"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    "torch>=2.1",
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.28",
    "torchaudio>=2.1",
    "librosa>=0.10",
    "soundfile>=0.12",
    "huggingface-hub>=0.22",
    "pyyaml>=6.0",
    "tensorboard>=2.16",
    "jiwer>=3.0",
]
```

**Step 3: Commit**

---

### Task 2: Реализовать AudioEncoder + Projector + OmniAudioModel

**Files:**
- Create: `omniaudio/src/omniaudio/model.py`
- Create: `omniaudio/tests/test_model.py`

Архитектура:
- `AudioEncoder`: 2 Conv1d (stride=2, 4x downsampling) + positional encoding + 4 Transformer encoder слоёв
- `AudioProjector`: Linear(384 -> 576) + LayerNorm
- `OmniAudioModel`: Encoder -> Projector -> LLM decoder (custom Llama with RMSNorm + SwiGLU)
- Forward: concat [audio_embeds, text_embeds] -> causal LLM -> loss only on text positions
- Generate: autoregressive decoding from audio embeddings

**Tests:**
- test_audio_encoder_output_shape
- test_projector
- test_omni_audio_forward (loss is scalar > 0)

---

### Task 3: Реализовать data pipeline (Common Voice kk)

**Files:**
- Create: `omniaudio/src/omniaudio/data.py`
- Create: `omniaudio/tests/test_data.py`

Components:
- `load_commonvoice_kk(split, max_samples)`: загрузка mozilla-foundation/common_voice_17_0 kk
- `AudioCollator`: waveform -> resample to 16kHz -> mel spectrogram -> log mel; text -> tokenize with kazakh-bpe-32k; pad batch

---

### Task 4: Реализовать train.py (2-stage training)

**Files:**
- Create: `omniaudio/src/omniaudio/train.py`
- Create: `omniaudio/configs/base.yaml`
- Create: `omniaudio/configs/stage1_alignment.yaml`
- Create: `omniaudio/configs/stage2_finetune.yaml`

Training loop:
- Custom PyTorch loop (не HF Trainer, т.к. нестандартный forward)
- Config inheritance (как в основном проекте)
- Stage 1: freeze encoder + LLM, train projector only, lr=1e-3, 5 epochs
- Stage 2: unfreeze all, lr=2e-5, 10 epochs, init from stage1
- bf16, gradient accumulation, grad clipping, linear warmup scheduler

---

### Task 5: Интеграция с vast.ai cloud pipeline

**Files:**
- Create: `omniaudio/configs/cloud_stage1.yaml`
- Create: `omniaudio/configs/cloud_stage2.yaml`

Launch commands (1x RTX 4090, ~$0.40-0.60/hr):

```bash
# Stage 1
PYTHONPATH=src python -m slm.cloud launch \
    --config omniaudio/configs/cloud_stage1.yaml \
    --hf-repo saken-tukenov/omniaudio-kk-stage1 \
    --gpu RTX_4090 --max-price 0.60 --disk 80 \
    --train-module omniaudio.train --monitor

# Stage 2
PYTHONPATH=src python -m slm.cloud launch \
    --config omniaudio/configs/cloud_stage2.yaml \
    --hf-repo saken-tukenov/omniaudio-kk-v1 \
    --gpu RTX_4090 --max-price 0.60 --disk 80 \
    --train-module omniaudio.train --monitor
```

---

### Task 6: WER/CER Evaluation

**Files:**
- Create: `omniaudio/src/omniaudio/evaluate.py`

- Load model + tokenizer
- Run autoregressive generation on Common Voice kk test split
- Compute WER/CER with jiwer
- Print results table
