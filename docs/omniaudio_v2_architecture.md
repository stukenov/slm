# OmniAudio v2 — Technical Architecture

## Overview

OmniAudio v2 — нативная ASR модель для казахского языка, построенная полностью с нуля (без pretrained компонентов). Архитектура: audio encoder + linear projector + causal decoder. Все компоненты используют Llama-style блоки (RoPE, RMSNorm, SwiGLU).

**Параметры:** 69.58M total (266 MB checkpoint)
**Данные:** kzcalm-tts-kk-v1 (232K samples, 439 часов, казахская речь)
**Тренировка:** 2 стадии — CTC pretrain (encoder) → E2E (полная модель)

---

## 1. Audio Frontend

### Mel Spectrogram

```
Input:  waveform (16 kHz, mono, max 10 sec)
Output: mel spectrogram (80, T_frames)
```

| Параметр | Значение |
|----------|----------|
| sample_rate | 16,000 Hz |
| n_mels | 80 |
| n_fft | 400 (25 ms window) |
| hop_length | 160 (10 ms stride) |
| max_audio_len | 10.0 sec (= 1,000 frames) |

Mel вычисляется через `torchaudio.transforms.MelSpectrogram`, затем `log(clamp(mel, min=1e-10))`.

### Augmentation (только при тренировке)

**Speed Perturbation:**
- Случайный фактор из {0.9, 1.0, 1.1}
- Реализация через resampling: resample(waveform, sr*factor, sr)

**SpecAugment:**
- 2 frequency masks, F=27
- 2 time masks, T=100
- Значения занулённых областей = 0

---

## 2. Audio Encoder

```
Input:  mel (B, 80, T_frames)
Output: encoder features (B, T_enc, 256)
```

### 2.1 Conv Stack (Subsampling)

2 свёрточных слоя, каждый с stride=2 → общий subsampling 4x:

```
Conv1d(80 → 256, kernel=3, stride=2, padding=1) + GELU
Conv1d(256 → 256, kernel=3, stride=2, padding=1) + GELU
```

10 sec audio (1000 frames) → 250 encoder tokens.

### 2.2 RMSNorm

```python
RMSNorm(256)  # после conv, перед transformer
```

### 2.3 Transformer Encoder (6 layers)

Каждый EncoderBlock:

```
┌─────────────────────────────────────┐
│ x ──→ RMSNorm ──→ Self-Attention ──→ + x (residual)
│                    (bidirectional)
│ x ──→ RMSNorm ──→ SwiGLU FFN ──→ + x (residual)
└─────────────────────────────────────┘
```

**Self-Attention (bidirectional):**
- d_model = 256, n_heads = 4, head_dim = 64
- RoPE (Rotary Positional Embedding) на Q и K
- `F.scaled_dot_product_attention` (Flash Attention если доступен)
- Dropout = 0.1

**SwiGLU FFN:**
```
ffn_dim = round_to_64(256 * 8/3) = 704

gate = SiLU(gate_proj(x))    # Linear(256 → 704, no bias)
up   = up_proj(x)             # Linear(256 → 704, no bias)
out  = down_proj(gate * up)   # Linear(704 → 256, no bias)
```

**Encoder params:**
- Conv: 80*256*3 + 256*256*3 + biases ≈ 258K
- 6 Transformer blocks: 6 * (4*256*256 + 3*256*704) ≈ 4.8M
- **Total encoder: ~5.1M params**

---

## 3. Audio Projector

```
Input:  encoder features (B, T_enc, 256)
Output: decoder-space features (B, T_enc, 512)
```

```python
Linear(256 → 512) + RMSNorm(512)
```

**Projector params: ~131K**

---

## 4. CTC Head

```
Input:  encoder features (B, T_enc, 256)
Output: log probs (T_enc, B, 50257)
```

```python
Linear(256 → 50257)  # прямо из encoder dim
```

CTC loss с blank=0, zero_infinity=True.

**CTC head params: ~12.9M** (большая часть — матрица 256 → 50257)

---

## 5. Causal Decoder (from scratch)

```
Input:  [audio_embeds; text_embeds] (B, T_audio + T_text, 512)
Output: logits (B, T_audio + T_text, 50257)
```

### 5.1 Token Embedding

```python
Embedding(50257, 512)  # tied с lm_head
```

### 5.2 Transformer Decoder (8 layers)

Каждый DecoderBlock:

```
┌─────────────────────────────────────┐
│ x ──→ RMSNorm ──→ Self-Attention ──→ + x (residual)
│                    (CAUSAL, is_causal=True)
│ x ──→ RMSNorm ──→ SwiGLU FFN ──→ + x (residual)
└─────────────────────────────────────┘
```

**Causal Self-Attention:**
- d_model = 512, n_heads = 8, head_dim = 64
- RoPE на Q и K (тот же RotaryEmbedding, dim=64)
- `is_causal=True` в `scaled_dot_product_attention`
- Dropout = 0.1

**SwiGLU FFN:**
```
ffn_dim = round_to_64(512 * 8/3) = 1408

gate = SiLU(gate_proj(x))    # Linear(512 → 1408, no bias)
up   = up_proj(x)             # Linear(512 → 1408, no bias)
out  = down_proj(gate * up)   # Linear(1408 → 512, no bias)
```

### 5.3 Output

```python
RMSNorm(512)
Linear(512 → 50257, no bias)  # lm_head, tied с embed_tokens
```

**Decoder params:**
- Embedding: 50257 * 512 = 25.7M (shared with lm_head)
- 8 Transformer blocks: 8 * (4*512*512 + 3*512*1408) ≈ 25.6M
- RMSNorm: ~512
- **Total decoder: ~51.4M params**

---

## 6. Полный Parameter Count

| Компонент | Params |
|-----------|--------|
| Audio Encoder (Conv + 6 Transformer) | 5.1M |
| Audio Projector (Linear + Norm) | 131K |
| CTC Head | 12.9M |
| Token Embedding (shared) | 25.7M |
| Decoder (8 Transformer blocks) | 25.6M |
| Decoder Norm + LM Head (tied) | ~1K |
| **Total** | **69.58M** |

---

## 7. Training Pipeline

### Stage 1: CTC Pretrain

**Цель:** научить encoder извлекать фонетические фичи из аудио.

```
mel → Encoder → CTC Head → CTC Loss
```

| Параметр | Значение |
|----------|----------|
| Trainable | Encoder + CTC Head (18.0M) |
| Frozen | Всё остальное |
| Batch size | 8 * 4 grad_accum = 32 effective |
| LR | 1e-3, cosine decay |
| Warmup | 5% |
| Epochs | 5 |
| Augment | false |
| Steps/epoch | ~6,825 |
| Total steps | ~34,125 |

**Результат:** Val CTC loss: 0.4886

### Stage 2: E2E (End-to-End)

**Цель:** научить decoder генерировать текст из аудио.

```
mel → Encoder → Projector → [audio_embeds; text_embeds] → Decoder → Logits
                                                                    ↓
                              Hybrid Loss = 0.7 * CE + 0.3 * CTC
```

| Параметр | Значение |
|----------|----------|
| Trainable | ВСЕ параметры (69.58M) |
| init_from | checkpoint-best из Stage 1 |
| Batch size | 4 * 8 grad_accum = 32 effective |
| LR | 3e-4, cosine decay |
| Warmup | 5% |
| Epochs | 10 |
| CTC weight | 0.3 |
| Label smoothing | 0.1 |
| Augment | true (SpecAugment + speed perturb) |
| Steps/epoch | ~6,825 |
| Total steps | ~68,250 |

**CE Loss вычисление:**
```python
# Loss только на text позициях, audio позиции игнорируются
text_logits = logits[:, audio_len - 1 : -1]  # сдвиг на 1 (next-token prediction)
CE = cross_entropy(text_logits, text_ids, ignore_index=-100, label_smoothing=0.1)
```

**Текущий результат (эпоха 4/10):** Val loss: 2.82, train loss: 3.12

---

## 8. Inference (Generation)

```python
mel → Encoder → Projector → audio_embeds
                              ↓
                    [audio_embeds] → Decoder → next_token
                    [audio_embeds, tok1] → Decoder → next_token
                    [audio_embeds, tok1, tok2] → Decoder → next_token
                    ...до EOS или max_new_tokens
```

**Параметры генерации:**
- max_new_tokens: 200 (по умолчанию, конфигурируется через max_text_len)
- repetition_penalty: 1.2 (делит logits повторных токенов)
- Greedy decoding (argmax)
- Остановка по EOS token

**Без KV-cache:** на каждом шаге полный forward через decoder (медленнее, но проще). При 250 audio tokens + 100 text tokens = 350 seq length.

---

## 9. Data Pipeline

### Dataset: kzcalm-tts-kk-v1

| Split | Samples | % |
|-------|---------|---|
| Train | 218,409 | 94% |
| Val | 11,617 | 5% |
| Test | 2,320 | 1% |

**Total:** 232K samples, ~439 часов казахской речи (все спикеры, все эмоции).

### Tokenizer

- Путь: `./tokenizers/kazakh-gpt2-50k`
- Тип: ByteLevel BPE (GPT-2 style)
- Vocab size: 50,257
- EOS token: есть (используется для остановки генерации)

### Collator Flow

```
sample["audio"] → resample to 16kHz
                → speed perturbation (train only)
                → truncate to max_audio_len
                → MelSpectrogram + log
                → SpecAugment (train only)
                → pad to max_time in batch

sample["sentence"] → tokenize (BPE)
                   → append EOS token
                   → pad with -100 (CE ignore)
                   → CTC targets: без EOS, pad with 0
```

---

## 10. Файловая структура

```
omniaudio/
├── configs/
│   ├── v2_base.yaml                    # Базовый конфиг
│   ├── v2_scratch_kzcalm_s1.yaml       # CTC pretrain
│   └── v2_scratch_kzcalm_e2e.yaml      # E2E training
├── src/omniaudio/
│   ├── model_v2.py                     # Все модели и слои
│   │   ├── RotaryEmbedding             # RoPE
│   │   ├── RMSNorm                     # Llama-style norm
│   │   ├── EncoderBlock                # Bidirectional transformer
│   │   ├── AudioEncoderV2              # Conv + Transformer encoder
│   │   ├── AudioProjectorV2            # Linear projection
│   │   ├── DecoderBlock                # Causal transformer
│   │   ├── OmniAudioScratchModel       # Full model (encoder+decoder)
│   │   └── OmniAudioV2Model            # Pretrained LLM variant (deprecated)
│   ├── data_v2.py                      # Dataset loading + AudioCollatorV2
│   ├── train_v2.py                     # Training loop
│   ├── evaluate_v2.py                  # WER/CER evaluation
│   └── augment.py                      # SpecAugment + speed perturbation
└── outputs/
    ├── omniaudio_v2_scratch_kzcalm_s1_ctc/   # CTC checkpoints
    └── omniaudio_v2_scratch_kzcalm_e2e/      # E2E checkpoints
```

---

## 11. Архитектурная диаграмма

```
                        ┌──────────────────────┐
                        │   Audio Waveform      │
                        │   (16kHz, max 10s)    │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   Mel Spectrogram     │
                        │   (80 bins, T frames) │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    Conv Subsampling (4x)     │
                    │  Conv1d(80→256, s=2) + GELU  │
                    │  Conv1d(256→256, s=2) + GELU │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Transformer Encoder        │
                    │   6 layers, 256d, 4 heads    │
                    │   Bidirectional + RoPE        │
                    │   SwiGLU FFN (704d)           │
                    └──────┬───────────┬──────────┘
                           │           │
              ┌────────────▼──┐   ┌────▼────────────┐
              │  CTC Head     │   │  Projector       │
              │  256→50257    │   │  256→512 + Norm   │
              │  (Stage 1)    │   │                   │
              └───────────────┘   └────────┬─────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  Concat: [audio; text]   │
                              │  audio: (B, ~250, 512)   │
                              │  text:  (B, T_txt, 512)  │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  Transformer Decoder     │
                              │  8 layers, 512d, 8 heads │
                              │  CAUSAL + RoPE            │
                              │  SwiGLU FFN (1408d)       │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  LM Head (tied embed)    │
                              │  512 → 50257             │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  Output: text tokens     │
                              │  (Kazakh BPE)            │
                              └─────────────────────────┘
```

---

## 12. Training Progress (2026-03-29)

| Metric | CTC Pretrain | E2E (current) |
|--------|-------------|---------------|
| Status | Completed | Epoch 4/10 (~35%) |
| Train loss | 0.695 | 3.12 |
| Val loss | 0.489 | 2.82 |
| Duration | ~2.5h | ~20h total (ETA ~4h) |
| Hardware | NVIDIA A10 (GPU 1) | NVIDIA A10 (GPU 1) |

### Inference Quality (checkpoint at epoch 3)

Модель уже **правильно распознаёт** начала предложений:

| REF | HYP (начало) |
|-----|-------------|
| Іс материалдары бойынша бұл тұлғаның мас болғандығы анықталды. | Іс материалдары бойынша бұл тұлғаның мас болғандығы анықталды. (100%) |
| Мен Қарағанды облысының шағын қалаларының бірінде тұрамын. | Мен Қарағанды облысының шағын қалаларының бірінде тұрамын. (100%) |

**Известная проблема:** модель не останавливается на EOS (генерирует мусор после правильного текста). Фикс: EOS token добавлен в training targets (будет применён в следующем раунде).
