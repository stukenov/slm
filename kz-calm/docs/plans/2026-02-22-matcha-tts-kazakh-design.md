# Design: Matcha-TTS для казахского языка (exp003)

**Дата:** 2026-02-22
**Сервер:** kaznu (2x A10, 23GB each)
**Датасет:** `stukenov/kzcalm-tts-kk-v1` (232K samples, 439h, 24kHz)
**Vocoder:** Vocos `charactr/vocos-mel-24khz` (100 mel bins)

---

## Контекст

exp001 (Mimi 512-dim) и exp002 (mel 100-dim) провалились — loss застревает, речь неразборчива.
Корневая причина: нет explicit alignment между текстом и mel.
Matcha-TTS решает это через MAS + duration predictor.

## Архитектура

```
text_ids → TextEncoder (4 transformer layers, d=256)
              ↓
         μ, log_σ (per-phoneme mel statistics)
              ↓
    ┌─── Training: MAS(μ, mel) → hard alignment → durations (ground truth)
    │    Duration Predictor(encoder_out) → predicted durations → duration_loss
    │
    └─── Inference: Duration Predictor → predicted durations
              ↓
         μ_expanded: repeat μ per-frame using durations → (B, T_mel, d)
              ↓
         Flow Matching 1D U-Net:
           x_t = (1-t) * noise + t * mel
           U-Net(x_t, t, μ_expanded) → velocity v
           flow_loss = ||v - (mel - noise)||
              ↓
         Euler ODE (8 steps): noise → mel
              ↓
         Vocos → waveform (24kHz)
```

### Loss

```
total_loss = flow_loss + λ * duration_loss
```
- `flow_loss`: MSE/Huber на velocity (как в exp002)
- `duration_loss`: MSE на log-duration
- `λ = 1.0`

## Компоненты

### 1. Text Encoder (`src/kzcalm/model/text_encoder.py`)
- Embedding(vocab_size, d=256) + sinusoidal pos
- 4 transformer encoder layers (d=256, 4 heads, d_ff=1024)
- Linear projections → μ (B, S, 100), log_σ (B, S, 100)
- ~5M params

### 2. Monotonic Alignment Search (`src/kzcalm/model/mas.py`)
- Viterbi-style dynamic programming
- Input: μ (B, S, D) text features, mel (B, T, D) target
- Output: hard alignment (B, T) → index of text token for each mel frame
- Извлекаем durations: count frames per text token
- Python-only (no Cython), vectorized where possible

### 3. Duration Predictor (`src/kzcalm/model/duration_predictor.py`)
- 2x Conv1d(256, 256, kernel=3) + ReLU + LayerNorm + Dropout
- Linear(256, 1) → log-duration per phoneme
- ~0.5M params

### 4. 1D U-Net Decoder (`src/kzcalm/model/unet_1d.py`)
- Input: x_t (B, 100, T) + μ_expanded (B, 100, T) → concat → (B, 200, T)
- Timestep conditioning: sinusoidal → MLP → scale/shift (FiLM)
- Architecture: 3 downsample + 3 upsample blocks
  - Channels: 256 → 256 → 512 → 512
  - ResBlock: Conv1d + GroupNorm + SiLU + FiLM conditioning
  - Downsample: Conv1d(stride=2), Upsample: ConvTranspose1d(stride=2)
  - Skip connections between encoder/decoder
- Output: (B, 100, T) velocity
- ~10M params

### 5. MatchaTTS wrapper (`src/kzcalm/model/matcha.py`)
- Combines: TextEncoder + DurationPredictor + MAS + U-Net
- `forward(text, mel, mel_mask)` → flow_loss, duration_loss
- `synthesize(text)` → mel (using predicted durations + ODE)

### 6. Training (`src/kzcalm/train_matcha.py`)
- Новый train script (не модифицируем старый train.py)
- DDP: `torchrun --nproc_per_node=2`
- MelDataset (существующий) + mel_collate_fn
- bf16 autocast
- AdamW, lr=1e-4, warmup=4000, cosine decay
- Checkpoint: model + optimizer + step + config

## Mel Configuration

- n_mels=100 (Vocos requirement)
- n_fft=1024, hop_length=256, sample_rate=24000
- Normalization: mean=-1.42, std=3.80 (computed on dataset)
- Denorm at inference: mel = pred * 3.80 + (-1.42)

## Model Size

| Component | Params |
|-----------|--------|
| Text Encoder | ~5M |
| Duration Predictor | ~0.5M |
| 1D U-Net | ~10M |
| **Total** | **~17M** |

## DDP Setup

- `torchrun --nproc_per_node=2 -m kzcalm.train_matcha --config configs/experiments/exp003_matcha.yaml`
- DistributedDataParallel wrapper
- Streaming IterableDataset: каждый worker получает свой shard
- Gradient sync автоматический через DDP
- Effective batch = batch_per_gpu * 2 GPUs * grad_accum

## Experiment Stages

### Stage 0: Overfit test (50 samples)
- `dataset_subset: 50`
- batch_size=8, 1000 steps
- **Критерий:** flow_loss → ~0, duration_loss → ~0
- **Время:** ~30 минут
- **Если не сходится:** баг в коде

### Stage 1: Small subset (1% = ~2.3K samples)
- `split: "train[:1%]"`
- batch_size=32, 10K steps
- **Критерий:** разборчивые слова при генерации
- **Время:** ~2-3 часа

### Stage 2: Full dataset (232K samples)
- batch_size=32 per GPU (effective 64 с DDP)
- 200K steps
- **Критерий:** разборчивая казахская речь, MOS > 3.0
- **Время:** ~24-48 часов

## Config: `configs/experiments/exp003_matcha.yaml`

```yaml
inherits: base
experiment_name: exp003_matcha

data:
  hf_audio_dataset: stukenov/kzcalm-tts-kk-v1
  split: "train"

codec:
  type: mel
  latent_dim: 100
  n_mels: 100
  hop_length: 256
  n_fft: 1024

tokenizer:
  hf_repo: "stukenov/kzcalm-sp-tokenizer-4k-kk-v1"

model:
  type: matcha
  encoder_layers: 4
  encoder_dim: 256
  encoder_heads: 4
  encoder_ff: 1024
  unet_channels: [256, 256, 512, 512]
  duration_predictor_layers: 2
  dropout: 0.1
  max_text_len: 512

flow:
  num_sampling_steps: 8
  sigma_min: 0.001
  loss_type: mse

training:
  batch_size: 32
  gradient_accumulation_steps: 1
  max_steps: 200000
  learning_rate: 1.0e-4
  warmup_steps: 4000
  weight_decay: 0.01
  grad_clip: 1.0
  bf16: true
  num_workers: 4
  save_steps: 5000
  logging_steps: 100
  duration_loss_weight: 1.0
```

## Файловая структура (новые файлы)

```
src/kzcalm/model/
  text_encoder.py       — NEW: TextEncoder (transformer + μ/σ projection)
  mas.py                — NEW: Monotonic Alignment Search
  duration_predictor.py — NEW: Duration Predictor (conv stack)
  unet_1d.py            — NEW: 1D U-Net flow decoder
  matcha.py             — NEW: MatchaTTS (orchestrator)

src/kzcalm/
  train_matcha.py       — NEW: DDP training loop

configs/experiments/
  exp003_matcha.yaml    — NEW: experiment config
```

## Что переиспользуем

- `src/kzcalm/codec/mel.py` — MelExtractor (без изменений)
- `src/kzcalm/data/dataset.py` — MelDataset, mel_collate_fn (без изменений)
- `src/kzcalm/model/flow_head.py` — FlowMatchingLoss, sample_euler (без изменений)
- `src/kzcalm/tokenizer/sp_tokenizer.py` — KazakhTokenizer (без изменений)
- `src/kzcalm/inference.py` — адаптируем для Matcha
- Vocos vocoder (без изменений)
