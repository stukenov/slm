# KZ-CALM — Kazakh Consistency-Latent TTS

Decode-only TTS для казахского языка на основе нейрокодека Mimi и flow/consistency matching.

## Архитектура

```
Text → Normalizer → SentencePiece → Embeddings ─┐
                                                  ├→ Transformer Backbone → Flow/Consistency Head → Latents → Mimi Decoder → Waveform
Voice Prompt (P1) → Mimi Encoder → Latents ──────┘
```

| Компонент | Детали |
|-----------|--------|
| Кодек | Mimi (frozen), 24kHz |
| Токенизер | SentencePiece 4-8k, казахский |
| Backbone | Transformer 12-24L, d=1024-1536 |
| Flow Head | Consistency/flow matching, 4-8 steps |

## Фазы

- **Stage 0**: Sanity check на малом датасете
- **Stage 1** (P0): Flow matching, 8-step sampling
- **Stage 2** (P1): Self-distillation → 4-step
- **Stage 3** (P1): Voice prompt conditioning

## Запуск

```bash
cd kz-calm
uv venv && uv pip install -e .

# Подготовка данных
python -m kzcalm.scripts.prepare_data --config configs/base.yaml

# Обучение
python -m kzcalm.train --config configs/experiments/exp001_sanity.yaml

# Инференс
python -m kzcalm.inference --model_path outputs/exp001/ --text "Сәлем, әлем!"

# Оценка
python -m kzcalm.evaluate --model_path outputs/exp001/ --test_set eval/golden_phrases.txt
```
