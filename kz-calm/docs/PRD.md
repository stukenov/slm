# PRD: Kazakh Consistency-Latent TTS (KZ-CALM)

Дата: 19 февраля 2026

Основа: нейрокодек Mimi → латентный TTS на основе flow/consistency (few-step sampling) → декодер кодека в waveform. Референсная реализация по идеям pocket-tts от Kyutai Labs.

## 1. Контекст

Задача: получить адекватное по качеству TTS только для казахского языка. Ключевой риск: качество TTS определяется данными (чистота, соответствие текст↔аудио, разнообразие дикторов).

## 2. Метрики успеха

- MOS (естественность) >= порога
- ASR-based CER/WER на синтезированном аудио
- Доля клипов с артефактами (ручная разметка)
- RTF на GPU, TTFA для streaming

## 3. Архитектура

```
Text → Normalizer → SentencePiece → Embeddings ─┐
                                                  ├→ Transformer Backbone → Flow/Consistency → Latents → Mimi Decoder → Waveform
Voice Prompt (P1) → Mimi Encoder → Latents ──────┘
```

- Кодек: Mimi (frozen)
- Токенизер: SentencePiece 4-8k
- Backbone: 12-24L, d=1024-1536
- Flow head: 8 steps (P0), 4 steps (P1)

## 4. Фазы

- **A**: Data readiness (QC, 50-100ч clean)
- **B**: Baseline model (train loop, стабильная генерация)
- **C**: Quality hardening (домены, числа, просодика, MOS)
- **D**: Productization (API, мониторинг)
- **E**: Voice prompt + streaming (P1)
