# OmniAudio

Decode-only омнимодель (~68M params) для распознавания казахской речи.

## Архитектура

Qwen-Audio стиль: Audio Encoder (CNN + Transformer) → Projector → LLM Decoder (Llama).

| Компонент | Params | Детали |
|-----------|--------|--------|
| Audio Encoder | ~18M | 2 Conv1d + 4 Transformer layers (384d, 6h) |
| Projector | ~0.2M | Linear 384→576 + LayerNorm |
| LLM Decoder | ~50M | 8 layers, 576d, 8h, SwiGLU |

## Обучение

2 стадии на vast.ai (1x RTX 4090):

1. **Alignment** — обучаем только projector (freeze encoder + LLM)
2. **Finetune** — обучаем всё end-to-end

## Датасет

Common Voice kk (~200ч казахской речи)

## Запуск

```bash
# Stage 1
python -m slm.cloud launch --config omniaudio/configs/cloud_stage1.yaml \
    --hf-repo saken-tukenov/omniaudio-kk-stage1 --gpu RTX_4090 --monitor

# Stage 2
python -m slm.cloud launch --config omniaudio/configs/cloud_stage2.yaml \
    --hf-repo saken-tukenov/omniaudio-kk-v1 --gpu RTX_4090 --monitor
```
