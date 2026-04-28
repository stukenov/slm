# exp034: EkiTil-Translate — kk↔ru Translation Models

## Goal
Fine-tune EkiTil base models (123M/300M/600M) for bidirectional Kazakh↔Russian translation.
Compare against deepvk/kazRush (T5, 197M) and NLLB on FLORES+ devtest.

## Data

### Parallel Dataset: `stukenov/ekitil-parallel-kkru-v2`

| Source | kk-ru Pairs | Notes |
|--------|-------------|-------|
| KazParC (issai/kazparc) | 366,587 | Human-translated, 5 domains |
| OPUS-XLEnt | 75,989 | Cross-lingual entities |
| OPUS-KDE4 | 54,173 | Software localization |
| OPUS-wikimedia | 43,501 | Wikipedia |
| OPUS-WikiMatrix | 32,786 | Mined parallel sentences |
| OPUS-TED2020 | 7,047 | TED talks |
| OPUS-QED | 4,072 | Educational videos |
| OPUS-GNOME | 3,522 | Software localization |
| OPUS-OpenSubtitles | 2,091 | Movie subtitles |
| OPUS-NeuLab | 1,054 | TED talks |
| OPUS-Ubuntu | 137 | Software localization |
| **Total** | **590,959** | After dedup |

Also: 126,149 kk-en pairs (WMT19) in separate config.

### Translation Format
```
<|kk|> Қазақша мәтін <|translate|> <|ru|> Русский перевод
<|ru|> Русский текст <|translate|> <|kk|> Қазақша аударма
```

Labels mask the source side — model only predicts the target translation.

## Training Plan

### Stage 1: Continue Pretrain (Full Fine-tune)
- All 3 EkiTil models: 123M, 300M, 600M
- LR: 2e-5 (10x lower than pretrain), cosine decay
- Epochs: 3
- Max seq len: 512
- Both directions (kk→ru + ru→kk) = ~1.18M training examples

### Stage 2 (optional): SFT with LoRA
- If Stage 1 quality is insufficient
- LoRA r=32, alpha=64
- Only on high-quality pairs (KazParC subset)

## Evaluation
- FLORES+ devtest (1012 sentences per language)
- Metrics: BLEU, chrF (sacrebleu)
- Baselines: deepvk/kazRush-kk-ru, deepvk/kazRush-ru-kk, NLLB-200

## Files
| File | Purpose |
|------|---------|
| `prepare_parallel_data.py` | Collect, clean, dedup parallel data → HF |
| `train_translate.py` | Continue-pretrain training script |
| `evaluate_flores.py` | FLORES+ evaluation (BLEU, chrF) |

## Commands

```bash
# Prepare data (already done)
python3 prepare_parallel_data.py --upload

# Train 123M (fastest, for testing)
python3 train_translate.py \
  --base stukenov/ekitil-core-qwen3-123m-kkru-base-v1 \
  --epochs 3 --batch-size 32 --lr 2e-5

# Train 600M with LoRA
python3 train_translate.py \
  --base stukenov/ekitil-core-qwen3-600m-kkru-base-v1 \
  --lora --lora-r 32 --epochs 3 --batch-size 8

# Evaluate
python3 evaluate_flores.py --model stukenov/ekitil-core-qwen3-123m-kkru-translate-v1
python3 evaluate_flores.py --model stukenov/ekitil-core-qwen3-123m-kkru-translate-v1 --direction ru-kk
```

## HF Targets
| Model | HF Repo |
|-------|---------|
| 123M translate | `stukenov/ekitil-core-qwen3-123m-kkru-translate-v1` |
| 300M translate | `stukenov/ekitil-core-qwen3-300m-kkru-translate-v1` |
| 600M translate | `stukenov/ekitil-core-qwen3-600m-kkru-translate-v1` |
