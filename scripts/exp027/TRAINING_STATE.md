# EkiTil Training State — Context Handoff

**Last updated**: 2026-04-06 ~10:00 UTC+5

## Current Training: 600M on 8×H100

| Field | Value |
|-------|-------|
| Pod ID | `m6ycq762my7jw2` |
| SSH | `ssh -o StrictHostKeyChecking=no root@103.207.149.70 -p 14877` |
| GPU | 8× NVIDIA H100 80GB HBM3 |
| Model | Qwen3 673.8M params |
| Script | `/workspace/train_ekitil.py --size 600m --batch-size 12 --grad-accum 8 --no-compile` |
| Screen | `screen -r ekitil` |
| Log | `/workspace/train.log` |
| Checkpoints (local) | `/root/checkpoints/step_*` |
| Checkpoints (HF) | `stukenov/ekitil-core-qwen3-600m-kkru-checkpoints` (step_500, step_589, step_1000 uploaded) |
| HF final model | `stukenov/ekitil-core-qwen3-600m-kkru-base-v1` |
| Data | `/root/cache/train.bin` (4.4GB, pre-cached) |

### Training Config
- **Batch**: 12 per GPU × 8 grad_accum × 8 GPUs = **1,572,096 tok/step**
- **Total steps**: 7,853
- **Epochs**: 5 (5 × 2.47B = 12.35B tokens)
- **LR**: 2e-4, cosine decay, 2000 warmup steps
- **Save every**: 500 steps (each uploaded to HF with model + optimizer + meta)

### Progress at Handoff
- **Step ~1,300 / 7,853** (16.6%)
- **Loss**: 5.50
- **BPB**: 7.93
- **Throughput**: 395K tok/s
- **Epoch**: 0.83
- **ETA**: ~7.2 hours
- **VRAM**: ~80.5GB / 81.5GB per GPU (maxed out)

### Loss Trend
```
Step      Loss     Epoch
  100     10.77    0.06   (warmup)
  250      9.32    0.16
  400      8.18    0.25
  550      7.42    0.35
  700      6.85    0.45
  850      6.44    0.54
1,000      6.07    0.64   (checkpoint on HF)
1,150      5.75    0.73
1,300      5.50    0.83
1,450      5.26    0.92
```

### Cron Monitor
- **Cron ID**: `d841d100` (every 10 min, session-only — dies with Claude session)
- After context reset: **recreate the cron** using CronCreate

---

## Completed Models

### EkiTil-123M ✅
| Field | Value |
|-------|-------|
| HF Model | `stukenov/ekitil-core-qwen3-123m-kkru-base-v1` |
| Params | 124.7M (768d/12L/12h/4kv) |
| Final loss | 3.0748 |
| Final BPB | 4.436 |
| Tokens | 2.47B (1 epoch) |
| Time | 3.8h on 1×H100 |
| Cost | ~$10 |
| README | Uploaded with inference examples |

### EkiTil-300M ✅
| Field | Value |
|-------|-------|
| HF Model | `stukenov/ekitil-core-qwen3-300m-kkru-base-v1` |
| HF Checkpoints | `stukenov/ekitil-core-qwen3-300m-kkru-checkpoints` |
| Params | 245.9M (1024d/16L/16h/4kv) |
| Final loss | 2.925 |
| Final BPB | 4.220 |
| Tokens | 4.94B (2 epochs) |
| Time | 6.63h on 2×H100 |
| Cost | ~$30 |
| README | Uploaded with inference examples |

---

## Key Files
| File | Purpose |
|------|---------|
| `scripts/exp027/train_ekitil.py` | Unified training script (123m/300m/600m) |
| `scripts/exp027/train_ekitil_123m.py` | Legacy 123M-only script |
| `scripts/exp027/pod_info.json` | Current pod state for cron monitor |
| `scripts/exp027/inference_demo.py` | Inference examples script |
| `scripts/exp027/readme_123m.md` | 123M HF model card |
| `scripts/exp027/readme_300m.md` | 300M HF model card |
| `EKITIL_WHITEPAPER.md` | Full whitepaper (updated with 123M + 300M results) |

## HF Artifacts
| Repo | Type | Status |
|------|------|--------|
| `stukenov/ekitil-corpus-annotated-kk-v1` | Dataset | Published |
| `stukenov/ekitil-corpus-parallel-kkru-v1` | Dataset | Published |
| `stukenov/ekitil-vocab-bpe-64k-kkru-v1` | Tokenizer | Published |
| `stukenov/ekitil-corpus-tokenized-kkru-v1` | Dataset | Published (v1 sentence-level) |
| `stukenov/ekitil-core-qwen3-123m-kkru-base-v1` | Model | Published |
| `stukenov/ekitil-core-qwen3-300m-kkru-base-v1` | Model | Published |
| `stukenov/ekitil-core-qwen3-300m-kkru-checkpoints` | Checkpoints | Published |
| `stukenov/ekitil-core-qwen3-600m-kkru-checkpoints` | Checkpoints | Publishing (step_500, step_1000) |
| `stukenov/ekitil-core-qwen3-600m-kkru-base-v1` | Model | Training |

## RunPod API
- Config: `~/.runpod/config.json`
- Current pod: `m6ycq762my7jw2`
- **NEVER destroy without verifying model on HF first**

## What Happens When 600M Finishes
1. Script auto-uploads final model to `stukenov/ekitil-core-qwen3-600m-kkru-base-v1`
2. Prints `TRAINING_DONE` in log
3. Cron detects → verifies on HF → destroys pod → updates whitepaper
4. Need to: create README for 600M, run inference demo, update whitepaper
