# exp028: Llama 1.08B Kazakh Language Model

## Overview

**Goal**: Scale up from exp023 (600M) to create the largest dense Kazakh-only language model.

| Property | Value |
|----------|-------|
| Model | Custom Llama 1.08B |
| HuggingFace | [`stukenov/sozkz-core-llama-1b-kk-base-v1`](https://huggingface.co/stukenov/sozkz-core-llama-1b-kk-base-v1) |
| Training data | ~16.2B tokens (70% KK / 30% ENKK) |
| Hardware | 8×H100 SXM 80GB (RunPod) |
| Training time | 9.53 hours |
| Cost | ~$205 |
| Final loss | 2.15 |
| Val BPB | 0.8127 |
| Date | 2026-03-25 |
| Status | **Degraded** — QK-Norm weights lost during HF conversion |

## Architecture

```
Hidden size:       2048
Layers:            22
Attention heads:   16
KV heads:          4 (Grouped Query Attention)
Intermediate:      5504 (SwiGLU, ~2.69× hidden)
Vocab size:        50,257
Max seq length:    1024
Tied embeddings:   yes
Parameters:        ~1.08B (reduced from 1.22B due to GQA KV savings)
```

### Quality Improvements (vs exp023 600M)

| Feature | Description | HF Compatible? |
|---------|-------------|:-:|
| **GQA** (4 KV heads) | Reduces KV cache 4×, faster inference | Yes |
| **QK-Norm** (RMSNorm on Q,K) | Stabilizes attention, prevents logit growth | **No** — weights silently dropped |
| **Z-loss** (1e-4) | Logit regularization from PaLM/Gemma | Yes (training only) |
| **Embedding scaling** (×sqrt(2048)) | Gemma trick, scales embeddings up | **No** — code-only, not in weights |

## Data Pipeline

### Cleaning

Two datasets cleaned with `scripts/data/filter_foreign_chars.py`:

| Dataset | Original | After cleaning | Keep % |
|---------|----------|----------------|--------|
| KK (sozkz-corpus-clean-kk-text-v4) | ~13.2M texts | ~12.3M | 93.0% |
| ENKK (fineweb-edu-v2) | ~8.8M texts | ~7.8M | 88.4% |

Cleaning filters:
- CJK characters and other foreign scripts
- Encoding artifacts (Cyrillic mojibake patterns like `Ñ€`)
- High Latin ratio (>5% of characters)
- Repetitive garbage (repeated char ratio >15%)
- Unicode replacement characters (U+FFFD)

**Bug found and fixed**: Unicode regex `\u20000` was parsed as `\u2000` + literal `0`, creating a character range that matched ALL Cyrillic. This would have deleted the entire dataset.

### Tokenization

- Tokenizer: `stukenov/sozkz-core-gpt2-50k-kk-base-v1` (ByteLevel BPE, 50K vocab)
- Pre-tokenized on kaznu server, uploaded to HuggingFace:
  - `stukenov/sozkz-corpus-tokenized-kk-llama50k-v4` (~8.4B tokens)
  - `stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v2` (~7.8B tokens)
- Mix ratio: 70% KK / 30% ENKK
- Total: ~16.2B tokens
- Chinchilla ratio: 16.2B / 1.08B = **15:1** (near-optimal ~20:1)

## Training

### Configuration

```
Optimizer:         AdamW (lr=2e-4, weight_decay=0.1)
Warmup:            1000 steps
Schedule:          Cosine decay
Batch size:        16 per GPU × 2 grad_accum × 8 GPUs = 2,097,152 tokens/step
Total steps:       ~7,700
Precision:         bfloat16 (mixed precision)
torch.compile:     yes
```

### Pipeline (`autoresearch/run_1b_training.sh`)

Fully autonomous with Telegram notifications:
1. Environment verification (8 GPUs, disk, tokens)
2. Dependency installation
3. Data download (`prepare_1b.py` — 70/30 mix)
4. Speed benchmark (50 steps → tok/s, ETA, cost estimate)
5. Smoke test (verify loss decreasing)
6. Full training (`torchrun --nproc_per_node=8`)
7. Upload to HuggingFace
8. Self-destruct RunPod pod (only if upload verified)

### Results

| Metric | Value |
|--------|-------|
| Final training loss | 2.15 |
| Validation BPB | 0.8127 |
| Throughput | ~577K tok/s |
| Total training time | 9.53 hours |

## Critical Failure: HF Conversion

### What happened

After training completed successfully, the model was uploaded to HuggingFace using `upload_1b_to_hf.py`. The upload script used `model.load_state_dict(state, strict=False)` which **silently dropped** all QK-Norm weights because HF's `LlamaForCausalLM` does not have QK-Norm parameters.

Additionally, embedding scaling (`x * sqrt(2048)`) was applied in code only — the scaling factor was not baked into the embedding weights before upload.

The RunPod pod was destroyed after upload verification (HTTP 200), losing the original checkpoint forever.

### Impact

The model on HuggingFace is **incomplete**:
- Missing QK-Norm: attention operates without the normalization it was trained with
- Missing embedding scaling: embeddings are ~45× smaller than expected
- Result: model generates Kazakh text but with significant quality degradation

### Recovery Attempts

| Attempt | Method | Result |
|---------|--------|--------|
| 1. Embedding fix | Multiplied `embed_tokens.weight` by sqrt(2048), re-uploaded | Partial improvement — Kazakh text appears but with artifacts |
| 2. QK-Norm weight=ones | Applied RMSNorm with all-ones weights (identity approximation) | Model collapsed — output gibberish |
| 3. Fine-tune QK-Norm | Trained only 5,632 QK-Norm params on small data | Loss stuck at ~60, did not converge |

**Conclusion**: QK-Norm weights cannot be recovered. They were co-trained with all other weights over 9.5 hours and encode learned attention patterns. An identity initialization is not a valid approximation.

### Lessons Learned

Added to `CLAUDE.md` as critical rules:

1. **NEVER add custom architecture modifications without testing the full HF round-trip** (train → save → convert → load → inference → verify)
2. **NEVER destroy cloud instances before downloading/verifying checkpoints** — pod disk is ephemeral
3. **Always use `strict=True`** when loading state dicts to catch missing/unexpected keys
4. **Bake all scaling factors into weights** before upload, not in code

## Current State

The model at `stukenov/sozkz-core-llama-1b-kk-base-v1` works but with degraded quality:
- Embedding scaling has been fixed (baked into weights)
- QK-Norm is permanently lost
- GQA and Z-loss are intact (HF-compatible)

### To retrain (~$205)

If budget becomes available:
1. Remove QK-Norm and embedding scaling from `train_1b_ddp.py`
2. Keep GQA (4 KV heads) + Z-loss (both HF-compatible)
3. **Do full round-trip smoke test** before expensive training
4. **Keep pod alive** until model verified from HF Hub

## Files

| File | Purpose |
|------|---------|
| `configs/experiments/exp028_llama_1b.yaml` | Experiment config |
| `autoresearch/train_1b_ddp.py` | Training script (custom Llama + DDP) |
| `autoresearch/prepare_1b.py` | Data loading (70/30 KK/ENKK mix) |
| `autoresearch/upload_1b_to_hf.py` | HF upload with key mapping |
| `autoresearch/run_1b_training.sh` | Autonomous pipeline with Telegram |
| `autoresearch/launch_1b_pod.py` | RunPod 8×H100 provisioning |
| `scripts/inference/fix_1b_embeddings.py` | Post-hoc embedding scaling fix |
| `scripts/inference/eval_1b_patched.py` | Inference with monkey-patched QK-Norm |
| `scripts/inference/finetune_qknorm.py` | Failed QK-Norm recovery attempt |
| `scripts/data/filter_foreign_chars.py` | Dataset cleaning pipeline |
| `ansible/finetune_qknorm.yml` | Ansible: QK-Norm fine-tuning |
| `ansible/fix_1b_model.yml` | Ansible: embedding fix |
| `ansible/run_eval_1b_patched.yml` | Ansible: patched inference |
