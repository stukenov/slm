# Design: exp017 — MoE Shared Router 3B

**Date:** 2026-02-18
**Status:** Approved

## Goal

Create a 128-expert Mixture-of-Experts model (~3B total, ~180M active) by upcycling the trained dense Llama 150M base model, with a shared router architecture and training on combined Kazakh + EN-KK FineWeb-Edu datasets.

## Architecture

- **Base architecture:** Llama (from `stukenov/sozkz-core-llama-150m-kk-base-v1`)
- **MoE type:** Mixtral-style with modification for shared router
- **Experts:** 128 per layer, top-2 routing
- **Expert FFN:** `intermediate_size=640` (gate_proj + up_proj + down_proj)
- **Shared router:** Single `nn.Linear(768, 128)` shared across all 16 MoE layers
  - Same weights, but applied independently per layer on that layer's hidden states
  - Reduces router params from 16×768×128 = 1.57M to 768×128 = 98K
- **Total params:** ~3B
- **Active params per token:** ~180M (attention + embeddings + 2 experts)
- **Router aux loss:** load-balancing (coef=0.01) + jitter noise 0.1
- **Activation:** SiLU (SwiGLU)

### Key dimensions

| Component | Value |
|---|---|
| hidden_size | 768 |
| num_hidden_layers | 16 |
| num_attention_heads | 12 |
| num_key_value_heads | 12 |
| intermediate_size (per expert) | 640 |
| num_local_experts | 128 |
| num_experts_per_tok | 2 |
| vocab_size | 50257 |
| max_position_embeddings | 1024 |

## Initialization (Upcycling)

1. Load dense model `stukenov/sozkz-core-llama-150m-kk-base-v1`
2. For each layer: copy FFN (gate_proj, up_proj, down_proj) to all 128 experts
3. Add Gaussian noise (σ=0.01) to each expert copy for symmetry breaking
4. Resize expert projections from 2048→640 intermediate (truncate or reinit)
5. Initialize shared router with Xavier uniform
6. Copy attention, embeddings, LayerNorm unchanged
7. Upload init model to HF as `stukenov/sozkz-moe-mix-3b-kk-base-v1-init`

**Note:** Since expert intermediate_size (640) differs from dense (2048), we reinitialize expert FFN weights from scratch (Xavier) rather than truncating. The attention/embed layers carry over the pretrained knowledge.

## Datasets

### Already tokenized
- `stukenov/sozkz-corpus-tokenized-kk-llama50k-v3`

### Needs tokenization
- `stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1`
- Tokenizer: `saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1`
- Upload as: `stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1`
- Block size: 1024

### Combined training
- Interleave both datasets during training

## Batch Strategy

With 128 experts and top-2 routing, we need large batches to ensure all experts get gradients:
- **Minimum tokens per step:** ~500K (512 seqs × 1024 tokens)
- Achieved via: `per_device_train_batch_size × gradient_accumulation_steps × num_gpus × 1024`
- Example: bs=4 × grad_accum=128 × 1 GPU = 512 seqs = 524K tokens/step

## Hardware Options

| Config | VRAM | Price/hr | Est. hours | Est. cost |
|--------|------|----------|------------|-----------|
| 1× A100 80GB | 80GB | ~$1.5 | 40-60h | $60-90 |
| 1× H100 80GB | 80GB | ~$2.5 | 20-30h | $50-75 |
| 2× A100 40GB | 80GB | ~$2.0 | 30-40h | $60-80 |
| 4× RTX 4090 | 96GB | ~$2.0 | 25-35h | $50-70 |

**Memory estimate (bf16):** 3B params × 2 bytes = 6GB + AdamW states ×3 = 18GB + activations ~10-20GB = **~30-40GB** total. A100 80GB is comfortable.

## Custom Code Needed

1. **`src/slm/moe_upcycle.py`** — Upcycling script: dense Llama → MoE with shared router
2. **Shared router monkey-patch** — Modify MixtralSparseMoeBlock to share router weights
3. **`configs/experiments/exp017_moe_shared_router_3b.yaml`** — Experiment config
4. **Tokenization script** — For the new dataset

## Training Hyperparameters

- learning_rate: 1e-4 (lower since upcycled, not from scratch)
- weight_decay: 0.1
- warmup_steps: 1000
- lr_scheduler: cosine
- num_train_epochs: 1
- max_grad_norm: 1.0
- bf16: true
- router_aux_loss_coef: 0.01
- router_jitter_noise: 0.1
- torch_compile: true

## HuggingFace Naming

- Init model: `stukenov/sozkz-moe-mix-3b-kk-base-v1-init`
- Trained model: `stukenov/sozkz-moe-mix-3b-kk-base-v1`
- Tokenized dataset: `stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1`
