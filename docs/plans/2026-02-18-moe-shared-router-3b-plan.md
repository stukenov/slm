# MoE Shared Router 3B Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 128-expert MoE model (~3B total, ~180M active) by upcycling the dense Llama 150M, with shared router weights, and train on combined Kazakh + EN-KK FineWeb-Edu datasets on vast.ai.

**Architecture:** Mixtral-based MoE with a single shared `nn.Linear(768, 128)` router across all 16 layers. Each expert has `intermediate_size=640`. Initialized by upcycling `stukenov/sozkz-core-llama-150m-kk-base-v1` (attention/embed copied, FFN experts initialized fresh with Xavier since intermediate_size differs).

**Tech Stack:** transformers (MixtralConfig/MixtralForCausalLM), datasets, torch, huggingface_hub, vast.ai cloud pipeline

**Design doc:** `docs/plans/2026-02-18-moe-shared-router-3b-design.md`

---

## Phase 1: Dataset Tokenization

### Task 1: Create tokenization script for the new dataset

**Files:**
- Create: `scripts/tokenize_dataset.py`

**Step 1: Write tokenization script**

```python
"""Tokenize a HuggingFace dataset and upload the result.

Usage:
    python scripts/tokenize_dataset.py \
        --dataset stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1 \
        --tokenizer saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1 \
        --output stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1 \
        --block-size 1024
"""
from __future__ import annotations

import argparse
import logging
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def tokenize_and_upload(
    dataset_name: str,
    tokenizer_name: str,
    output_repo: str,
    block_size: int = 1024,
    text_column: str = "text",
    num_proc: int = 16,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> None:
    logger.info("Loading dataset: %s", dataset_name)
    ds = load_dataset(dataset_name)

    # Get train split
    if isinstance(ds, dict):
        if "train" in ds:
            raw = ds["train"]
        else:
            raw = next(iter(ds.values()))
    else:
        raw = ds

    logger.info("Dataset size: %d rows", len(raw))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples[text_column], return_attention_mask=False)

    logger.info("Tokenizing with %d workers...", num_proc)
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )

    # Group into blocks
    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Grouping into blocks of %d...", block_size)
    blocked = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping",
    )

    logger.info("Total blocks: %d", len(blocked))

    # Split train/val
    split = blocked.train_test_split(test_size=val_ratio, seed=seed)
    from datasets import DatasetDict
    final = DatasetDict({"train": split["train"], "validation": split["test"]})
    logger.info("Train: %d, Validation: %d", len(final["train"]), len(final["validation"]))

    # Upload
    logger.info("Uploading to %s...", output_repo)
    final.push_to_hub(output_repo, private=False)
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    tokenize_and_upload(
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        output_repo=args.output,
        block_size=args.block_size,
        text_column=args.text_column,
        num_proc=args.num_proc,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
```

**Step 2: Run tokenization on vast.ai (cheap CPU instance or locally)**

Option A — via cloud pipeline with pre-cmd:
```bash
# Dry run first to check pricing
PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud launch \
  --config configs/experiments/exp017_moe_shared_router_3b.yaml \
  --hf-repo stukenov/sozkz-moe-mix-3b-kk-base-v1 \
  --dry-run
```

Option B — run locally if dataset isn't too large:
```bash
python scripts/tokenize_dataset.py \
  --dataset stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1 \
  --tokenizer saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1 \
  --output stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1 \
  --block-size 1024 --num-proc 8
```

**Step 3: Verify upload**
```bash
python -c "from datasets import load_dataset; ds = load_dataset('stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1'); print(ds)"
```
Expected: DatasetDict with train and validation splits containing input_ids, labels columns.

**Step 4: Commit**
```bash
git add scripts/tokenize_dataset.py
git commit -m "feat: add dataset tokenization script for MoE training"
```

---

## Phase 2: MoE Upcycling & Shared Router

### Task 2: Create MoE upcycling script with shared router

**Files:**
- Create: `src/slm/moe_upcycle.py`

**Step 1: Write the upcycling script**

```python
"""Upcycle a dense Llama model into a Mixtral-style MoE with shared router.

Usage:
    python -m slm.moe_upcycle \
        --dense-model stukenov/sozkz-core-llama-150m-kk-base-v1 \
        --output-dir ./outputs/moe_init \
        --num-experts 128 \
        --num-experts-per-tok 2 \
        --expert-intermediate-size 640 \
        [--push-to-hub stukenov/sozkz-moe-mix-3b-kk-base-v1-init]
"""
from __future__ import annotations

import argparse
import copy
import logging
import math

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MixtralConfig,
    MixtralForCausalLM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def upcycle_to_moe(
    dense_model_name: str,
    num_experts: int = 128,
    num_experts_per_tok: int = 2,
    expert_intermediate_size: int = 640,
    router_jitter_noise: float = 0.1,
    router_aux_loss_coef: float = 0.01,
    noise_std: float = 0.01,
) -> tuple[MixtralForCausalLM, AutoTokenizer]:
    """Convert a dense Llama model to Mixtral MoE with shared router."""

    logger.info("Loading dense model: %s", dense_model_name)
    dense_model = AutoModelForCausalLM.from_pretrained(dense_model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(dense_model_name)
    dense_config = dense_model.config

    # Build Mixtral config from Llama config
    moe_config = MixtralConfig(
        vocab_size=dense_config.vocab_size,
        hidden_size=dense_config.hidden_size,
        intermediate_size=expert_intermediate_size,
        num_hidden_layers=dense_config.num_hidden_layers,
        num_attention_heads=dense_config.num_attention_heads,
        num_key_value_heads=getattr(dense_config, "num_key_value_heads", dense_config.num_attention_heads),
        max_position_embeddings=dense_config.max_position_embeddings,
        rms_norm_eps=getattr(dense_config, "rms_norm_eps", 1e-5),
        tie_word_embeddings=getattr(dense_config, "tie_word_embeddings", True),
        pad_token_id=dense_config.pad_token_id,
        bos_token_id=dense_config.bos_token_id,
        eos_token_id=dense_config.eos_token_id,
        num_local_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        output_router_logits=True,
        router_aux_loss_coef=router_aux_loss_coef,
        router_jitter_noise=router_jitter_noise,
    )

    logger.info("Creating MoE model: %d experts, top-%d, intermediate=%d",
                num_experts, num_experts_per_tok, expert_intermediate_size)
    moe_model = MixtralForCausalLM(moe_config)

    # Copy embeddings
    moe_model.model.embed_tokens.weight.data.copy_(dense_model.model.embed_tokens.weight.data)
    if hasattr(moe_model, "lm_head") and not moe_config.tie_word_embeddings:
        moe_model.lm_head.weight.data.copy_(dense_model.lm_head.weight.data)

    # Copy final norm
    moe_model.model.norm.weight.data.copy_(dense_model.model.norm.weight.data)

    # Copy per-layer weights
    for layer_idx in range(dense_config.num_hidden_layers):
        dense_layer = dense_model.model.layers[layer_idx]
        moe_layer = moe_model.model.layers[layer_idx]

        # Copy attention weights
        moe_layer.self_attn.q_proj.weight.data.copy_(dense_layer.self_attn.q_proj.weight.data)
        moe_layer.self_attn.k_proj.weight.data.copy_(dense_layer.self_attn.k_proj.weight.data)
        moe_layer.self_attn.v_proj.weight.data.copy_(dense_layer.self_attn.v_proj.weight.data)
        moe_layer.self_attn.o_proj.weight.data.copy_(dense_layer.self_attn.o_proj.weight.data)

        # Copy layer norms
        moe_layer.input_layernorm.weight.data.copy_(dense_layer.input_layernorm.weight.data)
        moe_layer.post_attention_layernorm.weight.data.copy_(dense_layer.post_attention_layernorm.weight.data)

        # Expert FFN: since intermediate_size differs, initialize with Xavier
        # (can't copy 2048->640 meaningfully), but add small noise for symmetry breaking
        for expert_idx in range(num_experts):
            expert = moe_layer.block_sparse_moe.experts[expert_idx]
            nn.init.xavier_uniform_(expert.w1.weight)  # gate_proj
            nn.init.xavier_uniform_(expert.w2.weight)  # down_proj
            nn.init.xavier_uniform_(expert.w3.weight)  # up_proj
            # Add unique noise per expert for symmetry breaking
            expert.w1.weight.data += torch.randn_like(expert.w1.weight) * noise_std
            expert.w2.weight.data += torch.randn_like(expert.w2.weight) * noise_std
            expert.w3.weight.data += torch.randn_like(expert.w3.weight) * noise_std

    # Shared router: make all layers share the same router parameters
    shared_gate = moe_model.model.layers[0].block_sparse_moe.gate
    nn.init.xavier_uniform_(shared_gate.weight)
    for layer_idx in range(1, dense_config.num_hidden_layers):
        # Replace gate with reference to shared gate (same nn.Module)
        moe_model.model.layers[layer_idx].block_sparse_moe.gate = shared_gate

    total_params = sum(p.numel() for p in moe_model.parameters())
    unique_params = sum(p.numel() for p in set(moe_model.parameters()))
    logger.info("Total params (with sharing): %.2fB", total_params / 1e9)
    logger.info("Unique params: %.2fB", unique_params / 1e9)

    return moe_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Upcycle dense Llama to MoE")
    parser.add_argument("--dense-model", required=True, help="HF model ID or local path")
    parser.add_argument("--output-dir", default="./outputs/moe_init")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--expert-intermediate-size", type=int, default=640)
    parser.add_argument("--push-to-hub", default=None, help="HF repo to upload to")
    args = parser.parse_args()

    model, tokenizer = upcycle_to_moe(
        dense_model_name=args.dense_model,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        expert_intermediate_size=args.expert_intermediate_size,
    )

    logger.info("Saving to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        logger.info("Pushing to %s", args.push_to_hub)
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        logger.info("Upload complete!")


if __name__ == "__main__":
    main()
```

**Step 2: Test locally (dry run — just init, no training)**
```bash
PYTHONPATH=src python -m slm.moe_upcycle \
  --dense-model stukenov/sozkz-core-llama-150m-kk-base-v1 \
  --output-dir ./outputs/moe_init_test \
  --num-experts 4 \
  --expert-intermediate-size 64
```
Expected: Model saves successfully, prints param counts. Use 4 experts for quick local test.

**Step 3: Verify shared router**
```python
# Quick sanity check
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("./outputs/moe_init_test")
# After loading from disk, gates are separate copies. The sharing happens at init time.
# During training, we need to re-share them. See Task 3.
```

**Step 4: Commit**
```bash
git add src/slm/moe_upcycle.py
git commit -m "feat: add MoE upcycling script with shared router support"
```

---

### Task 3: Add shared router hook to train.py

The shared router trick (all layers reference same `gate` module) works during upcycling but is lost when saving/loading (each gate gets its own copy of weights). We need a hook in training to re-share them after loading.

**Files:**
- Modify: `src/slm/train.py` (add shared router support)

**Step 1: Add shared router re-linking function to train.py**

After model loading (line ~91), add:

```python
# After model is loaded, before training
if config.get("shared_router"):
    logger.info("Enabling shared router: linking all MoE gates to layer 0")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        shared_gate = model.model.layers[0].block_sparse_moe.gate
        for layer_idx in range(1, len(model.model.layers)):
            if hasattr(model.model.layers[layer_idx], "block_sparse_moe"):
                model.model.layers[layer_idx].block_sparse_moe.gate = shared_gate
        logger.info("Shared router linked across %d layers", len(model.model.layers))
```

**Step 2: Add multi-dataset support to train.py**

After pretokenized dataset loading (line ~125), add support for `pretokenized_datasets` (plural):

```python
# Support loading and concatenating multiple pretokenized datasets
extra_datasets = config.get("extra_pretokenized_datasets", [])
if extra_datasets:
    from datasets import concatenate_datasets
    for extra_name in extra_datasets:
        logger.info("Loading extra dataset: %s", extra_name)
        extra_ds = _load_ds(extra_name)
        datasets["train"] = concatenate_datasets([datasets["train"], extra_ds["train"]])
        if "validation" in extra_ds:
            datasets["validation"] = concatenate_datasets([datasets["validation"], extra_ds["validation"]])
    logger.info("Combined train size: %d, validation size: %d",
                len(datasets["train"]), len(datasets["validation"]))
```

**Step 3: Commit**
```bash
git add src/slm/train.py
git commit -m "feat: add shared router re-linking and multi-dataset support"
```

---

## Phase 3: Experiment Config

### Task 4: Create experiment YAML config

**Files:**
- Create: `configs/experiments/exp017_moe_shared_router_3b.yaml`

**Step 1: Write config**

```yaml
inherits: base

experiment_name: exp017_moe_shared_router_3b
model_name: sozkz-moe-mix-3b-kk-base-v1

# Init from upcycled MoE (created by moe_upcycle.py)
init_from: stukenov/sozkz-moe-mix-3b-kk-base-v1-init

# Datasets (combined)
pretokenized_dataset: stukenov/sozkz-corpus-tokenized-kk-llama50k-v3
extra_pretokenized_datasets:
  - stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1

# Tokenizer
tokenizer_path: saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1
tokenizer_mode: pretrained

# Shared router
shared_router: true

# Training — large batch for 128 experts
learning_rate: 1e-4
weight_decay: 0.1
num_train_epochs: 1
per_device_train_batch_size: 4
gradient_accumulation_steps: 128
warmup_steps: 1000
max_grad_norm: 1.0
lr_scheduler_type: cosine

# Eval/Save
eval_steps: 2000
save_steps: 2000
logging_steps: 50
save_total_limit: 3

# Performance
bf16: true
dataloader_num_workers: 4
torch_compile: true
```

**Step 2: Commit**
```bash
git add configs/experiments/exp017_moe_shared_router_3b.yaml
git commit -m "feat: add exp017 MoE shared router 3B config"
```

---

## Phase 4: Tokenize New Dataset

### Task 5: Run tokenization of FineWeb-Edu dataset

**Step 1: Check dataset size first**
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1', streaming=True)
print(next(iter(ds['train'])))
"
```

**Step 2: Run tokenization (locally or on a cheap cloud instance)**
```bash
python scripts/tokenize_dataset.py \
  --dataset stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1 \
  --tokenizer saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1 \
  --output stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1 \
  --block-size 1024 --num-proc 8
```

**Step 3: Verify**
```bash
python -c "from datasets import load_dataset; ds = load_dataset('stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1'); print(ds); print(ds['train'][0].keys())"
```
Expected: DatasetDict with train/validation, columns: input_ids, labels

---

## Phase 5: Upcycle and Upload Init Model

### Task 6: Create and upload the upcycled MoE init model

**Step 1: Run full upcycling (needs ~12GB RAM for 3B model in bf16)**
```bash
PYTHONPATH=src python -m slm.moe_upcycle \
  --dense-model stukenov/sozkz-core-llama-150m-kk-base-v1 \
  --output-dir ./outputs/moe_init \
  --num-experts 128 \
  --num-experts-per-tok 2 \
  --expert-intermediate-size 640 \
  --push-to-hub stukenov/sozkz-moe-mix-3b-kk-base-v1-init
```

**Step 2: Verify upload**
```bash
python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('stukenov/sozkz-moe-mix-3b-kk-base-v1-init')
print(f'Params: {sum(p.numel() for p in m.parameters())/1e9:.2f}B')
print(f'Config experts: {m.config.num_local_experts}')
"
```
Expected: ~3B params, 128 experts

---

## Phase 6: GPU Selection & Training Launch

### Task 7: Select optimal GPU and launch training

**Step 1: Dry run to check GPU prices**
```bash
PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud launch \
  --config configs/experiments/exp017_moe_shared_router_3b.yaml \
  --hf-repo stukenov/sozkz-moe-mix-3b-kk-base-v1 \
  --max-price 3.00 --num-gpus 1 --disk 100 \
  --dry-run
```

**Step 2: Evaluate options and select GPU**

Priority order:
1. **H100 80GB** — fastest, best $/token if available <$3/hr
2. **A100 80GB** — proven reliable, ~$1.5/hr
3. **2× A100 40GB** — if single 80GB unavailable (needs DDP)
4. **L40S 48GB** — might work with careful memory management

**Step 3: Launch training**
```bash
PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud launch \
  --config configs/experiments/exp017_moe_shared_router_3b.yaml \
  --hf-repo stukenov/sozkz-moe-mix-3b-kk-base-v1 \
  --max-price 3.00 --num-gpus 1 --disk 100 \
  --monitor
```

**Step 4: Monitor training**
```bash
PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud monitor --instance-id <ID>
```

Key metrics to watch:
- Train loss: should decrease steadily
- Router entropy: should stay high (near log(128) ≈ 4.85) — indicates good expert utilization
- No expert collapse (one expert getting all tokens)

---

## Phase 7: Evaluation & Publishing

### Task 8: Evaluate and publish final model

**Step 1: Run evaluation**
```bash
python -m slm.evaluate \
  --model_path stukenov/sozkz-moe-mix-3b-kk-base-v1 \
  --prompts eval/prompts_kk.txt
```

**Step 2: Record results in WHITEPAPER.md**

**Step 3: Create model card and verify on HuggingFace**
