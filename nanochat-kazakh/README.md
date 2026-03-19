# nanochat-kazakh

Minimal, reproducible recipe for training a **50M-parameter Kazakh language model** from scratch — tokenizer, data pipeline, training, and evaluation.

**Final model**: [saken-tukenov/sozkz-core-llama-50m-kk-base-v2](https://huggingface.co/saken-tukenov/sozkz-core-llama-50m-kk-base-v2)
**Tokenizer**: [saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1)

## What you get

| Component | Details |
|-----------|---------|
| Architecture | LlamaForCausalLM, 50.29M params |
| Hidden / Layers / Heads | 512 / 8 / 8 |
| Context length | 1024 tokens |
| Vocab size | 50,257 (ByteLevel BPE) |
| Training tokens | ~1.04B |
| Training regime | 1 epoch, cosine LR 6e-4, bf16 |

## Quick start

### 0. Install

```bash
pip install torch transformers datasets accelerate tokenizers huggingface-hub
```

### 1. Train tokenizer (optional — already published)

```bash
python train_tokenizer.py
# -> saves to ./tokenizers/sozkz-core-gpt2-50k-kk-base-v1
# -> published: saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1
```

Trains a 50,257-token ByteLevel BPE tokenizer on the [clean Kazakh text corpus](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-text-v2).

### 2. Tokenize dataset (optional — already published)

```bash
python tokenize_data.py
# -> pushes to HF Hub as saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2
```

Packs text into 1024-token blocks with `<|endoftext|>` separators between documents. ~1M blocks, ~1.04B tokens total.

### 3. Train model

```bash
# Single GPU
python train.py

# Multi-GPU (DDP)
torchrun --nproc_per_node=2 train.py
```

Uses [pre-tokenized data](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2), so training starts immediately with no preprocessing.

**Expected**: ~1 epoch over 1B tokens. On 1x A100 this takes about 6-8 hours.

### 4. Generate text

```bash
python generate.py --model ./output/final
# or from HF Hub:
python generate.py --model saken-tukenov/sozkz-core-llama-50m-kk-base-v2
```

## File overview

```
nanochat-kazakh/
├── README.md              # this file
├── train_tokenizer.py     # step 1: train BPE tokenizer
├── tokenize_data.py       # step 2: tokenize + pack into blocks
├── train.py               # step 3: train LlamaForCausalLM from scratch
├── generate.py            # step 4: text generation / inference
└── prompts_kk.txt         # evaluation prompts in Kazakh
```

## Architecture

```
LlamaForCausalLM (50.29M parameters)
├── Embedding:      50,257 × 512 (tied with LM head)
├── 8× LlamaDecoderLayer:
│   ├── Self-attention: 8 heads, dim 64 each
│   ├── MLP (SwiGLU): 512 → 1344 → 512
│   └── RMSNorm (pre-norm)
└── RMSNorm + LM Head (tied)
```

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| Text (raw) | Cleaned Kazakh text, ~78K documents | [sozkz-corpus-clean-kk-text-v2](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-text-v2) |
| Tokenized | 1024-token blocks, ready for training | [sozkz-corpus-clean-kk-pretrain-v2](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2) |

The text corpus was cleaned through a multi-stage pipeline: NFC normalization, Kazakh character filtering, script profiling (Cyrillic ≥ 60%), fastText language ID, junk/boilerplate removal, repetition filtering, exact + near deduplication (MinHash LSH), and domain balancing.

## Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 6e-4 |
| LR scheduler | Cosine |
| Warmup steps | 500 |
| Weight decay | 0.1 |
| Max grad norm | 1.0 |
| Batch size | 16 × 2 (grad accum) = 32 per device |
| Precision | bfloat16 |
| Epochs | 1 (Chinchilla-optimal for ~1B tokens) |
| Optimizer | AdamW |

## License

MIT
