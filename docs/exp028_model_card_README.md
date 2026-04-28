---
license: mit
language:
  - kk
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - kazakh
  - llama
  - causal-lm
  - gqa
  - from-scratch
datasets:
  - stukenov/sozkz-corpus-clean-kk-text-v4
  - stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v2
base_model: []
model-index:
  - name: sozkz-core-llama-1b-kk-base-v1
    results: []
---

# SozKZ Core Llama 1B Kazakh Base v1

A 1.08B parameter Llama-architecture language model trained from scratch on ~16.2B cleaned Kazakh and English-Kazakh parallel tokens. This is the largest dense Kazakh-only language model in the SozKZ series.

> **Known Issue**: This model was trained with QK-Norm (RMSNorm on Q,K in attention), which is not supported by HuggingFace's `LlamaForCausalLM`. The QK-Norm weights were **lost during conversion** and cannot be recovered. The model generates Kazakh text but with degraded quality compared to what was achieved during training. See [Limitations](#limitations) for details.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Llama (decoder-only transformer) |
| Parameters | ~1.08B |
| Hidden size | 2048 |
| Layers | 22 |
| Attention heads | 16 |
| KV heads | 4 (Grouped Query Attention) |
| Intermediate size | 5504 (SwiGLU) |
| Vocab size | 50,257 |
| Max sequence length | 1024 |
| Tied embeddings | Yes |
| Precision | bfloat16 |

### Training Details

| Property | Value |
|----------|-------|
| Training tokens | ~16.2B (70% Kazakh / 30% EN-KK parallel) |
| Chinchilla ratio | 15:1 |
| Hardware | 8x NVIDIA H100 SXM 80GB |
| Training time | 9.53 hours |
| Final loss | 2.15 |
| Val BPB | 0.8127 |
| Optimizer | AdamW (lr=2e-4, cosine schedule) |
| Batch size | 2M tokens/step |

### Training Data

The model was trained on a curated mix of two cleaned datasets:

- **70%** — `stukenov/sozkz-corpus-clean-kk-text-v4`: ~12.3M cleaned Kazakh texts (~8.4B tokens)
- **30%** — `stukenov/sozkz-corpus-clean-enkk-fineweb-edu-v2`: ~7.8M cleaned English-Kazakh parallel texts (~7.8B tokens)

Data cleaning pipeline removed:
- CJK characters and foreign scripts
- Encoding artifacts and mojibake
- Texts with >5% Latin character ratio
- Repetitive/garbage content

## Usage

```python
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
import torch

model_id = "stukenov/sozkz-core-llama-1b-kk-base-v1"

# Load tokenizer (use PreTrainedTokenizerFast directly — AutoTokenizer may fail on transformers 5.x)
tok_file = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
tokenizer.pad_token_id = 1

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Generate
prompt = "Қазақстан Президенті"
ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    ids, max_new_tokens=200, do_sample=True,
    temperature=0.7, top_p=0.9, repetition_penalty=1.1,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Tokenizer

Uses [`stukenov/sozkz-core-gpt2-50k-kk-base-v1`](https://huggingface.co/stukenov/sozkz-core-gpt2-50k-kk-base-v1) — a ByteLevel BPE tokenizer with 50,257 tokens optimized for Kazakh text.

## Architectural Features

### Grouped Query Attention (GQA)
Uses 4 KV heads instead of 16, reducing KV cache by 4x for faster inference while maintaining quality. Fully compatible with HuggingFace Llama.

### Z-loss Regularization
Trained with z-loss coefficient of 1e-4 (`z_loss = 1e-4 * logits.logsumexp(-1).pow(2).mean()`) following PaLM/Gemma for training stability. This is a training-time technique and does not affect the saved weights.

## Limitations

### Lost QK-Norm Weights (Critical)

This model was trained with **QK-Norm** (RMSNorm applied to Q and K projections before the dot-product attention), a technique from Gemma 2 and Cohere Command-R that stabilizes attention and prevents logit growth.

However, HuggingFace's `LlamaForCausalLM` does not support QK-Norm parameters. During conversion to HF format, the QK-Norm weights (44 parameters per layer, 968 total across 22 layers) were **silently dropped** by `load_state_dict(strict=False)`. The original training checkpoint was on an ephemeral cloud GPU that was destroyed after upload.

**Impact**: The model generates Kazakh text but with quality artifacts. The attention mechanism operates without the normalization it was co-trained with for 9.5 hours across 16.2B tokens. Recovery attempts (identity weights, fine-tuning) were unsuccessful.

### Embedding Scaling (Partially Fixed)

The model was trained with Gemma-style embedding scaling (`x * sqrt(2048) ≈ x * 45.25`). This scaling was originally code-only, but has since been **baked into the embedding weights** via post-hoc multiplication. This fix is applied in the current version on the Hub.

### Other Limitations

- **Base model only**: No instruction tuning or RLHF — generates raw continuations
- **1024 token context**: Maximum sequence length is 1024 tokens
- **Kazakh-centric**: Primarily trained on Kazakh; English capability is limited to what was present in the parallel corpus
- **Quality degradation**: Due to the lost QK-Norm weights, output quality is below what was achieved during training (loss 2.15, BPB 0.8127)

## Model Family

| Model | Params | Data | Status |
|-------|--------|------|--------|
| [sozkz-core-llama-50m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-50m-kk-base-v1) | 50M | 1.8B tok | Completed |
| [sozkz-core-llama-150m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-150m-kk-base-v1) | 152M | 1.8B tok | Completed |
| [sozkz-core-llama-600m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-600m-kk-base-v1) | 587M | 9B tok | Completed |
| **sozkz-core-llama-1b-kk-base-v1** | **1.08B** | **16.2B tok** | **Degraded** |

## Citation

```bibtex
@misc{tukenov2026sozkz1b,
  title={SozKZ Core Llama 1B: A Kazakh Language Model},
  author={Tukenov, Saken},
  year={2026},
  url={https://huggingface.co/stukenov/sozkz-core-llama-1b-kk-base-v1}
}
```

## License

MIT
