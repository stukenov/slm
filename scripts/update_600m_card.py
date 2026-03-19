"""Upload README and set gated access for 600M model."""
import os
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

from huggingface_hub import HfApi

api = HfApi()
repo_id = "stukenov/sozkz-core-llama-600m-kk-base-v1"

readme = '''---
language:
- kk
license: mit
tags:
- llama
- kazakh
- causal-lm
- pretrained
datasets:
- stukenov/sozkz-corpus-tokenized-kk-llama50k-v3
pipeline_tag: text-generation
model-index:
- name: sozkz-core-llama-600m-kk-base-v1
  results:
  - task:
      type: text-generation
      name: Language Modeling
    metrics:
    - name: Validation BPB
      type: bpb
      value: 0.756
    - name: Training Loss
      type: loss
      value: 2.713
---

# SozKZ Core Llama 600M — Kazakh Base v1

A 587M parameter Llama model pretrained from scratch on 9 billion Kazakh tokens. Part of the SozKZ family of Kazakh language models.

## Model Family

| Model | Params | val_bpb | Train Loss | Status |
|-------|--------|---------|------------|--------|
| [sozkz-core-llama-150m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-150m-kk-base-v1) | 152M | — | — | Released |
| [sozkz-core-llama-300m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-300m-kk-base-v1) | 325M | 0.781 | 2.848 | Released |
| **sozkz-core-llama-600m-kk-base-v1** | **587M** | **0.756** | **2.713** | **This model** |

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | Llama (RMSNorm, RoPE, SwiGLU) |
| Parameters | 587M |
| Hidden size | 1280 |
| Layers | 22 |
| Attention heads | 20 |
| KV heads | 20 (MHA) |
| Intermediate size | 4480 |
| Context length | 1024 |
| Vocab size | 50,257 (GPT-2 BPE, Kazakh) |
| Precision | bfloat16 |
| Tied embeddings | Yes |

## Training

| Detail | Value |
|--------|-------|
| Dataset | [sozkz-corpus-tokenized-kk-llama50k-v3](https://huggingface.co/datasets/stukenov/sozkz-corpus-tokenized-kk-llama50k-v3) |
| Tokens | 9B |
| Hardware | 4x NVIDIA H100 80GB HBM3 |
| Training time | 5.9 hours |
| Throughput | 423K tok/s |
| Optimizer | AdamW (lr=4e-4, betas=0.9/0.95, wd=0.1) |
| Schedule | Cosine with 500-step warmup, min_lr=0.1x |
| Batch size | 32 per GPU x 4 GPUs = 128 |
| Gradient clipping | 1.0 |
| Framework | PyTorch 2.4 + torch.compile + DDP |

## Results

| Metric | Value |
|--------|-------|
| **Validation BPB** | **0.756** |
| Training loss | 2.713 |
| Peak VRAM | 64.0 GB/GPU |
| Tokens-to-params ratio | 15.3:1 |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "stukenov/sozkz-core-llama-600m-kk-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

prompt = "Қазақстан — "
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Tokenizer

Uses [sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-gpt2-50k-kk-base-v1) — a 50K vocab ByteLevel BPE tokenizer trained on Kazakh text.

## Limitations

- This is a **base model** (not instruction-tuned) — it completes text, not answers questions
- Training data is web-scraped Kazakh text (educational sites, Wikipedia, news)
- Context length is 1024 tokens
- May generate repetitive or factually incorrect text

## Citation

```bibtex
@misc{sozkz-llama-600m-kk-2026,
  title={SozKZ Core Llama 600M: Kazakh Language Model},
  author={Tukenov, Saken},
  year={2026},
  url={https://huggingface.co/stukenov/sozkz-core-llama-600m-kk-base-v1}
}
```

## License

MIT (gated access — manual approval required)
'''

# Upload README
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    commit_message="Update model card with full details",
)
print("README uploaded!")

# Set gated access
api.update_repo_settings(repo_id=repo_id, gated="manual")
print("Gated access enabled (manual approval)!")
