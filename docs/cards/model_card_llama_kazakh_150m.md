---
language:
- kk
license: apache-2.0
tags:
- safetensors
- kazakh
- llama
- language-model
- causal-lm
- from-scratch
- small-language-model
library_name: transformers
pipeline_tag: text-generation
datasets:
- kz-transformers/multidomain-kazakh-dataset
model-index:
- name: sozkz-core-llama-150m-kk-base-v1
  results: []
---

# Llama-Kazakh-150M

A 150M-parameter Llama language model trained **from scratch** on Kazakh text. This is the largest model in the [SLM (Small Language Models for Kazakh)](https://github.com/sakentukenov/slm) project, designed to be Chinchilla-optimal for 2.88B training tokens.

## Model Details

### Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | LlamaForCausalLM |
| Parameters | **149.8M** |
| Hidden size | 896 |
| Intermediate size | 2,560 (SwiGLU MLP) |
| Num layers | 12 |
| Attention heads | 16 (full MHA, no GQA) |
| Head dimension | 56 |
| Max context length | 1,024 tokens |
| Vocabulary size | 32,000 |
| Positional encoding | RoPE (theta=10,000) |
| Tied embeddings | Yes |
| Activation | SiLU |
| Norm | RMSNorm (eps=1e-6) |

### Tokenizer

Uses a **custom Kazakh ByteLevel BPE tokenizer** ([saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1)) specifically trained for Kazakh:

| Property | Value |
|----------|-------|
| Type | ByteLevel BPE |
| Vocabulary | 32,000 tokens |
| Training corpus | 23.6M Kazakh text samples |
| Special tokens | `<\|endoftext\|>` (id=0, bos/eos), `<\|padding\|>` (id=1, pad) |

**Why a custom tokenizer?** Multilingual tokenizers (e.g., GPT-NeoX, LLaMA) allocate very few vocabulary entries for Kazakh, leading to poor token efficiency (many characters split into individual bytes). A dedicated 32K Kazakh tokenizer provides much better compression and faster training convergence.

## Training

### Dataset

Trained on [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset):

| Property | Value |
|----------|-------|
| Language | Kazakh (kk) |
| Train samples | 23.6M |
| Validation samples | 1.2M |
| Domains | News, literature, Wikipedia, legal texts, academic content, web |
| Pre-tokenized version | [saken-tukenov/sozkz-core-llama-50m-kk-base-v1-tokenized](https://huggingface.co/datasets/saken-tukenov/sozkz-core-llama-50m-kk-base-v1-tokenized) |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Initialization | Random (from scratch) |
| Precision | bfloat16 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| LR scheduler | Cosine decay |
| Warmup steps | 500 |
| Weight decay | 0.1 |
| Max gradient norm | 1.0 |
| Block size (context) | 512 tokens |
| Per-device batch size | 32 |
| Gradient accumulation | 2 |
| Effective batch size | 512 (32 x 8 GPUs x 2) |
| Epochs | 1 |
| Total steps | ~10,972 |

### Infrastructure

| Component | Details |
|-----------|---------|
| Hardware | 8x NVIDIA H200 (vast.ai) |
| Framework | PyTorch 2.4 + Transformers 5.1 |
| Distributed | torchrun DDP (8 processes) |
| Training time | ~36 minutes |

### Training Curve

| Step | Train Loss | Grad Norm | Learning Rate | Epoch |
|------|-----------|-----------|---------------|-------|
| 500 | ~5.5 | - | 3.0e-4 | 0.046 |
| 2,500 | ~3.5 | - | 2.9e-4 | 0.228 |
| 5,000 | ~3.2 | ~0.29 | 2.5e-4 | 0.456 |
| 7,500 | ~3.0 | ~0.29 | 1.8e-4 | 0.683 |
| 10,000 | ~2.9 | ~0.28 | 0.8e-4 | 0.911 |
| 10,972 | ~2.9 | ~0.28 | 0.0e-4 | 1.000 |

Final train loss: **~2.9** | Final eval loss: **~3.0** | Perplexity: **~20**

## Usage

### Basic Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "saken-tukenov/sozkz-core-llama-150m-kk-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

prompt = "Қазақстан — бұл"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Generation Examples

**Prompt:** `Қазақстан — бұл`
> Қазақстан — бұл адамның өмір сүруіне қажетті жағдай жасайтын құрал. Біздің қоғамымызда осы мәселені шешуде мемлекеттің рөлі жоғары, себебі онда біз өзімізді өзіміз де дамытуымыз керек.

**Prompt:** `Бүгін ауа райы`
> Бүгін ауа райы болжамы «Қазгидромет» РМК 25 қыркүйекке арналған ауа райы болжамын жариялады. Ал Солтүстік Қазақстан облысында түнде 2 градус үсік жүреді...

**Prompt:** `Алыстағы ауылда бір кәрі шал тұратын,`
> Алыстағы ауылда бір кәрі шал тұратын, көпқабатты үйдің баласындай өсіп-өнген. Ол кезде де талай бала арманда кеткен болатын...

**Prompt:** `Жасанды интеллект технологиялары`
> Жасанды интеллект технологиялары» атты халықаралық ғылыми-тәжірибелік конференциясы өтті. Конференция жұмысына университет ғалымдары...

## Limitations

- **Factual accuracy**: The model generates plausible-sounding but often factually incorrect text. It should not be used as a knowledge source.
- **Domain bias**: Heavily biased toward news and government press release style, reflecting the training data distribution.
- **Language mixing**: Occasionally inserts Russian words or phrases mid-sentence.
- **Repetition**: May fall into repetitive patterns, especially with longer generations.
- **Context length**: Limited to 1,024 tokens maximum.
- **Size**: At 150M parameters, this is a small model intended for research, not production use.

## Intended Use

This model is intended for:
- Research on small language models for low-resource languages
- Kazakh NLP benchmarking and evaluation
- Text generation experiments
- Fine-tuning on downstream Kazakh NLP tasks
- Educational purposes

This model is **not** intended for:
- Factual question answering
- Production applications requiring reliability
- Content generation without human review

## Related Models

| Model | Params | Description |
|-------|--------|-------------|
| [saken-tukenov/sozkz-core-llama-150m-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-llama-150m-kk-base-v1) | 150M | **This model** — Llama from scratch |
| [saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1) | - | Custom Kazakh tokenizer used by this model |

## Citation

```bibtex
@misc{tukenov2026llamakazakh150m,
  title={Llama-Kazakh-150M: A Small Language Model for Kazakh},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/saken-tukenov/sozkz-core-llama-150m-kk-base-v1}
}
```

## Acknowledgments

- Training dataset: [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset)
- Architecture: Based on [LLaMA](https://arxiv.org/abs/2302.13971) by Meta AI
- Scaling laws: Model size follows [Chinchilla](https://arxiv.org/abs/2203.15556) optimal allocation for ~2.88B tokens
- Training infrastructure: [vast.ai](https://vast.ai)
