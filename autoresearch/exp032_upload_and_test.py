#!/usr/bin/env python3
"""exp032: Test model, generate examples, create README, upload to HuggingFace."""

import json
import logging
import math
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_DIR = "/root/exp032_stage7_out"
REPO_ID = "stukenov/sozkz-core-tinyllama-1b-kk-ru-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPTS = {
    "kaz_country": "Қазақстан — бұл",
    "kaz_weather": "Бүгін ауа райы",
    "kaz_education": "Білім — ол",
    "kaz_history": "Қазақ халқының тарихы",
    "kaz_nature": "Алтай тауларында",
    "kaz_food": "Қазақтың ұлттық тағамы — бешбармақ",
    "kaz_city": "Алматы қаласы — бұл",
    "kaz_science": "Ғылым мен технология",
    "rus_country": "Казахстан — это",
    "rus_weather": "Сегодня погода",
    "rus_education": "Образование — это",
    "rus_history": "История казахского народа",
    "rus_culture": "Культура Казахстана богата",
    "rus_economy": "Экономика Казахстана основана на",
    "rus_sport": "Спорт в Казахстане развивается",
    "rus_future": "В будущем Казахстан планирует",
    "eng_country": "Kazakhstan is a country",
    "eng_capital": "The capital of Kazakhstan",
    "eng_hello": "Hello, my name is",
}


def generate(model, tokenizer, prompt, max_new_tokens=100):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.2,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def compute_perplexity(model, tokenizer, texts, max_length=512):
    total_loss = 0
    total_tokens = 0
    model.eval()
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        total_loss += out.loss.item() * enc["input_ids"].shape[1]
        total_tokens += enc["input_ids"].shape[1]
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def create_readme(examples, perplexities, fertility):
    kaz_examples = "\n".join(
        f"**{k}**: `{v[:200]}`" for k, v in examples.items() if k.startswith("kaz")
    )
    rus_examples = "\n".join(
        f"**{k}**: `{v[:200]}`" for k, v in examples.items() if k.startswith("rus")
    )
    eng_examples = "\n".join(
        f"**{k}**: `{v[:200]}`" for k, v in examples.items() if k.startswith("eng")
    )

    kaz_ppl = perplexities.get("kaz", 0)
    rus_ppl = perplexities.get("rus", 0)
    eng_ppl = perplexities.get("eng", 0)

    return f"""---
language:
- kk
- ru
- en
license: apache-2.0
tags:
- kazakh
- russian
- language-adaptation
- tinyllama
- tokenizer-extension
- eeve
- chinese-llama
base_model: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
datasets:
- kz-transformers/multidomain-kazakh-dataset
pipeline_tag: text-generation
---

# SozKZ Core TinyLlama 1B Kazakh-Russian v1

A TinyLlama-1.1B model adapted for **Kazakh** and **Russian** through tokenizer extension and 7-stage continual pretraining.

## Model Description

This model demonstrates a reproducible pipeline for adapting an English-only LLM to low-resource languages, following the methodology of Chinese-LLaMA (Cui et al., 2023), EEVE (Kim et al., 2024), and Swallow (Fujii et al., 2024).

| Property | Value |
|----------|-------|
| Base model | TinyLlama-1.1B-intermediate-step-1431k-3T |
| Parameters | 1.14B |
| Vocabulary | 42,048 (32,000 original + 10,000 Kazakh/Russian) |
| Languages | Kazakh (kk), Russian (ru), English (en) |
| Training data | Multi-Domain Bilingual Kazakh Dataset (6.17B tokens) |
| Training method | 7-stage progressive unfreezing (EEVE-inspired) |
| Total tokens seen | ~2.5B tokens across all stages |
| Hardware | 1x NVIDIA H100 80GB SXM |
| License | Apache 2.0 |

## Tokenizer Extension

The original TinyLlama tokenizer (32K vocab) was extended with 10,000 Kazakh/Russian tokens via SentencePiece protobuf merge (Chinese-LLaMA method).

| Language | Fertility Before | Fertility After | Improvement |
|----------|-----------------|-----------------|-------------|
| Kazakh | 4.00 tok/word | **1.17 tok/word** | 3.4x |
| Russian | 2.00 tok/word | **1.29 tok/word** | 1.6x |
| English | 1.22 tok/word | 1.00 tok/word | ~same |

## 7-Stage Training Pipeline

| Stage | Description | Loss | Trainable |
|-------|-------------|------|-----------|
| 1 | Input embeddings only | 5.73 -> 4.85 | 7.5% |
| 2 | Input + output embeddings | 3.2 -> 2.91 | 15% |
| 3 | Embeddings + LoRA(QKV) r=16 | 2.54 -> 2.68 | 15.3% |
| 4 | Embeddings + LoRA(QKVO+MLP) r=16 | 3.59 -> 2.61 | 16% |
| 5 | Merge LoRA, unfreeze top 50% | 3.02 -> 2.61 | 57.5% |
| 6 | Full fine-tuning | 2.63 -> 2.44 | 100% |
| 7 | Cooldown (freeze embeddings) | ~2.24 | 85% |

## Perplexity

| Language | Perplexity |
|----------|-----------|
| Kazakh | {kaz_ppl:.2f} |
| Russian | {rus_ppl:.2f} |
| English | {eng_ppl:.2f} |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{REPO_ID}")
tokenizer = AutoTokenizer.from_pretrained("{REPO_ID}")

prompt = "Қазақстан — бұл"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Generation Examples

### Kazakh
{kaz_examples}

### Russian
{rus_examples}

### English
{eng_examples}

## Training Details

- **Dataset**: kz-transformers/multidomain-kazakh-dataset (24.9M rows, 6.17B tokens)
  - Kazakh: 3.68B tokens (59.6%), Russian: 2.49B tokens (40.4%)
- **Tokenizer init**: Subword mean (EEVE method)
- **Optimizer**: AdamW (beta1=0.9, beta2=0.95)
- **Precision**: bf16
- **W&B**: [exp032-kazakh-adapt](https://wandb.ai/saken/exp032-kazakh-adapt)

## Methodology

Full methodology: [Adapting TinyLlama to Kazakh: A Reproducible 7-Stage Pipeline](https://github.com/stukenov/slm/blob/main/docs/papers/kazakh-llm-adaptation.md)

## References

1. Cui et al. (2023). Chinese-LLaMA-Alpaca
2. Kim et al. (2024). EEVE: Efficient and Effective Vocabulary Expansion
3. Fujii et al. (2024). Swallow: Continual Pre-Training for Cross-Lingual LLM Adaptation

## Citation

```bibtex
@misc{{sozkz-tinyllama-kk-ru-v1,
  title={{SozKZ Core TinyLlama 1B Kazakh-Russian v1}},
  author={{Saken Tukenov}},
  year={{2026}},
  url={{https://huggingface.co/{REPO_ID}}}
}}
```
"""


def main():
    log.info("Loading model from %s", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    log.info("Model loaded. Vocab: %d, Device: %s", len(tokenizer), DEVICE)

    # Generate examples
    log.info("Generating examples...")
    examples = {}
    for name, prompt in TEST_PROMPTS.items():
        text = generate(model, tokenizer, prompt)
        examples[name] = text
        log.info("[%s] %s", name, text[:200])

    with open(Path(MODEL_DIR) / "examples.json", "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    # Perplexity
    log.info("Computing perplexity...")
    kaz_texts = [
        "Қазақстан Республикасы — Орталық Азиядағы мемлекет. Астанасы — Астана қаласы.",
        "Қазақ тілі — түркі тілдерінің қыпшақ тобына жататын тіл.",
        "Білім беру жүйесі — мемлекеттік саясаттың маңызды бағыттарының бірі.",
        "Алматы — Қазақстанның ең ірі қаласы және мәдени астанасы.",
        "Қазақстан экономикасы мұнай мен газ өндірісіне негізделген.",
    ]
    rus_texts = [
        "Республика Казахстан — государство в Центральной Азии. Столица — город Астана.",
        "Казахский язык относится к кыпчакской группе тюркских языков.",
        "Система образования является одним из важнейших направлений государственной политики.",
        "Алматы — крупнейший город Казахстана и культурная столица страны.",
        "Экономика Казахстана основана на добыче нефти и газа.",
    ]
    eng_texts = [
        "The Republic of Kazakhstan is a country in Central Asia. Its capital is Astana.",
        "The Kazakh language belongs to the Kipchak group of Turkic languages.",
        "The education system is one of the most important areas of state policy.",
        "Almaty is the largest city in Kazakhstan and the cultural capital of the country.",
    ]

    perplexities = {
        "kaz": compute_perplexity(model, tokenizer, kaz_texts),
        "rus": compute_perplexity(model, tokenizer, rus_texts),
        "eng": compute_perplexity(model, tokenizer, eng_texts),
    }
    log.info("PPL — kaz: %.2f, rus: %.2f, eng: %.2f",
             perplexities["kaz"], perplexities["rus"], perplexities["eng"])

    with open(Path(MODEL_DIR) / "perplexity.json", "w") as f:
        json.dump(perplexities, f, indent=2)

    # Fertility
    fertility = {}
    for lang, text in [("kaz", "Қазақстан Республикасы — Орталық Азиядағы мемлекет"),
                        ("rus", "Республика Казахстан — государство в Центральной Азии"),
                        ("eng", "Republic of Kazakhstan — a state in Central Asia")]:
        toks = tokenizer.encode(text, add_special_tokens=False)
        fertility[lang] = round(len(toks) / len(text.split()), 2)

    # README
    log.info("Creating README...")
    readme = create_readme(examples, perplexities, fertility)
    with open(Path(MODEL_DIR) / "README.md", "w") as f:
        f.write(readme)

    # Upload
    log.info("Uploading to %s...", REPO_ID)
    api = HfApi()
    api.create_repo(REPO_ID, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        ignore_patterns=["checkpoint-*", "training_args.bin"],
    )
    log.info("UPLOADED: https://huggingface.co/%s", REPO_ID)


if __name__ == "__main__":
    main()
