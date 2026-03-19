---
language:
- kk
license: apache-2.0
tags:
- llama
- kazakh
- instruct
- sft
- text-generation
datasets:
- saken-tukenov/sozkz-corpus-synthetic-kk-instruct-v1
base_model: saken-tukenov/sozkz-core-llama-150m-kk-base-v1
pipeline_tag: text-generation
---

# SozKZ Core Llama 150M Kazakh Instruct v1

A 152M-parameter instruction-following language model for Kazakh, fine-tuned from the [base model](https://huggingface.co/saken-tukenov/sozkz-core-llama-150m-kk-base-v1) using SFT (Supervised Fine-Tuning) with loss masking on prompt tokens.

## Model Details

| | |
|---|---|
| Architecture | LlamaForCausalLM |
| Parameters | 152M |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Vocab size | 50,000 |
| Max length | 512 |
| Base model | [sozkz-core-llama-150m-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-llama-150m-kk-base-v1) |
| Tokenizer | [sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1) |

## Chat Template (Alpaca-Kazakh)

The model uses a Kazakh-language Alpaca-style prompt format:

```
### Нұсқаулық:
{instruction}

### Кіріс:
{input}

### Жауап:
{output}
```

The `### Кіріс:` section is omitted when `input` is empty.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "saken-tukenov/sozkz-core-llama-150m-kk-instruct-v1",
    dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(
    "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1"
)
model.eval()

prompt = "### Нұсқаулық:\nНаурыз мейрамы туралы қысқаша айтып бер.\n\n### Жауап:\n"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
    )
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Real Examples

All outputs below are actual model generations (temperature=0.7, top_p=0.9, repetition_penalty=1.2).

### History

> **Нұсқаулық:** Қазақстанның астанасы туралы айтып бер.
>
> **Жауап:** Астананың туған күні - бұл әлемдегі ең үлкен және танымал қалалардың бірі, оның тұрғындары миллионнан асады. Бұл таңқаларлық емес қала, көптеген көрікті жерлер мен керемет мәдени-тарихи ескерткіштер бар.

### Culture

> **Нұсқаулық:** Наурыз мейрамы туралы қысқаша айтып бер.
>
> **Жауап:** Көктемде күн жылынып, жер жарқырап тұрар еді. Дәстүрлі түрде Наурызды атап өту - бұл көптеген дәстүрлер мен әдет-ғұрыптар бар мереке. Бұл адамдар бір-біріне және табиғаттың сұлулығына байланысты барлық нәрсе үшін ризашылық білдіретін керемет уақыт болды.

### Technology

> **Нұсқаулық:** Жасанды интеллект дегеніміз не?
>
> **Жауап:** AI - бұл адамның араласуынсыз машиналарды оқыту, мысалы, жасанды нейрондық желілер және автоматтандырылған жүйелер сияқты тапсырмалар.

### Health

> **Нұсқаулық:** Дұрыс тамақтану ережелері қандай?
>
> **Жауап:** Салауатты өмір салтын ұстану және дұрыс тамақтануды жеу үшін теңдестірілген диетаны, мысалы, ақуыздар мен пайдалы майлар сияқты қоректік заттарды жеу маңызды.

### Creative Writing

> **Нұсқаулық:** Көктем туралы қысқа өлең жаз.
>
> **Жауап:**
> Жаздың басы жазғы күн,
> Мені және менің жүрегімде бір сәт;
> Жапырақтары түсіп кетті ме?
> Күзгі салқындық пен үмітсіздік сезімі;
> Ешқандай бүршіктердің иісі жоқ, бірақ мен барлығым!

### Economics

> **Нұсқаулық:** Инфляция дегеніміз не?
>
> **Жауап:** Ағымдағы бағалар мен тауарлар бағасы арасындағы айырмашылық - бұл инфляция (ADR) деп аталатын баға деңгейінің жоғарылауы. ТБИ-дің өзі сатып алу қабілетінің төмендеуіне әкеледі, өйткені тауарлардың құны өскен кезде бағалардың өсуі мүмкін емес.

### Literature

> **Нұсқаулық:** Абай Құнанбаев туралы айтып бер.
>
> **Жауап:** Ақынның өмірі мен шығармашылығы біздің дәуіріміздің ортасына дейін болды, ал оның өлеңі және өлеңдері бізге әлі күнге дейін белгілі. Оның өлеңдері «Көңілім», «Өткендер» сияқты өлеңдерден тұрады; «Сөніп-жану» - бұл өлеңнің негізгі тақырыбы.

### Advice

> **Нұсқаулық:** Қазақ тілін үйренуге кеңес бер.
>
> **Жауап:** Мен қазір қазақ тілінде сөйлеу үшін «Басты бет» бағдарламасын іске қосамын, бұл сіздің тіліңіз туралы көбірек білгіңіз келетін кез келген адамға мүмкіндік береді. Бұл бағдарламада сіз білетін және қызықтыратын барлық сөздер мен сөз тіркестерін білуге көмектесетін сұрақтар қоюға болады.

## Benchmarks

### kk-socio-cultural-bench-mc

Evaluated on [kz-transformers/kk-socio-cultural-bench-mc](https://huggingface.co/datasets/kz-transformers/kk-socio-cultural-bench-mc) — 7,111 multiple-choice questions about Kazakh culture, history, traditions, and society. Scoring: highest logit among A/B/C/D tokens.

**Overall accuracy: 10.4%** (742/7111) | Random baseline: 25.0%

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Cinema | 15 | 112 | 13.4% |
| Literature: Poetry and Prose | 67 | 501 | 13.4% |
| Traditional Clothing | 28 | 224 | 12.5% |
| Cuisine and Beverages | 35 | 286 | 12.2% |
| Proverbs, Sayings, Mythology | 58 | 499 | 11.6% |
| Song Lyrics | 58 | 521 | 11.1% |
| Social Relationships | 45 | 406 | 11.1% |
| Agriculture and Animal Husbandry | 63 | 579 | 10.9% |
| Politics and Social Stratification | 47 | 453 | 10.4% |
| Sports and Games | 22 | 212 | 10.4% |
| Humor | 20 | 196 | 10.2% |
| History | 111 | 1103 | 10.1% |
| Onomatopoeia and Naming | 61 | 621 | 9.8% |
| Architecture and Housing Elements | 21 | 249 | 8.4% |
| Arts and Crafts, Performing Arts | 30 | 370 | 8.1% |
| Musical Instruments | 23 | 283 | 8.1% |
| Traditions | 31 | 439 | 7.1% |

> **Note:** Performance below random baseline is expected for a 152M model on culturally specific knowledge tasks. This benchmark requires deep factual knowledge that small models struggle to memorize.

### Kaz-Offline-Arena

Evaluated on [Kaz-Offline-Arena](https://github.com/horde-research/Kaz-Offline-Arena) — open-ended QA benchmark with LLM-judge scoring (GPT). 500 questions across 5 types, scored 0-10.

**Overall average score: 0.48 / 10**

| Question Type | Avg Score | N |
|---------------|-----------|---|
| WHAT_QS | 0.67 | 100 |
| HOW_QS | 0.59 | 100 |
| DESCRIBE_QS | 0.42 | 100 |
| WHY_QS | 0.38 | 100 |
| ANALYZE_QS | 0.36 | 100 |

Average output length: 24.7 tokens. Very low scores reflect the model's limited capacity (152M params) for open-ended generation requiring reasoning and factual knowledge.

## Limitations

- **152M parameters** — this is a small model; responses can be inaccurate or shallow
- Weak at math reasoning and translation tasks
- Prone to factual errors (hallucination)
- Kazakh language only
- 512 token context limit
- Trained on only ~49K instruction examples

## Training Details

| | |
|---|---|
| Dataset | [sozkz-corpus-synthetic-kk-instruct-v1](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-synthetic-kk-instruct-v1) |
| Source | AmanMussa/kazakh-instruction-v2 (49,543 examples) |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 128 (16 x 4 accum x 2 GPU) |
| Steps | 1,152 |
| Eval loss | 2.918 |
| GPU | 2x RTX A5000 |
| Training time | ~12 min |
| Loss masking | Prompt tokens masked (loss on completions only) |
| Optimizer | AdamW |
| Scheduler | Cosine with 3% warmup |
| Precision | bf16 |

## Project

This model is part of the [SozKZ](https://github.com/saken-tukenov/slm) project — research on building small language models for the Kazakh language.

## License

Apache 2.0
