---
language:
  - kk
  - ru
license: mit
tags:
  - causal-lm
  - qwen3
  - kazakh
  - russian
  - bilingual
  - ekitil
  - pretrained
library_name: transformers
pipeline_tag: text-generation
datasets:
  - stukenov/ekitil-corpus-tokenized-kkru-v1
base_model: []
model-index:
  - name: ekitil-core-qwen3-123m-kkru-base-v1
    results: []
---

# EkiTil-123M: Bilingual Kazakh-Russian Language Model

**EkiTil** (Екі Тіл — "Two Languages") is a family of bilingual Kazakh-Russian causal language models trained from scratch.

EkiTil-123M is the smallest model in the family — a 124.7M parameter Qwen3 model trained on 2.47B tokens of mixed Kazakh and Russian text.

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | Qwen3ForCausalLM |
| **Parameters** | 124.7M |
| **Hidden size** | 768 |
| **Layers** | 12 |
| **Attention heads** | 12 (4 KV heads, GQA 3:1) |
| **Intermediate size** | 2,048 |
| **Vocab size** | 64,000 |
| **Context length** | 2,048 |
| **Precision** | bf16 |
| **Tied embeddings** | Yes |

## Training

| Property | Value |
|----------|-------|
| **Dataset** | [ekitil-corpus-tokenized-kkru-v1](https://huggingface.co/datasets/stukenov/ekitil-corpus-tokenized-kkru-v1) |
| **Tokens** | 2.47B (1 epoch) |
| **Chinchilla ratio** | 19.8:1 (optimal) |
| **Optimizer** | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| **Learning rate** | 6e-4 (cosine decay, 2K warmup) |
| **Batch size** | 16 × 8 grad accum = 262K tok/step |
| **Hardware** | 1× NVIDIA H100 80GB HBM3 |
| **Training time** | 3.8 hours |
| **Final loss** | 3.07 |
| **Final BPB** | 4.44 |

### Loss Curve

```
Step      Loss    BPB
   500    7.07    10.20
 1,000    5.48     7.91
 2,000    3.99     5.75   (peak lr)
 4,000    3.40     4.91
 6,000    3.18     4.59
 8,000    3.10     4.47
 9,424    3.07     4.44   (final)
```

## Tokenizer

Custom ByteLevel BPE tokenizer trained on balanced kk+ru corpus:
- **Vocab**: 64,000 tokens
- **Fertility**: 1.56 tokens/word (mixed kk+ru)
- **Special tokens**: `<|kk|>`, `<|ru|>`, `<|translate|>` for language tagging

See [ekitil-vocab-bpe-64k-kkru-v1](https://huggingface.co/stukenov/ekitil-vocab-bpe-64k-kkru-v1) for details.

## Data Composition

| Language | Tokens | Share |
|----------|--------|-------|
| Kazakh | 1.33B | 53.9% |
| Russian | 1.46B | 59.1% |
| Parallel kk↔ru | 7.2M | 0.3% |

Source: [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) (24.9M documents, sentence-split and language-annotated).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("stukenov/ekitil-core-qwen3-123m-kkru-base-v1")
tokenizer = AutoTokenizer.from_pretrained("stukenov/ekitil-core-qwen3-123m-kkru-base-v1")

text = "Қазақстан — Орталық Азиядағы"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Generation Examples

Generated on RTX 3090 with `temperature=0.8, top_p=0.9, repetition_penalty=1.1`:

**Kazakh (geography):**
> **Prompt:** Қазақстан — Орталық Азиядағы
>
> **Output:** Қазақстан — Орталық Азиядағы ең ірі сауда орталығы. Жыл сайын әлемнің 300-ге жуық елімен сауда айналымы артып келеді.

**Russian (general):**
> **Prompt:** Искусственный интеллект — это
>
> **Output:** Искусственный интеллект — это человеческий организм, который он использует в повседневной жизни. У человека есть своя система интеллекта, и каждый человек знает его в своем мире.

**Kazakh (weather):**
> **Prompt:** Бүгін ауа райы өте
>
> **Output:** Бүгін ауа райы өте маңызды. Дегенмен, ауа райы да тұрақсыз болып тұрады. Өйткені ауа температурасы түнде +10-15 градус болады.

**Translation (kk→ru):**
> **Prompt:** `<|kk|>` Менің атым Сакен. `<|translate|>` `<|ru|>`
>
> **Output:** Смотреть *(translation not yet reliable at 123M scale — see 300M for better results)*

## EkiTil Model Family

| Model | Params | Loss | BPB | Tokens | Epochs |
|-------|--------|------|-----|--------|--------|
| **EkiTil-123M** (this) | 124.7M | 3.07 | 4.44 | 2.47B | 1.0 |
| [EkiTil-300M](https://huggingface.co/stukenov/ekitil-core-qwen3-300m-kkru-base-v1) | 245.9M | 2.93 | 4.22 | 4.94B | 2.0 |
| [EkiTil-600M](https://huggingface.co/stukenov/ekitil-core-qwen3-600m-kkru-base-v1) | 673.8M | 3.33 | 4.80 | 7.07B | 2.87 |

## Limitations

- This is a **base model** (not instruction-tuned) — it generates continuations, not answers
- Trained on web-crawled data which may contain biases and inaccuracies
- Context window limited to 2,048 tokens
- Primarily handles Kazakh and Russian; other languages may produce low-quality output

## Citation

```bibtex
@misc{ekitil2026,
  title={EkiTil: Bilingual Kazakh-Russian Language Models},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/stukenov/ekitil-core-qwen3-123m-kkru-base-v1}
}
```

## License

MIT
