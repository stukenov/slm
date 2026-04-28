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
  - name: ekitil-core-qwen3-300m-kkru-base-v1
    results: []
---

# EkiTil-300M: Bilingual Kazakh-Russian Language Model

**EkiTil** (Екі Тіл — "Two Languages") is a family of bilingual Kazakh-Russian causal language models trained from scratch.

EkiTil-300M is the mid-size model — a 245.9M parameter Qwen3 model trained on 4.94B tokens (2 epochs) of mixed Kazakh and Russian text.

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | Qwen3ForCausalLM |
| **Parameters** | 245.9M |
| **Hidden size** | 1,024 |
| **Layers** | 16 |
| **Attention heads** | 16 (4 KV heads, GQA 4:1) |
| **Intermediate size** | 2,816 |
| **Vocab size** | 64,000 |
| **Context length** | 2,048 |
| **Precision** | bf16 |
| **Tied embeddings** | Yes |

## Training

| Property | Value |
|----------|-------|
| **Dataset** | [ekitil-corpus-tokenized-kkru-v1](https://huggingface.co/datasets/stukenov/ekitil-corpus-tokenized-kkru-v1) |
| **Tokens** | 4.94B (2 epochs over 2.47B unique) |
| **Chinchilla ratio** | 20.1:1 |
| **Optimizer** | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| **Learning rate** | 3e-4 (cosine decay, 2K warmup) |
| **Batch size** | 8 × 8 grad accum × 2 GPUs = 262K tok/step |
| **Hardware** | 2× NVIDIA H100 80GB HBM3 (DDP) |
| **Training time** | 6.63 hours |
| **Final loss** | 2.93 |
| **Final BPB** | 4.22 |

### Loss Curve

```
Step       Loss    BPB     Epoch
   942     6.27    9.05    0.10  (warmup)
 1,884     4.62    6.67    0.20
 3,768     3.54    5.11    0.40
 5,652     3.31    4.77    0.60
 7,536     3.15    4.54    0.80
 9,420     3.09    4.46    1.00  (epoch 2 start)
11,304     3.01    4.34    1.20
13,188     2.95    4.25    1.40
15,072     2.96    4.27    1.60
16,956     2.93    4.23    1.80
18,849     2.93    4.22    2.00  (final)
```

### Scaling Improvement

| Model | Params | Loss | BPB | Improvement |
|-------|--------|------|-----|-------------|
| EkiTil-123M | 124.7M | 3.07 | 4.44 | baseline |
| **EkiTil-300M** | **245.9M** | **2.93** | **4.22** | **−4.6% loss, −5.0% BPB** |

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

Source: [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) (24.9M documents, sentence-split and language-annotated). Data seen twice during 2-epoch training.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("stukenov/ekitil-core-qwen3-300m-kkru-base-v1")
tokenizer = AutoTokenizer.from_pretrained("stukenov/ekitil-core-qwen3-300m-kkru-base-v1")

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
> **Output:** Қазақстан — Орталық Азиядағы ең ірі, әрі беделді мемлекет. Біздің ел Орталық Азияда тұңғыш рет біздің мемлекетіміздің тәуелсіздігін таныған, әлемдік қауымдастықтағы орны мен ролін жоғары бағалайды.

**Russian (general):**
> **Prompt:** Искусственный интеллект — это
>
> **Output:** Искусственный интеллект — это основа всех наших усилий. Надеемся на то, что наше поколение будет использовать все лучшее для дальнейшего процветания страны.

**Kazakh (weather):**
> **Prompt:** Бүгін ауа райы өте
>
> **Output:** Бүгін ауа райы өте құбылмалы болып, түнде күн жылынады. Алдын ала болжам бойынша, бұл өңірлерде су тасқыны болмайды. Сондықтан, тұрғындар сусыз қалмауы керек.

**Translation (kk→ru):**
> **Prompt:** `<|kk|>` Менің атым Сакен. `<|translate|>` `<|ru|>`
>
> **Output:** После рождения я вышел в другой человек. *(translation quality still limited — parallel data was only 0.3% of corpus)*

## EkiTil Model Family

| Model | Params | Loss | BPB | Tokens | Epochs |
|-------|--------|------|-----|--------|--------|
| [EkiTil-123M](https://huggingface.co/stukenov/ekitil-core-qwen3-123m-kkru-base-v1) | 124.7M | 3.07 | 4.44 | 2.47B | 1.0 |
| **EkiTil-300M** (this) | 245.9M | 2.93 | 4.22 | 4.94B | 2.0 |
| [EkiTil-600M](https://huggingface.co/stukenov/ekitil-core-qwen3-600m-kkru-base-v1) | 673.8M | 3.33 | 4.80 | 7.07B | 2.87 |

## Limitations

- This is a **base model** (not instruction-tuned) — it generates continuations, not answers
- Trained on web-crawled data which may contain biases and inaccuracies
- Context window limited to 2,048 tokens
- 2-epoch training means the model has seen all data twice — slight overfitting possible on rare patterns
- Primarily handles Kazakh and Russian; other languages may produce low-quality output

## Citation

```bibtex
@misc{ekitil2026,
  title={EkiTil: Bilingual Kazakh-Russian Language Models},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/stukenov/ekitil-core-qwen3-300m-kkru-base-v1}
}
```

## License

MIT
