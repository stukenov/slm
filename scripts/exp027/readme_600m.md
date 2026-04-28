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
  - name: ekitil-core-qwen3-600m-kkru-base-v1
    results: []
---

# EkiTil-600M: Bilingual Kazakh-Russian Language Model

**EkiTil** (Екі Тіл — "Two Languages") is a family of bilingual Kazakh-Russian causal language models trained from scratch.

EkiTil-600M is the largest model in the family — a 673.8M parameter Qwen3 model trained on 7.07B tokens (~2.87 epochs) of mixed Kazakh and Russian text.

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | Qwen3ForCausalLM |
| **Parameters** | 673.8M |
| **Hidden size** | 1,280 |
| **Layers** | 28 |
| **Attention heads** | 20 (4 KV heads, GQA 5:1) |
| **Head dim** | 64 |
| **Intermediate size** | 4,480 |
| **Vocab size** | 64,000 |
| **Context length** | 2,048 |
| **Precision** | bf16 |
| **Tied embeddings** | Yes |

## Training

| Property | Value |
|----------|-------|
| **Dataset** | [ekitil-corpus-tokenized-kkru-v1](https://huggingface.co/datasets/stukenov/ekitil-corpus-tokenized-kkru-v1) |
| **Tokens** | 7.07B (2.87 epochs over 2.47B unique) |
| **Chinchilla ratio** | 10.5:1 |
| **Optimizer** | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| **Learning rate** | 2e-4 (cosine decay, 2K warmup) |
| **Batch size** | 12 × 8 grad accum × 8 GPUs = 1,572K tok/step |
| **Hardware** | 8× NVIDIA H100 80GB HBM3 (DDP) |
| **Training time** | 5.8 hours |
| **Final loss** | 3.33 |
| **Final BPB** | 4.80 |

### Loss Curve

```
Step      Loss    BPB     Epoch
   500    8.18    11.80   0.32  (warmup)
 1,000    6.07     8.76   0.64
 1,500    5.18     7.48   0.96
 2,000    4.54     6.55   1.27  (warmup end)
 2,500    4.02     5.80   1.59
 3,000    3.71     5.35   1.91
 3,500    3.55     5.12   2.23
 4,000    3.39     4.89   2.55
 4,500    3.33     4.80   2.87  (final)
```

### Scaling Across the Family

| Model | Params | Loss | BPB | Tokens | Epochs |
|-------|--------|------|-----|--------|--------|
| EkiTil-123M | 124.7M | 3.07 | 4.44 | 2.47B | 1.0 |
| EkiTil-300M | 245.9M | 2.93 | 4.22 | 4.94B | 2.0 |
| **EkiTil-600M** | **673.8M** | **3.33** | **4.80** | **7.07B** | **2.87** |

> **Note:** EkiTil-600M has higher loss than smaller models because it is significantly undertrained (Chinchilla ratio 10.5:1 vs optimal ~20:1). The model would benefit from more data or more training epochs. Despite higher loss, generation quality is noticeably better due to the larger capacity.

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

Source: [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) (24.9M documents, sentence-split and language-annotated). Data seen ~2.87 times during training.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "stukenov/ekitil-core-qwen3-600m-kkru-base-v1",
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("stukenov/ekitil-core-qwen3-600m-kkru-base-v1")

text = "Қазақстан — бұл"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Generation Examples

Generated on H100 with `temperature=0.8, top_p=0.9, repetition_penalty=1.1`:

**Kazakh (general):**
> **Prompt:** Қазақстан — бұл
>
> **Output:** Қазақстан — бұл кезде осы, менде бұл нәрсе емес, ол менің ойымша, мен сол сияқты басқа да адамдарым емес. Бізде ешкім жоқ, олар да жоқ, ал енді олардың бәрі мүмкін.

**Russian (economics):**
> **Prompt:** Россия и Казахстан
>
> **Output:** Россия и Казахстан, который является единственным источником экономического роста. С его помощью будет производиться более 100% поставок товаров. Если говорить о Казахстане, то с точки зрения импорта, импорт из стран СНГ будет очень высоким.

**Kazakh (language):**
> **Prompt:** Қазақ тілі — бұл
>
> **Output:** Қазақ тілі — бұл ғылым. «Білім» — халықаралық ақпараттық технологиялар орталығы. «Назарбаев Зияткерлік мектептері» — қазақ тілінде оқытатын және оқытуға арналған электрондық оқулықтар орталығы.

**Kazakh (education):**
> **Prompt:** Білім алу өте маңызды
>
> **Output:** Білім алу өте маңызды. Бұл — ең алдымен, адам баласының рухани байлығы мен жан-жақты дамуы үшін үлкен жауапкершілік жүктейді. Сондықтан да оның әрбір жеке адамның өмір сүруі — біздің бүгінгі басты міндет.

**Kazakh (folklore):**
> **Prompt:** Ертеде бір патша болыпты
>
> **Output:** Ертеде бір патша болыпты. Оның баласы Әбілда болған. Сонда оның атасы Қарға мен шешесі — Мұхаммед (с.ғ.с.) болатын. Ол кезде осы хан, әрі сұлтандары болып, сол әулеттен шыққан.

## Checkpoints

Training checkpoints are available at [stukenov/ekitil-core-qwen3-600m-kkru-checkpoints](https://huggingface.co/stukenov/ekitil-core-qwen3-600m-kkru-checkpoints) (saved every 500 steps with model weights + optimizer state).

## EkiTil Model Family

| Model | Params | Loss | BPB | Tokens | Epochs | Time |
|-------|--------|------|-----|--------|--------|------|
| [EkiTil-123M](https://huggingface.co/stukenov/ekitil-core-qwen3-123m-kkru-base-v1) | 124.7M | 3.07 | 4.44 | 2.47B | 1.0 | 3.8h |
| [EkiTil-300M](https://huggingface.co/stukenov/ekitil-core-qwen3-300m-kkru-base-v1) | 245.9M | 2.93 | 4.22 | 4.94B | 2.0 | 6.6h |
| **EkiTil-600M** (this) | 673.8M | 3.33 | 4.80 | 7.07B | 2.87 | 5.8h |

## Limitations

- This is a **base model** (not instruction-tuned) — it generates continuations, not answers
- Significantly undertrained (Chinchilla-optimal would be ~13.5B tokens for this size)
- Trained on web-crawled data which may contain biases and inaccuracies
- Context window limited to 2,048 tokens
- ~2.87 epochs means the model has seen all data nearly 3 times — some overfitting possible on rare patterns
- Primarily handles Kazakh and Russian; other languages may produce low-quality output

## Citation

```bibtex
@misc{ekitil2026,
  title={EkiTil: Bilingual Kazakh-Russian Language Models},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/stukenov/ekitil-core-qwen3-600m-kkru-base-v1}
}
```

## License

MIT
