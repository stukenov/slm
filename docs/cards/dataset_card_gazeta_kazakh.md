---
language:
- kk
- ru
license: apache-2.0
size_categories:
- 10K<n<100K
task_categories:
- summarization
- translation
- text-generation
tags:
- synthetic
- machine-translation
- kazakh
- russian
- news
- summarization
- parallel-corpus
source_datasets:
- IlyaGusev/gazeta
dataset_info:
  features:
  - name: text
    dtype: string
  - name: summary
    dtype: string
  - name: title
    dtype: string
  - name: date
    dtype: string
  - name: url
    dtype: string
  - name: text_kk
    dtype: string
  - name: summary_kk
    dtype: string
  - name: title_kk
    dtype: string
  splits:
  - name: train
    num_examples: 60964
  - name: validation
    num_examples: 6369
  - name: test
    num_examples: 6793
  config_name: default
---

# Gazeta-Kazakh: Synthetic Russian-Kazakh Parallel News Corpus

A synthetically translated **Russian-Kazakh parallel dataset** of news articles with summaries, created by translating the Russian-language [IlyaGusev/gazeta](https://huggingface.co/datasets/IlyaGusev/gazeta) dataset into Kazakh using [deepvk/kazRush-ru-kk](https://huggingface.co/deepvk/kazRush-ru-kk).

## Dataset Description

### Overview

This dataset provides **74,126 Russian news articles with their Kazakh translations** — including article titles, summaries, and full texts. Each record contains the original Russian text alongside its Kazakh translation, forming a parallel corpus suitable for training translation models, summarization systems, and Kazakh language models.

### Source

The original [IlyaGusev/gazeta](https://huggingface.co/datasets/IlyaGusev/gazeta) dataset consists of Russian news articles from Gazeta.ru, a major Russian news website. Each article includes a title, summary, and full text body.

### Translation

All text fields were translated from Russian to Kazakh using [deepvk/kazRush-ru-kk](https://huggingface.co/deepvk/kazRush-ru-kk):

| Property | Value |
|----------|-------|
| Translation model | deepvk/kazRush-ru-kk |
| Model type | T5 (encoder-decoder) |
| Model size | 197M parameters |
| Translation direction | Russian → Kazakh |
| Decoding | Greedy (num_beams=1) |
| Precision | bfloat16 |

**Long text handling**: The T5 model has a ~512 token input limit. For article bodies exceeding this limit, texts were split into sentence-level chunks (max 400 tokens each), translated independently, and reassembled. Titles and summaries generally fit within the 512-token limit and were translated directly.

## Dataset Structure

### Fields

| Field | Language | Description |
|-------|----------|-------------|
| `title` | Russian | Original article headline |
| `title_kk` | Kazakh | Translated article headline |
| `summary` | Russian | Original article summary |
| `summary_kk` | Kazakh | Translated article summary |
| `text` | Russian | Original full article body |
| `text_kk` | Kazakh | Translated full article body |
| `date` | - | Publication date |
| `url` | - | Source URL |

### Splits

| Split | Examples |
|-------|---------|
| train | 60,964 |
| validation | 6,369 |
| test | 6,793 |
| **Total** | **74,126** |

Splits are preserved from the original dataset.

### Data Example

```json
{
  "title": "Налог в бак",
  "title_kk": "Бакқа салынатын салық",
  "summary": "С 2011 года правительство отменяет самый раздражающий граждан налог – транспортный...",
  "summary_kk": "Бірақ автокөлік жинау тоқтамайды – салықты бензин акциздері мен ақылы жолдарға бүркемелеп...",
  "text": "Сегодня транспортный налог начисляется в зависимости от мощности автомобиля...",
  "text_kk": "Сонымен қатар, салықтан босату аймақтық билікке азайтылуы мүмкін...",
  "date": "2010-11-15",
  "url": "https://www.gazeta.ru/..."
}
```

## Usage

### Load the Dataset

```python
from datasets import load_dataset

ds = load_dataset("saken-tukenov/sozkz-corpus-raw-kk-gazeta-v1")

# Access a specific split
train = ds["train"]
print(f"Train: {len(train)} examples")

# View an example
example = train[0]
print(f"Title (ru): {example['title']}")
print(f"Title (kk): {example['title_kk']}")
```

### Use as Parallel Corpus (Translation)

```python
# Extract Russian-Kazakh sentence pairs
for example in ds["train"]:
    ru_text = example["summary"]
    kk_text = example["summary_kk"]
    # Use for translation model training...
```

### Use for Kazakh Summarization

```python
# Kazakh article → Kazakh summary
for example in ds["train"]:
    article = example["text_kk"]
    summary = example["summary_kk"]
    # Fine-tune a summarization model...
```

### Use for Language Model Training

```python
# Kazakh text corpus
kazakh_texts = [example["text_kk"] for example in ds["train"]]
```

## Translation Examples

### Example 1: News

| | Text |
|-|------|
| **Title (ru)** | Секс, наркотики и темный зал |
| **Title (kk)** | Секс, есірткі және қараңғы зал |
| **Summary (ru)** | Британские затворники, московские модники, бразильский фанк и исламский панк... |
| **Summary (kk)** | Британдық бекіткіштер, мәскеулік сәнгерлер, бразилиялық фанк және исламдық панк... |

### Example 2: Politics

| | Text |
|-|------|
| **Title (ru)** | Осудить и отпустить |
| **Title (kk)** | Соттау және босату |
| **Summary (ru)** | Совбез ООН собрался на экстренное совещание для обсуждения захвата «Флотилии свободы» Израилем. |
| **Summary (kk)** | БҰҰ Қауіпсіздік кеңесі Израильдің «Бостандық флотилиясын» басып алуын талқылау үшін шұғыл жиналыс өткізді. |

### Example 3: Economy

| | Text |
|-|------|
| **Title (ru)** | Тарифы инфляцию не остановят |
| **Title (kk)** | Инфляция тарифтері тоқтамайды |
| **Summary (ru)** | Правительство хочет сдержать рост тарифов естественных монополий в следующем году. |
| **Summary (kk)** | Үкімет келесі жылы табиғи монополиялар тарифтерінің өсуін тоқтатқысы келеді. |

### Example 4: Disaster

| | Text |
|-|------|
| **Title (ru)** | «Агата» открыла страшный сезон |
| **Title (kk)** | «Агата» қорқынышты маусымды ашты |
| **Summary (ru)** | Ураган «Агата» в Центральной Америке унес жизни 146 человек. |
| **Summary (kk)** | Орталық Америкадағы «Агата» дауылы 146 адамды қазаға ұшыратты. |

## Limitations and Biases

### Translation Quality

- **Synthetic translation**: All Kazakh text is machine-translated, not human-written. Quality is limited by the capabilities of the kazRush model.
- **Chunking artifacts**: Long articles are split into chunks for translation. This may cause minor coherence issues at chunk boundaries.
- **Domain mismatch**: The kazRush model may not be optimized for all news domains (sports, science, culture, etc.).
- **Named entities**: Proper nouns and named entities may be transliterated inconsistently or incorrectly.
- **Terminology**: Domain-specific terminology (legal, medical, financial) may not be accurately translated.

### Original Dataset Biases

- **Temporal bias**: Articles are from a specific time period and reflect events of that era.
- **Source bias**: All articles come from a single Russian news source (Gazeta.ru).
- **Geographic bias**: Coverage is Russia-centric.

### Recommended Use

This dataset is best suited for:
- Pre-training or fine-tuning Kazakh language models (where perfect translation quality is less critical)
- Building prototypes of Kazakh NLP systems
- Research on synthetic data for low-resource languages
- Cross-lingual transfer learning experiments

This dataset should **not** be used for:
- Evaluating translation quality (it is synthetic, not gold-standard)
- Factual information retrieval (content reflects Russian news)
- Training high-stakes NLP systems without additional human validation

## Creation Details

### Pipeline

1. Load `IlyaGusev/gazeta` from HuggingFace
2. For each article, translate `title`, `summary`, and `text` (ru → kk)
3. Long texts (>400 tokens) are split by sentences, translated in batches, and reassembled
4. Results saved as parquet checkpoints every 1,000 samples
5. Merge checkpoints and upload to HuggingFace

### Infrastructure

| Component | Details |
|-----------|---------|
| Hardware | 2x NVIDIA A10 (23GB each) |
| Precision | bfloat16 |
| Batch size | 16 (rows) x 64 (chunks) |
| Translation speed | ~4.6 samples/sec (combined) |
| Estimated total time | ~4.5 hours |

### Reproducibility

The translation script is available at: [scripts/translate_gazeta.py](https://github.com/sakentukenov/slm/blob/main/scripts/translate_gazeta.py)

```bash
# Reproduce on 2 GPUs
python scripts/translate_gazeta.py --gpu-id 0 --split-id 0 --num-splits 2 \
  --bf16 --num-beams 1 --batch-size 16 --chunk-batch-size 64 &
python scripts/translate_gazeta.py --gpu-id 1 --split-id 1 --num-splits 2 \
  --bf16 --num-beams 1 --batch-size 16 --chunk-batch-size 64 &
```

## Citation

```bibtex
@misc{tukenov2026gazetakazakh,
  title={Gazeta-Kazakh: A Synthetic Russian-Kazakh Parallel News Corpus},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-raw-kk-gazeta-v1}
}
```

If you use this dataset, please also cite the original:

```bibtex
@inproceedings{gusev2020dataset,
  title={Dataset for Automatic Summarization of Russian News},
  author={Gusev, Ilya},
  booktitle={Artificial Intelligence and Natural Language},
  pages={122--134},
  year={2020},
  publisher={Springer}
}
```

And the translation model:

```bibtex
@misc{deepvk2024kazrush,
  title={kazRush: Russian-Kazakh Translation Model},
  author={DeepVK},
  year={2024},
  url={https://huggingface.co/deepvk/kazRush-ru-kk}
}
```

## License

Apache 2.0 (same as the original dataset and translation model)
