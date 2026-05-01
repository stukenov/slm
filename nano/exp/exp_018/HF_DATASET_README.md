---
language:
  - kk
license: mit
tags:
  - morpheme-segmentation
  - kazakh
  - agglutinative
  - nlp
  - corpus
dataset_info:
  features:
    - name: text_segmented
      dtype: string
  splits:
    - name: train
      num_examples: 55539970
size_categories:
  - 10M<n<100M
source_datasets:
  - stukenov/ekitil-corpus-annotated-kk-v1
---

# sozkz-corpus-segmented-kk-v1

55.5M Kazakh texts with morpheme boundaries marked by a BiLSTM neural segmenter. Built for training morpheme-aware tokenizers.

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("stukenov/sozkz-corpus-segmented-kk-v1", split="train", streaming=True)
sample = next(iter(ds))["text_segmented"]
# "Қазақстан\x1fның экономика\x1fсы тұрақты дам\x1fып кел\x1fеді."
```

The morpheme boundary marker is `\x1F` (ASCII Unit Separator). To visualize:

```python
print(sample.replace("\x1f", " · "))
# "Қазақстан · ның экономика · сы тұрақты дам · ып кел · еді."
```

## What's Inside

| Property | Value |
|---|---|
| Rows | 55,539,970 |
| Column | `text_segmented` |
| Boundary marker | `\x1F` (ASCII 31, Unit Separator) |
| Format | 12 parquet shards |
| Source | `stukenov/ekitil-corpus-annotated-kk-v1` (filtered: `detected_lang=kk`, confidence >= 0.95) |

## Segmentation Model

Morpheme boundaries are predicted by a BiLSTM model trained on the QazCorpora dataset with BIO tagging (`B-ROOT`, `I-ROOT`, `B-SUFFIX`, `I-SUFFIX`). The model operates at the character level — each character is classified, and `B-SUFFIX` tags mark where a new morpheme begins.

### Examples

| Original | Segmented | Gloss |
|---|---|---|
| үйлерімізде | үй · лер · іміз · де | house · PL · 1PL.POSS · LOC |
| Қазақстанның | Қазақстан · ның | Kazakhstan · GEN |
| оқушылар | оқушы · лар | student · PL |
| университеттерде | университеттер · де | universities · LOC |
| дайындалуда | дайындал · у · да | prepare · NMLZ · LOC |

### Performance

- **Cache hit rate:** 96.3% on this corpus (500K unique word cache)
- **Processing speed:** ~5.5K documents/sec on 16 vCPU (c7i.4xlarge)
- **Total segmentation time:** ~2.5 hours for 55.5M documents

## Intended Use

- Training morpheme-aware BPE tokenizers (see [`sozkz-morphbpe-256k-kk-v1`](https://huggingface.co/stukenov/sozkz-morphbpe-256k-kk-v1))
- Morphological analysis and linguistic research on Kazakh
- Any task that benefits from pre-segmented agglutinative text

## Limitations

- Segmentation quality depends on the BiLSTM model, which may produce incorrect boundaries for loanwords, proper nouns, and rare word forms.
- Only Kazakh text is included. Russian and other languages from the source corpus were filtered out.
- Some splits are linguistically debatable (e.g., `математи · ка`) — the model occasionally breaks stems where native speakers wouldn't.

## Citation

The morpheme-aware BPE method follows HyperCLOVA X:

```bibtex
@article{yoo2024hyperclova,
  title={HyperCLOVA X Technical Report},
  author={Yoo, Kang Min and Han, Jaegeun and In, Sookyo and Jeon, Heewon and Jeong, Jisu and Kang, Jaewook and Kim, Hyunwook and Kim, Kyung-Min and Kim, Munhyong and Kim, Sungju and others},
  journal={arXiv preprint arXiv:2404.01954},
  year={2024}
}
```

## Author

Saken Tukenov ([@stukenov](https://huggingface.co/stukenov))

## License

MIT
