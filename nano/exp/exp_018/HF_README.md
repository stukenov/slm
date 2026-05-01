---
language:
  - kk
license: mit
tags:
  - tokenizer
  - bpe
  - morpheme-aware
  - kazakh
  - agglutinative
  - byte-level-bpe
datasets:
  - stukenov/ekitil-corpus-annotated-kk-v1
pipeline_tag: token-classification
library_name: tokenizers
vocab_size: 256000
---

# sozkz-morphbpe-256k-kk-v1

A morpheme-aware byte-level BPE tokenizer for Kazakh with a 256K vocabulary. Merges never cross morpheme boundaries, so every token is a linguistically meaningful unit.

## Quick Start

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stukenov/sozkz-morphbpe-256k-kk-v1")
tokens = tokenizer.tokenize("Қазақстанның университеттерде оқушылары")
```

## Why Morpheme-Aware BPE?

Kazakh is agglutinative. A single surface form packs multiple morphemes into one whitespace-delimited word:

| Word | Morphemes | Meaning |
|---|---|---|
| үйлерімізде | үй · лер · іміз · де | in our houses |
| Қазақстанның | Қазақстан · ның | of Kazakhstan |
| оқушылар | оқушы · лар | students |
| университеттерде | университеттер · де | at universities |

Standard BPE is blind to these boundaries. It merges frequent byte pairs regardless of morphological structure, producing tokens like `ерім` that span the stem-suffix boundary and carry no meaning on their own. This wastes vocabulary capacity on linguistically arbitrary fragments.

Morpheme-aware BPE constrains merges to operate within morpheme spans. The result: higher coverage per token, better generalization to unseen word forms, and a vocabulary that reflects the actual structure of the language.

## Method

We follow the HyperCLOVA X approach (Yoo et al., 2024):

1. **Morpheme segmentation.** A BiLSTM neural model from QazCorpora segments each word into morphemes. The ASCII Unit Separator (`\x1F`) is inserted at every boundary.

2. **Constrained BPE training.** The pre-tokenizer splits on `\x1F` (consuming the separator), then applies ByteLevel encoding. BPE merges proceed normally but cannot cross the separator — each merge is confined to a single morpheme span.

3. **Inference.** At inference time no segmentation is needed. The trained tokenizer operates directly on raw text; morpheme awareness is baked into the merge table.

## Training Details

| Parameter | Value |
|---|---|
| Vocab size | 256,000 |
| Algorithm | Byte-level BPE (HuggingFace `tokenizers`) |
| Pre-tokenizer | Split(`\x1F`, removed) → ByteLevel |
| Corpus | `stukenov/ekitil-corpus-annotated-kk-v1` |
| Corpus size | 55.5M documents, filtered for Kazakh (confidence >= 0.95) |
| Segmented corpus | `stukenov/sozkz-corpus-segmented-kk-v1` (12 parquet shards) |
| Special tokens | `<\|endoftext\|>`, `<\|padding\|>`, `<\|startoftext\|>` |
| Max length | 4,096 |
| Infrastructure | AWS EC2 c7i.4xlarge (16 vCPU, 32 GB RAM), CPU-only |
| Previous version | [`sozkz-morphbpe-100k-kk-v1`](https://huggingface.co/stukenov/sozkz-morphbpe-100k-kk-v1) (100K vocab) |

## Intended Use

- Pre-training and fine-tuning language models for Kazakh
- Any NLP pipeline that benefits from morphologically coherent subword units
- Bilingual or multilingual models where Kazakh tokenization quality matters

## Limitations

- Morpheme segmentation quality depends on the BiLSTM model from QazCorpora, which may not handle loanwords, code-mixed text, or rare dialectal forms well.
- Optimized for Kazakh. Other Turkic languages share agglutinative structure but would need their own segmentation models.
- The 256K vocabulary is large. For parameter-constrained models, consider the 100K variant.

## Citation

If you use this tokenizer, please cite the HyperCLOVA X paper that introduced the morpheme-aware BPE method:

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
