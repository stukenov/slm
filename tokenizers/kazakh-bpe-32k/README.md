---
language:
  - kk
license: apache-2.0
library_name: transformers
tags:
  - tokenizer
  - bpe
  - kazakh
  - byte-level-bpe
pipeline_tag: text-generation
---

# Kazakh BPE Tokenizer (32K)

A ByteLevel BPE tokenizer trained from scratch on a large-scale Kazakh text corpus. Designed as a general-purpose subword tokenizer for Kazakh language models.

## Overview

| Property | Value |
|---|---|
| Vocabulary size | 32,000 |
| Algorithm | ByteLevel BPE |
| Training data | [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) (23.6M samples) |
| Special tokens | `<\|endoftext\|>` (bos/eos), `<\|padding\|>` (pad) |
| Max length | 1024 (configurable) |

## Why a Custom Kazakh Tokenizer?

General-purpose multilingual tokenizers (GPT-2, LLaMA, etc.) allocate only a small fraction of their vocabulary to Kazakh. This leads to:

- **Poor compression**: Kazakh text is split into many small, often meaningless byte-level fragments
- **Longer sequences**: More tokens per sentence means slower inference and higher memory usage
- **Wasted capacity**: Most of the vocabulary is occupied by tokens from other languages

This tokenizer was trained exclusively on Kazakh text, so every token in its 32K vocabulary is useful for representing the Kazakh language. This results in significantly better compression ratios and more efficient model training and inference.

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1")

text = "Қазақстан — Орталық Азиядағы мемлекет."
tokens = tokenizer.encode(text)
print(tokens)
# Decode back
print(tokenizer.decode(tokens))

# Batch encoding
batch = tokenizer(
    ["Сәлем, әлем!", "Қазақ тілі — түркі тілдерінің бірі."],
    padding=True,
    return_tensors="pt",
)
print(batch)
```

## Special Tokens

| Token | ID | Role |
|---|---|---|
| `<\|endoftext\|>` | 0 | BOS / EOS token |
| `<\|padding\|>` | 1 | Padding token |

The tokenizer does **not** have an `unk_token` — ByteLevel BPE can encode any UTF-8 input by falling back to individual bytes.

## Training Details

The tokenizer was trained using the HuggingFace `tokenizers` library with the following configuration:

- **Algorithm**: ByteLevel BPE
- **Vocab size**: 32,000
- **Training corpus**: Full training split of `kz-transformers/multidomain-kazakh-dataset` (23.6M samples covering news, literature, Wikipedia, legal texts, and more)
- **Pre-tokenizer**: ByteLevel (with regex splitting, no prefix space)
- **Normalization**: None (raw text preserved)

## Integration with Models

This tokenizer is designed to be used with GPT-2 style causal language models. To use it with a model:

```python
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

tokenizer = AutoTokenizer.from_pretrained("saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1")

# Initialize a model with matching vocab size
config = GPT2Config(vocab_size=tokenizer.vocab_size)
model = GPT2LMHeadModel(config)

# Tokenize and generate
inputs = tokenizer("Қазақстан", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## License

Apache 2.0
