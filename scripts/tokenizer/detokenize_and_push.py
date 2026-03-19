#!/usr/bin/env python3
"""Detokenize the already-pushed tokenized dataset and re-upload as clean text.

Loads saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2 (tokenized blocks),
decodes input_ids back to text, and pushes as clean text dataset.
"""

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

TOKENIZED_REPO = "saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2"
CLEAN_REPO = "saken-tukenov/sozkz-corpus-clean-kk-text-v2"
TOKENIZER_PATH = "./tokenizers/kazakh-bpe-32k"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print(f"Loading tokenized dataset from {TOKENIZED_REPO}...")
    ds = load_dataset(TOKENIZED_REPO)

    splits = {}
    for split_name in ds:
        print(f"Detokenizing {split_name} ({len(ds[split_name])} blocks)...")
        texts = []
        domains = []
        for row in ds[split_name]:
            text = tokenizer.decode(row["input_ids"], skip_special_tokens=True).strip()
            if len(text) > 0:
                texts.append(text)
                domains.append(row.get("domain", "unknown"))

        splits[split_name] = Dataset.from_dict({"text": texts, "domain": domains})
        print(f"  {split_name}: {len(texts)} texts")

    ds_dict = DatasetDict(splits)

    print(f"Pushing to {CLEAN_REPO}...")
    ds_dict.push_to_hub(CLEAN_REPO, private=False)
    print(f"Done! https://huggingface.co/datasets/{CLEAN_REPO}")


if __name__ == "__main__":
    main()
