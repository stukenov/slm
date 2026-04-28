#!/usr/bin/env python3
"""Prepare clean Kazakh seed sentences for synthetic GEC generation."""

from __future__ import annotations

import argparse
import os
import re

from datasets import load_dataset


CYR_RE = re.compile(r"[\u0400-\u04FF]")


def cyr_ratio(text: str) -> float:
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return 0.0
    return len(CYR_RE.findall(text)) / alpha


def is_good_sentence(text: str) -> bool:
    text = text.strip()
    words = text.split()
    if not (6 <= len(words) <= 30):
        return False
    if not text or not text[0].isupper():
        return False
    if text[-1] not in ".!?":
        return False
    if cyr_ratio(text) < 0.75:
        return False
    bad = ["http", "www", "<ref", "{{", "}}", "ISBN", ".jpg", ".png"]
    if any(x in text for x in bad):
        return False
    return True


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="saken-tukenov/sozkz-corpus-clean-v3")
    parser.add_argument("--column", default="text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=20000)
    parser.add_argument("--max_docs", type=int, default=200000)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split="train", streaming=True)
    seen = set()
    kept = []
    docs = 0

    for row in ds:
        docs += 1
        text = row.get(args.column, "")
        if not isinstance(text, str) or not text.strip():
            continue
        for sent in split_sentences(text):
            if not is_good_sentence(sent):
                continue
            if sent in seen:
                continue
            seen.add(sent)
            kept.append(sent)
            if len(kept) >= args.target:
                break
        if docs % 5000 == 0:
            print(f"docs={docs} kept={len(kept)}", flush=True)
        if len(kept) >= args.target or docs >= args.max_docs:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for sent in kept:
            f.write(sent + "\n")

    print(f"Saved {len(kept)} seeds to {args.output}", flush=True)


if __name__ == "__main__":
    main()
