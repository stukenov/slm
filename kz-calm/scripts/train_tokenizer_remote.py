#!/usr/bin/env python3
"""Train SentencePiece tokenizer via DuckDB (fast remote parquet column pruning)."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset", default="stukenov/kzcalm-tts-kk-v1")
    parser.add_argument("--hf-repo", default="stukenov/kzcalm-sp-tokenizer-4k-kk-v1")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--config", help="Ignored")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import duckdb
    import sentencepiece as spm
    from huggingface_hub import HfApi

    tmpdir = Path(tempfile.mkdtemp())
    texts_path = tmpdir / "texts.txt"
    model_prefix = str(tmpdir / "kz_tts_sp")

    # Step 1: Extract texts via DuckDB (column pruning — only reads text, not audio)
    logger.info("Step 1: Extracting texts via DuckDB...")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # Build explicit list of parquet shard URLs
    urls = [f"https://huggingface.co/datasets/{args.hf_dataset}/resolve/main/data/train-{i:05d}-of-00176.parquet" for i in range(176)]
    url_list = ", ".join([f"'{u}'" for u in urls])
    query = f"SELECT text FROM read_parquet([{url_list}]) WHERE text IS NOT NULL AND text != ''"

    # Write directly to file
    result = con.execute(query)
    count = 0
    with open(texts_path, "w", encoding="utf-8") as f:
        while True:
            batch = result.fetchmany(10000)
            if not batch:
                break
            for row in batch:
                t = row[0].strip()
                if t:
                    f.write(t + "\n")
                    count += 1
            if count % 50000 < 10000:
                logger.info(f"  {count} texts...")

    con.close()
    logger.info(f"Extracted {count} texts")

    # Step 2: Train SentencePiece
    logger.info(f"Step 2: Training SentencePiece (vocab={args.vocab_size})...")
    spm.SentencePieceTrainer.Train(
        input=str(texts_path),
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        num_threads=os.cpu_count() or 4,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )
    logger.info("SentencePiece trained")

    # Step 3: Push to HF
    logger.info(f"Step 3: Pushing to {args.hf_repo}...")
    api = HfApi()
    api.create_repo(args.hf_repo, exist_ok=True)
    for ext in [".model", ".vocab"]:
        p = f"{model_prefix}{ext}"
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"tokenizer{ext}", repo_id=args.hf_repo)
            logger.info(f"  uploaded tokenizer{ext}")

    readme = f"""---
language:
- kk
license: apache-2.0
tags:
- tokenizer
- sentencepiece
- kazakh
- tts
---

# KZ-CALM SentencePiece Tokenizer ({args.vocab_size} vocab)

BPE tokenizer trained on {count} Kazakh TTS utterances from `{args.hf_dataset}`.

Special tokens: pad=0, bos=1, eos=2, unk=3

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
print(sp.Encode("Сәлем, әлем!"))
```
"""
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=args.hf_repo)
    logger.info("Done!")


if __name__ == "__main__":
    main()
