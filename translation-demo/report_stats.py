#!/usr/bin/env python3
"""Generate stats report for the translated dataset."""

from huggingface_hub import HfApi
import pyarrow.parquet as pq

REPO = "saken-tukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1"

def main():
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    print(f"Shards: {len(files)}")

    total_rows = 0
    total_sents = 0
    total_words_en = 0
    total_words_kk = 0
    total_chars_en = 0
    total_chars_kk = 0
    samples_en = []
    samples_kk = []
    shard_stats = []

    for i, f in enumerate(files):
        print(f"Processing {f}...", flush=True)
        local = api.hf_hub_download(REPO, f, repo_type="dataset")
        pf = pq.ParquetFile(local)
        n = pf.metadata.num_rows
        total_rows += n

        shard_words_en = 0
        shard_sents = 0

        for batch in pf.iter_batches(batch_size=100000, columns=["text_en", "text_kk", "num_sentences"]):
            sents = batch["num_sentences"].to_pylist()
            shard_sents += sum(s for s in sents if s is not None)
            ens = batch["text_en"].to_pylist()
            kks = batch["text_kk"].to_pylist()
            for e in ens:
                if e:
                    w = len(e.split())
                    shard_words_en += w
                    total_chars_en += len(e)
            for k in kks:
                if k:
                    total_words_kk += len(k.split())
                    total_chars_kk += len(k)

        total_sents += shard_sents
        total_words_en += shard_words_en
        shard_stats.append((f, n, shard_sents, shard_words_en))
        print(f"  {n} rows, {shard_sents} sents, {shard_words_en} words_en", flush=True)

        # Samples from first shard
        if i == 0:
            first = next(pf.iter_batches(batch_size=5, columns=["text_en", "text_kk"]))
            samples_en = [t[:500] if t else "" for t in first["text_en"].to_pylist()]
            samples_kk = [t[:500] if t else "" for t in first["text_kk"].to_pylist()]

    print(f"\n{'='*60}")
    print(f"REPORT: {REPO}")
    print(f"{'='*60}")
    print(f"Shards:          {len(files)}")
    print(f"Total rows:      {total_rows:,}")
    print(f"Total sentences: {total_sents:,}")
    print(f"Words (EN):      {total_words_en:,}")
    print(f"Words (KK):      {total_words_kk:,}")
    print(f"Chars (EN):      {total_chars_en:,}")
    print(f"Chars (KK):      {total_chars_kk:,}")
    print(f"\nPer-shard breakdown:")
    for f, n, s, w in shard_stats:
        print(f"  {f}: {n:>10,} rows, {s:>12,} sents, {w:>12,} words_en")
    print(f"\nSample texts (first 5):")
    for j, (en, kk) in enumerate(zip(samples_en, samples_kk)):
        print(f"\n--- Sample {j+1} ---")
        print(f"EN: {en}")
        print(f"KK: {kk}")

if __name__ == "__main__":
    main()
