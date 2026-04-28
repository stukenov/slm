"""
exp034: Collect and merge all kk-ru parallel datasets into a single HF dataset.

Sources (same as deepvk/kazRush):
1. KazParC (issai/kazparc) — 372K human-translated kk-ru pairs
2. OPUS corpora (opus.nlpl.eu) — kk-ru pairs from multiple sub-corpora
3. WMT19 (wmt19, kk-en) — 127K pairs (kk-en, no ru — skip or use for kk-en)
4. TIL corpus (GitHub) — Turkic Interlingua kk-ru pairs

Output: stukenov/ekitil-parallel-kkru-v2 on HuggingFace
Format: {"kk": "...", "ru": "...", "source": "...", "domain": "..."}
"""

import os
import json
import hashlib
import tempfile
import urllib.request
import zipfile
import gzip
from pathlib import Path
from collections import Counter

# ============================================================
# 1. KazParC — high-quality human translations
# ============================================================
def load_kazparc():
    """Load issai/kazparc from HuggingFace. Columns: id, kk, en, ru, tr, domain."""
    from datasets import load_dataset
    print("Loading KazParC...")
    ds = load_dataset("issai/kazparc", split="train")
    pairs = []
    for row in ds:
        kk = (row.get("kk") or "").strip()
        ru = (row.get("ru") or "").strip()
        domain = row.get("domain", "unknown")
        if kk and ru and len(kk) > 2 and len(ru) > 2:
            pairs.append({"kk": kk, "ru": ru, "source": "kazparc", "domain": domain})
    print(f"  KazParC: {len(pairs):,} kk-ru pairs")
    return pairs


# ============================================================
# 2. OPUS corpora — download kk-ru from opus.nlpl.eu
# ============================================================
OPUS_CORPORA = [
    # (corpus_name, url_template) — moses format (two .gz files: kk, ru)
    ("WikiMatrix", "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/kk-ru.txt.zip"),
    ("CCAligned", "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/kk-ru.txt.zip"),
    ("QED", "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/kk-ru.txt.zip"),
    ("GNOME", "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/kk-ru.txt.zip"),
    ("KDE4", "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/kk-ru.txt.zip"),
    ("Ubuntu", "https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/kk-ru.txt.zip"),
    ("Tatoeba", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2024-07-01/moses/kk-ru.txt.zip"),
    ("TED2020", "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/kk-ru.txt.zip"),
    ("OpenSubtitles", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/kk-ru.txt.zip"),
    ("GlobalVoices", "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/kk-ru.txt.zip"),
    ("wikimedia", "https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/moses/kk-ru.txt.zip"),
    ("MultiUN", "https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/kk-ru.txt.zip"),
    ("UN", "https://object.pouta.csc.fi/OPUS-UN/v20090831/moses/kk-ru.txt.zip"),
    ("bible-uedin", "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/kk-ru.txt.zip"),
    ("ELRC-3075", "https://object.pouta.csc.fi/OPUS-ELRC-3075/v1/moses/kk-ru.txt.zip"),
    ("XLEnt", "https://object.pouta.csc.fi/OPUS-XLEnt/v1.2/moses/kk-ru.txt.zip"),
    ("NeuLab-TedTalks", "https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/moses/kk-ru.txt.zip"),
]

def load_opus_corpus(name, url, tmpdir):
    """Download and parse a single OPUS moses-format zip."""
    zip_path = os.path.join(tmpdir, f"{name}.zip")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"  {name}: download failed ({e})")
        return []

    pairs = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            kk_file = [n for n in names if n.endswith(".kk")]
            ru_file = [n for n in names if n.endswith(".ru")]
            if not kk_file or not ru_file:
                print(f"  {name}: no .kk/.ru files in zip ({names})")
                return []
            kk_lines = zf.read(kk_file[0]).decode("utf-8").strip().split("\n")
            ru_lines = zf.read(ru_file[0]).decode("utf-8").strip().split("\n")
            for kk, ru in zip(kk_lines, ru_lines):
                kk, ru = kk.strip(), ru.strip()
                if kk and ru and len(kk) > 2 and len(ru) > 2:
                    pairs.append({"kk": kk, "ru": ru, "source": f"opus-{name.lower()}", "domain": "general"})
    except Exception as e:
        print(f"  {name}: parse error ({e})")
        return []

    os.remove(zip_path)
    return pairs


def load_all_opus(tmpdir):
    """Download all OPUS kk-ru corpora."""
    print("Loading OPUS corpora...")
    all_pairs = []
    for name, url in OPUS_CORPORA:
        pairs = load_opus_corpus(name, url, tmpdir)
        if pairs:
            print(f"  {name}: {len(pairs):,} pairs")
            all_pairs.extend(pairs)
    print(f"  OPUS total: {len(all_pairs):,} pairs")
    return all_pairs


# ============================================================
# 3. TIL corpus — Turkic Interlingua
# ============================================================
TIL_URL = "https://raw.githubusercontent.com/turkic-interlingua/til-mt/master/til_corpus/kk/kk-ru.tsv"

def load_til(tmpdir):
    """Download TIL kk-ru parallel data from GitHub."""
    print("Loading TIL corpus...")
    tsv_path = os.path.join(tmpdir, "til_kk_ru.tsv")
    pairs = []
    try:
        urllib.request.urlretrieve(TIL_URL, tsv_path)
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    kk, ru = parts[0].strip(), parts[1].strip()
                    if kk and ru and len(kk) > 2 and len(ru) > 2:
                        pairs.append({"kk": kk, "ru": ru, "source": "til", "domain": "general"})
        print(f"  TIL: {len(pairs):,} pairs")
    except Exception as e:
        print(f"  TIL: download failed ({e}), trying directory listing...")
        # Try alternative URL patterns
        for alt_url in [
            "https://raw.githubusercontent.com/turkic-interlingua/til-mt/master/til_corpus/kk/train.kk-ru.kk",
        ]:
            try:
                urllib.request.urlretrieve(alt_url, tsv_path)
                print(f"  TIL: found at {alt_url}")
                break
            except Exception:
                continue
    os.remove(tsv_path) if os.path.exists(tsv_path) else None
    return pairs


# ============================================================
# 4. WMT19 kk-en (bonus — no ru, but useful for kk-en direction)
# ============================================================
def load_wmt19():
    """Load WMT19 kk-en pairs. These are kk-en only, no Russian."""
    from datasets import load_dataset
    print("Loading WMT19 kk-en...")
    ds = load_dataset("wmt19", "kk-en", split="train")
    pairs = []
    for row in ds:
        t = row["translation"]
        kk = t.get("kk", "").strip()
        en = t.get("en", "").strip()
        if kk and en and len(kk) > 2 and len(en) > 2:
            pairs.append({"kk": kk, "en": en, "source": "wmt19", "domain": "news"})
    print(f"  WMT19: {len(pairs):,} kk-en pairs (no ru)")
    return pairs


# ============================================================
# Deduplication and cleaning
# ============================================================
def dedup_pairs(pairs):
    """Deduplicate by (kk, ru) hash."""
    seen = set()
    unique = []
    for p in pairs:
        key = hashlib.md5(f"{p['kk']}|||{p['ru']}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def clean_text(text):
    """Basic text cleaning."""
    import re
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Remove control chars
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    return text


def clean_pairs(pairs):
    """Clean all pairs."""
    cleaned = []
    for p in pairs:
        p["kk"] = clean_text(p["kk"])
        p["ru"] = clean_text(p["ru"])
        # Skip if either side is too short or too long
        if len(p["kk"]) < 3 or len(p["ru"]) < 3:
            continue
        if len(p["kk"]) > 5000 or len(p["ru"]) > 5000:
            continue
        # Skip if kk == ru (copy-paste)
        if p["kk"] == p["ru"]:
            continue
        cleaned.append(p)
    return cleaned


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/ekitil_parallel")
    parser.add_argument("--hf-repo", default="stukenov/ekitil-parallel-kkru-v2")
    parser.add_argument("--upload", action="store_true", help="Upload to HF")
    parser.add_argument("--skip-opus", action="store_true", help="Skip OPUS download")
    parser.add_argument("--skip-til", action="store_true", help="Skip TIL download")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_kk_ru = []
    all_kk_en = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. KazParC
        kazparc = load_kazparc()
        all_kk_ru.extend(kazparc)

        # 2. OPUS
        if not args.skip_opus:
            opus = load_all_opus(tmpdir)
            all_kk_ru.extend(opus)

        # 3. TIL
        if not args.skip_til:
            til = load_til(tmpdir)
            all_kk_ru.extend(til)

        # 4. WMT19 kk-en
        wmt = load_wmt19()
        all_kk_en.extend(wmt)

    # Clean and dedup kk-ru
    print(f"\nRaw kk-ru pairs: {len(all_kk_ru):,}")
    all_kk_ru = clean_pairs(all_kk_ru)
    print(f"After cleaning: {len(all_kk_ru):,}")
    all_kk_ru = dedup_pairs(all_kk_ru)
    print(f"After dedup: {len(all_kk_ru):,}")

    # Stats by source
    source_counts = Counter(p["source"] for p in all_kk_ru)
    print("\nkk-ru by source:")
    for src, cnt in source_counts.most_common():
        print(f"  {src}: {cnt:,}")

    # Clean and dedup kk-en
    print(f"\nRaw kk-en pairs: {len(all_kk_en):,}")
    all_kk_en = [p for p in all_kk_en if len(p["kk"]) >= 3 and len(p["en"]) >= 3]
    print(f"After cleaning: {len(all_kk_en):,}")

    # Save locally
    kk_ru_path = os.path.join(args.output_dir, "parallel_kkru.jsonl")
    with open(kk_ru_path, "w", encoding="utf-8") as f:
        for p in all_kk_ru:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved kk-ru: {kk_ru_path}")

    kk_en_path = os.path.join(args.output_dir, "parallel_kken.jsonl")
    with open(kk_en_path, "w", encoding="utf-8") as f:
        for p in all_kk_en:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved kk-en: {kk_en_path}")

    # Upload to HF
    if args.upload:
        from datasets import Dataset
        from huggingface_hub import HfApi

        print(f"\nUploading to {args.hf_repo}...")

        # kk-ru dataset
        ds_kkru = Dataset.from_list(all_kk_ru)
        ds_kkru.push_to_hub(args.hf_repo, config_name="kk-ru", split="train")
        print(f"  Uploaded kk-ru split: {len(ds_kkru):,} pairs")

        # kk-en dataset (separate config)
        if all_kk_en:
            ds_kken = Dataset.from_list(all_kk_en)
            ds_kken.push_to_hub(args.hf_repo, config_name="kk-en", split="train")
            print(f"  Uploaded kk-en split: {len(ds_kken):,} pairs")

        print(f"Done! https://huggingface.co/datasets/{args.hf_repo}")

    print(f"\nTotal kk-ru: {len(all_kk_ru):,}")
    print(f"Total kk-en: {len(all_kk_en):,}")


if __name__ == "__main__":
    main()
