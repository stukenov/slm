"""
Download ALL kk-ru parallel data from every available source.

Sources:
1. KazParC (issai/kazparc) — 372K human-translated pairs [HuggingFace]
2. WMT19 crawl.kk-ru.gz — ~4.5M kk-ru pairs [statmt.org]
3. TIL corpus — kk-ru train split [Google Cloud Storage]
4. OPUS corpora — all available kk-ru sub-corpora [OPUS]
5. WMT19 kazakhtv.kk-en.tsv.gz — kk-en pairs [statmt.org]

Output: stukenov/ekitil-parallel-kkru-v2 (updated with all data)
"""

import os
import sys
import json
import gzip
import hashlib
import tempfile
import urllib.request
import zipfile
import subprocess
from pathlib import Path
from collections import Counter

DATA_DIR = "/tmp/ekitil_parallel_v2"
os.makedirs(DATA_DIR, exist_ok=True)


def download_file(url, dest, desc=""):
    """Download a file with progress."""
    if os.path.exists(dest):
        print(f"  [cached] {desc or dest}")
        return True
    print(f"  Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Done: {size_mb:.1f}MB")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ============================================================
# 1. KazParC — 372K human-translated pairs
# ============================================================
def load_kazparc():
    from datasets import load_dataset
    print("\n=== 1. KazParC (issai/kazparc) ===")
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
# 2. WMT19 crawl.kk-ru.gz — THE BIG ONE (~4.5M pairs)
# ============================================================
def load_wmt19_crawl():
    print("\n=== 2. WMT19 crawl.kk-ru (~4.5M pairs) ===")
    url = "http://data.statmt.org/wmt19/translation-task/crawl.kk-ru.gz"
    gz_path = os.path.join(DATA_DIR, "crawl.kk-ru.gz")

    if not download_file(url, gz_path, "WMT19 crawl.kk-ru.gz (507MB)"):
        return []

    pairs = []
    print("  Parsing...")
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                kk = parts[0].strip()
                ru = parts[1].strip()
                if kk and ru and len(kk) > 2 and len(ru) > 2:
                    pairs.append({"kk": kk, "ru": ru, "source": "wmt19-crawl", "domain": "web"})

    print(f"  WMT19 crawl: {len(pairs):,} kk-ru pairs")
    return pairs


# ============================================================
# 3. TIL corpus — from Google Cloud Storage
# ============================================================
def load_til():
    print("\n=== 3. TIL Corpus (gs://til-corpus) ===")
    til_dir = os.path.join(DATA_DIR, "til")
    os.makedirs(til_dir, exist_ok=True)

    # Try gsutil first
    kk_file = os.path.join(til_dir, "train.kk")
    ru_file = os.path.join(til_dir, "train.ru")

    if not os.path.exists(kk_file):
        print("  Downloading via gsutil...")
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", "gs://til-corpus/corpus/train/kk-ru/", til_dir],
                capture_output=True, text=True, timeout=300
            )
            # Find the downloaded files
            for f in Path(til_dir).rglob("*.kk"):
                kk_file = str(f)
            for f in Path(til_dir).rglob("*.ru"):
                ru_file = str(f)
        except FileNotFoundError:
            print("  gsutil not found. Trying gcloud...")
            try:
                subprocess.run(
                    ["gcloud", "storage", "cp", "-r", "gs://til-corpus/corpus/train/kk-ru/", til_dir],
                    capture_output=True, text=True, timeout=300
                )
                for f in Path(til_dir).rglob("*.kk"):
                    kk_file = str(f)
                for f in Path(til_dir).rglob("*.ru"):
                    ru_file = str(f)
            except FileNotFoundError:
                print("  Neither gsutil nor gcloud found. Trying pip install...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "gcloud", "gsutil"],
                                   capture_output=True, timeout=120)
                except Exception:
                    pass

    # Also try direct HTTP download (backup method)
    if not os.path.exists(kk_file):
        print("  Trying direct download from GitHub releases...")
        # Try alternative: download from OPUS which has TIL data
        # TIL data is often mirrored in OPUS as well
        print("  gsutil/gcloud not available. TIL will be skipped.")
        print("  (Install gsutil: pip install gsutil, then rerun)")
        return []

    pairs = []
    if os.path.exists(kk_file) and os.path.exists(ru_file):
        with open(kk_file, "r", encoding="utf-8") as fk, \
             open(ru_file, "r", encoding="utf-8") as fr:
            for kk_line, ru_line in zip(fk, fr):
                kk = kk_line.strip()
                ru = ru_line.strip()
                if kk and ru and len(kk) > 2 and len(ru) > 2:
                    pairs.append({"kk": kk, "ru": ru, "source": "til", "domain": "general"})
        print(f"  TIL: {len(pairs):,} kk-ru pairs")
    else:
        # Try zip files
        for f in Path(til_dir).rglob("*.zip"):
            with zipfile.ZipFile(str(f)) as zf:
                names = zf.namelist()
                kk_names = [n for n in names if n.endswith(".kk")]
                ru_names = [n for n in names if n.endswith(".ru")]
                if kk_names and ru_names:
                    kk_lines = zf.read(kk_names[0]).decode("utf-8").strip().split("\n")
                    ru_lines = zf.read(ru_names[0]).decode("utf-8").strip().split("\n")
                    for kk, ru in zip(kk_lines, ru_lines):
                        kk, ru = kk.strip(), ru.strip()
                        if kk and ru and len(kk) > 2 and len(ru) > 2:
                            pairs.append({"kk": kk, "ru": ru, "source": "til", "domain": "general"})
        print(f"  TIL: {len(pairs):,} kk-ru pairs")
    return pairs


# ============================================================
# 4. OPUS corpora — all kk-ru sub-corpora
# ============================================================
OPUS_CORPORA = [
    ("WikiMatrix", "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/kk-ru.txt.zip"),
    ("QED", "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/kk-ru.txt.zip"),
    ("GNOME", "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/kk-ru.txt.zip"),
    ("KDE4", "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/kk-ru.txt.zip"),
    ("Ubuntu", "https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/kk-ru.txt.zip"),
    ("TED2020", "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/kk-ru.txt.zip"),
    ("OpenSubtitles", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/kk-ru.txt.zip"),
    ("wikimedia", "https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/moses/kk-ru.txt.zip"),
    ("XLEnt", "https://object.pouta.csc.fi/OPUS-XLEnt/v1.2/moses/kk-ru.txt.zip"),
    ("NeuLab-TedTalks", "https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/moses/kk-ru.txt.zip"),
    # Additional OPUS corpora to try
    ("CCAligned", "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses/kk-ru.txt.zip"),
    ("Tatoeba", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/kk-ru.txt.zip"),
    ("GlobalVoices", "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/kk-ru.txt.zip"),
    ("MultiUN", "https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/kk-ru.txt.zip"),
    ("bible-uedin", "https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/kk-ru.txt.zip"),
    ("ELRC-3075", "https://object.pouta.csc.fi/OPUS-ELRC-3075/v1/moses/kk-ru.txt.zip"),
    ("ELRC_2922", "https://object.pouta.csc.fi/OPUS-ELRC_2922/v1/moses/kk-ru.txt.zip"),
    ("Tanzil", "https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/kk-ru.txt.zip"),
    ("JW300", "https://object.pouta.csc.fi/OPUS-JW300/v1c/moses/kk-ru.txt.zip"),
    ("ParaCrawl", "https://object.pouta.csc.fi/OPUS-ParaCrawl/v9/moses/kk-ru.txt.zip"),
    ("CCMatrix", "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/kk-ru.txt.zip"),
    ("NLLB", "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/kk-ru.txt.zip"),
]


def load_opus_corpus(name, url, tmpdir):
    zip_path = os.path.join(tmpdir, f"{name}.zip")
    if not download_file(url, zip_path, f"OPUS-{name}"):
        return []

    pairs = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            kk_file = [n for n in names if n.endswith(".kk")]
            ru_file = [n for n in names if n.endswith(".ru")]
            if not kk_file or not ru_file:
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
    print("\n=== 4. OPUS corpora ===")
    all_pairs = []
    for name, url in OPUS_CORPORA:
        pairs = load_opus_corpus(name, url, tmpdir)
        if pairs:
            print(f"  {name}: {len(pairs):,} pairs")
            all_pairs.extend(pairs)
    print(f"  OPUS total: {len(all_pairs):,} pairs")
    return all_pairs


# ============================================================
# 5. WMT19 kk-en (bonus)
# ============================================================
def load_wmt19_kken():
    from datasets import load_dataset
    print("\n=== 5. WMT19 kk-en ===")
    ds = load_dataset("wmt19", "kk-en", split="train")
    pairs = []
    for row in ds:
        t = row["translation"]
        kk = t.get("kk", "").strip()
        en = t.get("en", "").strip()
        if kk and en and len(kk) > 2 and len(en) > 2:
            pairs.append({"kk": kk, "en": en, "source": "wmt19", "domain": "news"})
    print(f"  WMT19 kk-en: {len(pairs):,} pairs")
    return pairs


# ============================================================
# Cleaning and deduplication
# ============================================================
import re

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    return text


def clean_and_dedup(pairs, pair_type="kk-ru"):
    """Clean and deduplicate pairs."""
    tgt_key = "ru" if pair_type == "kk-ru" else "en"

    # Clean
    cleaned = []
    for p in pairs:
        p["kk"] = clean_text(p["kk"])
        p[tgt_key] = clean_text(p[tgt_key])
        if len(p["kk"]) < 3 or len(p[tgt_key]) < 3:
            continue
        if len(p["kk"]) > 5000 or len(p[tgt_key]) > 5000:
            continue
        if p["kk"] == p[tgt_key]:
            continue
        cleaned.append(p)

    # Dedup
    seen = set()
    unique = []
    for p in cleaned:
        key = hashlib.md5(f"{p['kk']}|||{p[tgt_key]}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="stukenov/ekitil-parallel-kkru-v2")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--skip-wmt19-crawl", action="store_true", help="Skip 507MB WMT19 crawl download")
    args = parser.parse_args()

    all_kk_ru = []
    all_kk_en = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. KazParC
        all_kk_ru.extend(load_kazparc())

        # 2. WMT19 crawl.kk-ru (THE BIG ONE)
        if not args.skip_wmt19_crawl:
            all_kk_ru.extend(load_wmt19_crawl())

        # 3. TIL
        all_kk_ru.extend(load_til())

        # 4. OPUS
        all_kk_ru.extend(load_all_opus(tmpdir))

        # 5. WMT19 kk-en
        all_kk_en.extend(load_wmt19_kken())

    # Clean and dedup
    print(f"\n{'='*60}")
    print(f"Raw kk-ru pairs: {len(all_kk_ru):,}")
    all_kk_ru = clean_and_dedup(all_kk_ru, "kk-ru")
    print(f"After clean+dedup: {len(all_kk_ru):,}")

    # Stats
    source_counts = Counter(p["source"] for p in all_kk_ru)
    print(f"\nkk-ru by source:")
    for src, cnt in source_counts.most_common():
        print(f"  {src:25s}: {cnt:>10,}")

    print(f"\nkk-en pairs: {len(all_kk_en):,}")

    # Save locally
    kk_ru_path = os.path.join(DATA_DIR, "parallel_kkru.jsonl")
    with open(kk_ru_path, "w", encoding="utf-8") as f:
        for p in all_kk_ru:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved: {kk_ru_path}")

    # Upload to HF
    if args.upload:
        from datasets import Dataset
        print(f"\nUploading to {args.hf_repo}...")

        ds_kkru = Dataset.from_list(all_kk_ru)
        ds_kkru.push_to_hub(args.hf_repo, config_name="kk-ru", split="train")
        print(f"  Uploaded kk-ru: {len(ds_kkru):,} pairs")

        if all_kk_en:
            ds_kken = Dataset.from_list(all_kk_en)
            ds_kken.push_to_hub(args.hf_repo, config_name="kk-en", split="train")
            print(f"  Uploaded kk-en: {len(ds_kken):,} pairs")

        print(f"Done! https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
