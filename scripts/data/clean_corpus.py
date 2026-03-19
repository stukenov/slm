#!/usr/bin/env python3
"""Deep cleaning pipeline for Kazakh pretrain corpus.

Stages:
  1. Load CSV domains (streaming)
  2. Normalize (NFC, control chars, whitespace, truncate)
  3. Script profile filter (cyrillic/latin/arabic ratios)
  4. Language ID via fastText lid.176.bin
  5. Technical junk removal (URLs, emails, HTML, boilerplate)
  6. Repetition / template filter (repeated n-grams, gzip ratio)
  7. Exact + near dedup (MD5 + MinHash LSH)
  8. PPL filter (optional, KenLM)
  9. Domain balancing
  10. Tokenize + pack into 1024-token blocks
  11. Publish to HF Hub

Usage:
    python scripts/clean_corpus.py --config configs/clean_corpus.yaml
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
# transformers not needed — dataset is published as clean text

logger = logging.getLogger(__name__)

# ---------- regex patterns (compiled once) ----------

RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")
RE_LONG_UNDERSCORE = None  # set from config
RE_KAZAKH_CHAR = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")


# ============================================================
# Stage helpers
# ============================================================


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# --- Stage 2: Normalization ---

def normalize_text(text: str, max_chars: int) -> str:
    """NFC normalize, strip control chars, collapse whitespace, truncate."""
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        # truncate on paragraph boundary
        cut = text.rfind("\n\n", 0, max_chars)
        if cut == -1:
            cut = text.rfind("\n", 0, max_chars)
        if cut == -1:
            cut = max_chars
        text = text[:cut].rstrip()
    return text


# --- Stage 3: Script profile ---

def compute_script_profile(text: str) -> dict[str, float]:
    """Return fraction of chars in each script category."""
    counts = Counter()
    for ch in text:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if "\u0400" <= ch <= "\u04ff" or "\u0500" <= ch <= "\u052f":
            counts["cyrillic"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["latin"] += 1
        elif "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f":
            counts["arabic"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        elif cat.startswith("P") or cat.startswith("S"):
            counts["punctuation"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values()) or 1
    return {k: counts[k] / total for k in ["cyrillic", "latin", "arabic", "digit", "punctuation", "other"]}


def passes_script_filter(profile: dict, thresholds: dict) -> bool:
    return (
        profile.get("cyrillic", 0) >= thresholds["min_cyrillic"]
        and profile.get("latin", 0) <= thresholds["max_latin"]
        and profile.get("arabic", 0) <= thresholds["max_arabic"]
        and profile.get("other", 0) <= thresholds["max_other"]
    )


# --- Stage 4: fastText LID ---

_ft_model = None


def get_ft_model(model_path: str):
    global _ft_model
    if _ft_model is None:
        import fasttext
        fasttext.FastText.eprint = lambda x: None  # suppress warnings
        _ft_model = fasttext.load_model(model_path)
    return _ft_model


def lid_check(text: str, cfg_lid: dict) -> bool:
    """Return True if text is classified as Kazakh with sufficient confidence."""
    model = get_ft_model(cfg_lid["model_path"])
    # fastText expects single line
    line = text.replace("\n", " ")[:5000]
    result = model.predict(line, k=3)
    labels = result[0]
    scores = list(result[1]) if hasattr(result[1], '__iter__') else [result[1]]
    # labels like ['__label__kk', '__label__ru', ...]
    label_score = {l.replace("__label__", ""): float(s) for l, s in zip(labels, scores)}

    kk_score = label_score.get("kk", 0.0)
    if kk_score < cfg_lid["min_confidence"]:
        return False

    # gap check vs ru and en
    for rival in ("ru", "en"):
        rival_score = label_score.get(rival, 0.0)
        if kk_score - rival_score < cfg_lid["min_gap"]:
            return False
    return True


# --- Stage 5: Technical junk ---

def junk_check(text: str, cfg_junk: dict) -> str | None:
    """Return rejection reason or None if clean."""
    text_len = len(text) or 1

    # URL density
    urls = RE_URL.findall(text)
    url_density = len(urls) / (text_len / 1000)
    if url_density > cfg_junk["max_url_density"]:
        return "url_density"

    # HTML tags
    if len(RE_HTML_TAG.findall(text)) > 5:
        return "html_tags"

    # Special char ratio
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special / text_len > cfg_junk["max_special_char_ratio"]:
        return "special_chars"

    # Long underscore runs
    min_run = cfg_junk["min_underscore_run"]
    if re.search(r"_{" + str(min_run) + r",}", text):
        return "underscore_run"

    # Boilerplate patterns
    text_lower = text.lower()
    for pat in cfg_junk["boilerplate_patterns"]:
        if pat.lower() in text_lower:
            return f"boilerplate:{pat}"

    return None


# --- Stage 6: Repetition ---

def repeated_ngram_ratio(text: str, n: int) -> float:
    """Fraction of text covered by n-grams that appear more than once."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated_positions = sum(c for c in counts.values() if c > 1)
    return repeated_positions / len(ngrams)


def compression_ratio(text: str) -> float:
    """gzip compression ratio (compressed/original). Lower = more repetitive."""
    encoded = text.encode("utf-8")
    if len(encoded) == 0:
        return 1.0
    compressed = gzip.compress(encoded, compresslevel=6)
    return len(compressed) / len(encoded)


def passes_repetition_filter(text: str, cfg_rep: dict) -> bool:
    if repeated_ngram_ratio(text, cfg_rep["ngram_size"]) >= cfg_rep["max_repeated_ngram_ratio"]:
        return False
    if compression_ratio(text) < cfg_rep["min_compression_ratio"]:
        return False
    return True


# --- Stage 7: Dedup ---

def exact_dedup(texts: list[str]) -> list[int]:
    """Return indices of unique texts (MD5-based)."""
    seen = set()
    keep = []
    for i, t in enumerate(texts):
        h = hashlib.md5(t.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def near_dedup_minhash(texts: list[str], cfg_dedup: dict) -> list[int]:
    """Return indices to keep after MinHash LSH near-dedup."""
    from datasketch import MinHash, MinHashLSH

    threshold = cfg_dedup["minhash_threshold"]
    num_perm = cfg_dedup["minhash_num_perm"]
    ngram_size = cfg_dedup["minhash_ngram_size"]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []

    for i, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        words = text.split()
        for j in range(len(words) - ngram_size + 1):
            ngram = " ".join(words[j:j + ngram_size])
            m.update(ngram.encode("utf-8"))
        minhashes.append(m)

        try:
            lsh.insert(str(i), m)
        except ValueError:
            # duplicate detected by LSH — skip
            pass

    # query each and mark duplicates
    duplicates = set()
    for i, m in enumerate(minhashes):
        if i in duplicates:
            continue
        results = lsh.query(m)
        for r in results:
            ri = int(r)
            if ri != i and ri not in duplicates:
                duplicates.add(ri)

    keep = [i for i in range(len(texts)) if i not in duplicates]
    return keep


# ============================================================
# Main pipeline
# ============================================================


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "clean_corpus.log")
    # Force unbuffered stdout for tee compatibility
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
    )
    logger.info("Logging to %s", log_path)


def run_pipeline(cfg: dict):
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)

    hf_raw_url = cfg["hf_raw_url"]
    domain_files = cfg["domain_files"]
    output_dir = cfg["output_dir"]
    num_proc = cfg.get("num_proc", 4)

    setup_logging(output_dir)
    logger.info("=== Deep cleaning pipeline started ===")

    # Stats tracking: domain -> reason -> count
    stats = defaultdict(lambda: defaultdict(int))

    all_domain_texts: dict[str, list[str]] = {}

    def filter_texts(domain: str, texts_iter, raw_count: int | None = None):
        """Apply stages 2-6 filters to an iterable of texts."""
        clean_texts = []
        count_label = f"/ {raw_count}" if raw_count else ""
        for row_idx, text in enumerate(texts_iter):
            if row_idx % 10000 == 0:
                logger.info("[%s] Processing %d %s (kept %d) ...", domain, row_idx, count_label, len(clean_texts))
                sys.stdout.flush()

            text = text or ""

            # Stage 2: Normalize
            text = normalize_text(text, cfg["max_chars"])
            if len(text) < cfg["min_text_len"]:
                stats[domain]["2_too_short"] += 1
                continue

            # Must contain Kazakh chars
            if not RE_KAZAKH_CHAR.search(text):
                stats[domain]["2_no_kaz_chars"] += 1
                continue

            # Stage 3: Script profile
            profile = compute_script_profile(text)
            if not passes_script_filter(profile, cfg["script_thresholds"]):
                stats[domain]["3_script_profile"] += 1
                continue

            # Stage 4: LID
            if cfg["lid"].get("model_path") and os.path.exists(cfg["lid"]["model_path"]):
                if not lid_check(text, cfg["lid"]):
                    stats[domain]["4_lid"] += 1
                    continue

            # Stage 5: Junk
            reason = junk_check(text, cfg["junk"])
            if reason:
                stats[domain][f"5_junk:{reason}"] += 1
                continue

            # Stage 6: Repetition
            if not passes_repetition_filter(text, cfg["repetition"]):
                stats[domain]["6_repetition"] += 1
                continue

            clean_texts.append(text)
        return clean_texts

    # --- Stage 1+2+3+4+5+6: Load and filter per domain ---
    for filename, domain in domain_files.items():
        url = f"{hf_raw_url}/{filename}"
        logger.info("[%s] Loading from %s ...", domain, url)
        ds = load_dataset("csv", data_files=url, split="train")
        raw_count = len(ds)
        stats[domain]["0_raw"] = raw_count
        logger.info("[%s] %d raw rows", domain, raw_count)

        # Check for cached result
        cache_path = os.path.join(output_dir, f"cache_{domain}.npy")
        if os.path.exists(cache_path):
            import pickle
            with open(cache_path, "rb") as cf:
                clean_texts = pickle.load(cf)
            stats[domain]["0_raw"] = raw_count
            stats[domain]["6_passed"] = len(clean_texts)
            logger.info("[%s] Loaded %d texts from cache", domain, len(clean_texts))
        else:
            texts_iter = (row.get("text", "") for row in ds)
            clean_texts = filter_texts(domain, texts_iter, raw_count)
            stats[domain]["6_passed"] = len(clean_texts)
            logger.info("[%s] %d / %d passed stages 2-6", domain, len(clean_texts), raw_count)
            # Cache result
            import pickle
            with open(cache_path, "wb") as cf:
                pickle.dump(clean_texts, cf)
            logger.info("[%s] Cached to %s", domain, cache_path)
        all_domain_texts[domain] = clean_texts

    # --- Extra HF datasets (Wikipedia, CulturaX, etc.) ---
    for extra in cfg.get("extra_datasets", []):
        domain = extra["domain"]
        text_field = extra.get("text_field", "text")
        streaming = extra.get("streaming", False)
        max_docs = extra.get("max_docs", None)
        logger.info("[%s] Loading extra dataset: %s (%s) ...", domain, extra["name"], extra.get("subset", ""))

        kwargs = {"split": "train"}
        if extra.get("subset"):
            kwargs["name"] = extra["subset"]
        if streaming:
            kwargs["streaming"] = True

        extra_ds = load_dataset(extra["name"], **kwargs)

        if streaming:
            def stream_texts(ds_iter, field, limit):
                for i, row in enumerate(ds_iter):
                    if limit and i >= limit:
                        break
                    yield row.get(field, "") or ""
            texts_iter = stream_texts(extra_ds, text_field, max_docs)
            raw_count = max_docs
        else:
            raw_count = len(extra_ds)
            texts_iter = (row.get(text_field, "") for row in extra_ds)

        stats[domain]["0_raw"] = raw_count or 0
        logger.info("[%s] %s raw docs", domain, raw_count or "streaming")

        # Check for cached result
        cache_path = os.path.join(output_dir, f"cache_{domain}.npy")
        if os.path.exists(cache_path):
            import pickle
            with open(cache_path, "rb") as cf:
                clean_texts = pickle.load(cf)
            stats[domain]["0_raw"] = raw_count or 0
            stats[domain]["6_passed"] = len(clean_texts)
            logger.info("[%s] Loaded %d texts from cache", domain, len(clean_texts))
        else:
            clean_texts = filter_texts(domain, texts_iter, raw_count)
            stats[domain]["6_passed"] = len(clean_texts)
            logger.info("[%s] %d passed stages 2-6", domain, len(clean_texts))
            import pickle
            with open(cache_path, "wb") as cf:
                pickle.dump(clean_texts, cf)
            logger.info("[%s] Cached to %s", domain, cache_path)
        all_domain_texts[domain] = clean_texts

    # --- Stage 7: Dedup (exact then near) ---
    logger.info("=== Stage 7: Dedup ===")

    # 7a: Exact dedup within each domain
    for domain, texts in all_domain_texts.items():
        before = len(texts)
        keep_idx = exact_dedup(texts)
        all_domain_texts[domain] = [texts[i] for i in keep_idx]
        removed = before - len(keep_idx)
        stats[domain]["7a_exact_dedup"] = removed
        logger.info("[%s] Exact dedup: %d -> %d (-%d)", domain, before, len(keep_idx), removed)

    # 7b: Near dedup — combine all domains, dedup, then split back
    logger.info("Near-dedup across all domains (MinHash LSH) ...")
    combined_texts = []
    combined_domains = []
    for domain, texts in all_domain_texts.items():
        combined_texts.extend(texts)
        combined_domains.extend([domain] * len(texts))

    before_total = len(combined_texts)
    keep_idx = near_dedup_minhash(combined_texts, cfg["dedup"])
    keep_set = set(keep_idx)

    # Rebuild per-domain
    near_dedup_removed = defaultdict(int)
    new_domain_texts: dict[str, list[str]] = defaultdict(list)
    for i, (text, domain) in enumerate(zip(combined_texts, combined_domains)):
        if i in keep_set:
            new_domain_texts[domain].append(text)
        else:
            near_dedup_removed[domain] += 1

    for domain in all_domain_texts:
        stats[domain]["7b_near_dedup"] = near_dedup_removed.get(domain, 0)
        all_domain_texts[domain] = new_domain_texts.get(domain, [])
        logger.info(
            "[%s] Near dedup: -%d, remaining: %d",
            domain,
            near_dedup_removed.get(domain, 0),
            len(all_domain_texts[domain]),
        )

    after_total = sum(len(t) for t in all_domain_texts.values())
    logger.info("Total after dedup: %d -> %d", before_total, after_total)

    # --- Stage 8: PPL filter (optional) ---
    if cfg.get("ppl", {}).get("enabled", False):
        logger.info("=== Stage 8: PPL filter (enabled) ===")
        # Placeholder for KenLM integration
        logger.warning("PPL filter not yet implemented, skipping.")
    else:
        logger.info("=== Stage 8: PPL filter (disabled) ===")

    # --- Stage 9: Domain balancing ---
    target_props = cfg.get("target_proportions")

    if target_props:
        logger.info("=== Stage 9: Domain balancing ===")
        # Find the constraining factor
        available = {d: len(t) for d, t in all_domain_texts.items()}
        scales = []
        for domain, prop in target_props.items():
            if prop > 0 and domain in available and available[domain] > 0:
                scales.append(available[domain] / prop)
        budget = min(scales)
        logger.info("Budget (constrained by smallest domain): %d docs equivalent", int(budget))

        balanced_texts: dict[str, list[str]] = {}
        for domain, texts in all_domain_texts.items():
            target_n = min(len(texts), int(budget * target_props.get(domain, 0)))
            if target_n < len(texts):
                indices = rng.choice(len(texts), size=target_n, replace=False)
                indices.sort()
                balanced_texts[domain] = [texts[i] for i in indices]
            else:
                balanced_texts[domain] = texts
            stats[domain]["9_balanced"] = len(balanced_texts[domain])
            logger.info("[%s] Balanced: %d -> %d", domain, len(texts), len(balanced_texts[domain]))
    else:
        logger.info("=== Stage 9: Domain balancing SKIPPED (use all data) ===")
        balanced_texts = dict(all_domain_texts)
        for domain, texts in balanced_texts.items():
            stats[domain]["9_balanced"] = len(texts)
            logger.info("[%s] Keeping all: %d", domain, len(texts))

    # --- Stage 10: Build clean text dataset ---
    logger.info("=== Stage 10: Build clean text dataset ===")

    all_texts = []
    all_domains = []
    for domain, texts in balanced_texts.items():
        all_texts.extend(texts)
        all_domains.extend([domain] * len(texts))
        logger.info("[%s] %d texts", domain, len(texts))

    # Shuffle
    perm = rng.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in perm]
    all_domains = [all_domains[i] for i in perm]

    logger.info("Total clean texts: %d", len(all_texts))

    # --- Stage 11: Train/val split ---
    logger.info("=== Stage 11: Train/val split ===")

    n_val = max(1, int(len(all_texts) * 0.01))
    n_train = len(all_texts) - n_val

    train_ds = Dataset.from_dict({
        "text": all_texts[:n_train],
        "domain": all_domains[:n_train],
    })
    val_ds = Dataset.from_dict({
        "text": all_texts[n_train:],
        "domain": all_domains[n_train:],
    })

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})
    logger.info("Train: %d texts, Validation: %d texts", len(train_ds), len(val_ds))

    # Save locally
    save_path = os.path.join(output_dir, "dataset")
    ds_dict.save_to_disk(save_path)
    logger.info("Saved to %s", save_path)

    # --- Stage 12: Push to Hub ---
    if cfg.get("push_to_hub", False):
        hub_repo = cfg["hub_repo"]
        logger.info("Pushing to %s ...", hub_repo)
        ds_dict.push_to_hub(hub_repo, private=False)
        logger.info("Published: https://huggingface.co/datasets/%s", hub_repo)

    # --- Report ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("CLEANING REPORT")
    logger.info("=" * 70)

    for domain in sorted(stats.keys()):
        logger.info("")
        logger.info("--- %s ---", domain)
        for reason in sorted(stats[domain].keys()):
            logger.info("  %-30s: %8d", reason, stats[domain][reason])

    # Summary
    total_raw = sum(stats[d]["0_raw"] for d in stats)
    total_final = sum(stats[d].get("9_balanced", 0) for d in stats)
    logger.info("")
    logger.info("TOTAL: %d raw -> %d clean texts", total_raw, total_final)
    logger.info("=" * 70)
    logger.info("=== Pipeline finished ===")


def main():
    parser = argparse.ArgumentParser(description="Deep cleaning pipeline for Kazakh corpus")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
