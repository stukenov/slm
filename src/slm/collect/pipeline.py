"""Download, normalize, deduplicate and export Kazakh text from HF datasets."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path

import yaml
from datasets import load_dataset

from .sources import SOURCES, DataSource

logger = logging.getLogger(__name__)

# --- Cleaning regexes (from scripts/clean_corpus.py) ---
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
RE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")


def normalize_text(text: str, max_chars: int = 500_000) -> str:
    """NFC normalize, strip control chars, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub("", text)
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()
    if max_chars and len(text) > max_chars:
        cut = text.rfind("\n\n", 0, max_chars)
        if cut == -1:
            cut = text.rfind("\n", 0, max_chars)
        if cut == -1:
            cut = max_chars
        text = text[:cut].rstrip()
    return text


def clean_text(text: str, cfg: dict) -> str | None:
    """Apply cleaning filters. Returns cleaned text or None if rejected."""
    text = normalize_text(text, cfg.get("max_chars", 500_000))

    min_len = cfg.get("min_text_len", 50)
    if len(text) < min_len:
        return None

    # Remove texts with too many URLs or HTML
    max_url_density = cfg.get("max_url_density", 5.0)
    text_len = len(text) or 1
    urls = RE_URL.findall(text)
    if len(urls) / (text_len / 1000) > max_url_density:
        return None

    if len(RE_HTML_TAG.findall(text)) > cfg.get("max_html_tags", 5):
        return None

    return text


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def iter_source(source: DataSource):
    """Yield (text, source_name) tuples from a DataSource."""
    logger.info("Loading %s (%s) ...", source.name, source.dataset_id)
    kwargs: dict = {"split": source.split}
    if source.data_files:
        kwargs["data_files"] = source.data_files
    elif source.config:
        kwargs["name"] = source.config
    if source.streaming:
        kwargs["streaming"] = True

    try:
        ds = load_dataset(source.dataset_id, **kwargs)
    except Exception as e:
        logger.error("Failed to load %s: %s", source.name, e)
        return

    for row in ds:
        # Multi-column concat (e.g. instruction datasets)
        if source.text_columns:
            parts = []
            for col in source.text_columns:
                val = row.get(col, "") or ""
                if val:
                    parts.append(str(val))
            text = "\n".join(parts)
        else:
            text = row.get(source.text_column, "") or ""
        # Handle token lists (NER datasets) and other list fields
        if isinstance(text, list):
            text = " ".join(str(t) for t in text if t)
        # Handle nested translation dicts (e.g. open_subtitles {"kk": ..., "en": ...})
        if isinstance(text, dict):
            # Try to find Kazakh text in the dict
            text = text.get("kk", "") or text.get("kaz", "") or ""
        if text:
            yield text


def collect(
    sources: list[str] | None = None,
    output_dir: str = "data/collected",
    push_to_hub: str | None = None,
    dry_run: bool = False,
    clean_cfg: dict | None = None,
):
    """Main collection pipeline.

    Args:
        sources: List of source names to collect (None = all).
        output_dir: Directory for parquet output.
        push_to_hub: HF repo ID to push merged dataset.
        dry_run: If True, just print what would be collected.
        clean_cfg: Cleaning config overrides.
    """
    clean_cfg = clean_cfg or {}

    # Resolve sources
    if sources:
        selected = []
        for name in sources:
            if name not in SOURCES:
                logger.warning("Unknown source: %s (available: %s)", name, list(SOURCES.keys()))
                continue
            selected.append(SOURCES[name])
    else:
        selected = list(SOURCES.values())

    if not selected:
        logger.error("No valid sources selected.")
        return

    if dry_run:
        logger.info("=== DRY RUN ===")
        for src in selected:
            logger.info("  %s: %s (config=%s, col=%s)", src.name, src.dataset_id, src.config, src.text_column)
            logger.info("    %s", src.description)
        logger.info("Total: %d sources", len(selected))
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect and deduplicate
    seen_hashes: set[str] = set()
    total_collected = 0
    total_duplicates = 0
    total_filtered = 0

    # Process each source and write per-source parquet
    from datasets import Dataset

    for src in selected:
        texts = []
        src_dupes = 0
        src_filtered = 0
        count = 0

        for text in iter_source(src):
            count += 1
            if count % 100_000 == 0:
                logger.info("[%s] Processed %d rows, kept %d ...", src.name, count, len(texts))

            cleaned = clean_text(text, clean_cfg)
            if cleaned is None:
                src_filtered += 1
                continue

            h = hashlib.md5(cleaned.encode()).hexdigest()
            if h in seen_hashes:
                src_dupes += 1
                continue
            seen_hashes.add(h)
            texts.append(cleaned)

        logger.info(
            "[%s] Done: %d raw -> %d kept (filtered=%d, dupes=%d)",
            src.name, count, len(texts), src_filtered, src_dupes,
        )

        total_collected += len(texts)
        total_duplicates += src_dupes
        total_filtered += src_filtered

        if texts:
            ds = Dataset.from_dict({
                "text": texts,
                "source": [src.name] * len(texts),
            })
            out_path = os.path.join(output_dir, f"{src.name}.parquet")
            ds.to_parquet(out_path)
            logger.info("[%s] Saved %d texts to %s", src.name, len(texts), out_path)

    logger.info("=== COLLECTION SUMMARY ===")
    logger.info("Total collected: %d", total_collected)
    logger.info("Total filtered:  %d", total_filtered)
    logger.info("Total duplicates: %d", total_duplicates)
    logger.info("Unique hashes:   %d", len(seen_hashes))

    # Merge all parquets and optionally push to hub
    if push_to_hub:
        logger.info("Merging and pushing to %s ...", push_to_hub)
        from datasets import concatenate_datasets

        parts = []
        for src in selected:
            path = os.path.join(output_dir, f"{src.name}.parquet")
            if os.path.exists(path):
                parts.append(Dataset.from_parquet(path))

        if parts:
            merged = concatenate_datasets(parts)
            merged.push_to_hub(push_to_hub, private=False)
            logger.info("Pushed %d texts to %s", len(merged), push_to_hub)
        else:
            logger.warning("No data to push.")
