"""Registry of public HuggingFace datasets containing Kazakh text."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataSource:
    name: str
    dataset_id: str
    config: str | None
    text_column: str
    split: str = "train"
    streaming: bool = True
    description: str = ""
    # For datasets with huge file trees, specify data_files to avoid slow listing
    data_files: str | None = None
    # Multiple text columns to concatenate (e.g. instruction datasets)
    text_columns: list[str] | None = None


# === Wave 1: web corpora (already collected) ===
WAVE1: list[DataSource] = [
    DataSource(
        name="culturax",
        dataset_id="uonlp/CulturaX",
        config="kk",
        text_column="text",
        description="CulturaX Kazakh subset (~2.8B tokens)",
    ),
    DataSource(
        name="madlad400",
        dataset_id="allenai/MADLAD-400",
        config=None,
        text_column="text",
        data_files="data/kk/kk_clean_*.jsonl.gz",
        description="MADLAD-400 Kazakh clean subset (~1-1.6B tokens)",
    ),
    DataSource(
        name="moscar",
        dataset_id="oscar-corpus/mOSCAR",
        config="kaz_Cyrl",
        text_column="text",
        description="mOSCAR Kazakh Cyrillic (open, replaces gated OSCAR-2301)",
    ),
    DataSource(
        name="mc4",
        dataset_id="allenai/c4",
        config="kk",
        text_column="text",
        description="mC4 Kazakh via allenai/c4 (~1-2B tokens)",
    ),
    DataSource(
        name="hplt",
        dataset_id="HPLT/HPLT2.0_cleaned",
        config=None,
        text_column="text",
        data_files="kaz_Cyrl/train-*.parquet",
        description="HPLT v2.0 cleaned Kazakh Cyrillic (45 shards)",
    ),
    DataSource(
        name="wikipedia",
        dataset_id="wikimedia/wikipedia",
        config="20231101.kk",
        text_column="text",
        description="Kazakh Wikipedia (20231101 dump)",
    ),
]

# === Wave 2: new sources ===
WAVE2: list[DataSource] = [
    # --- Deprecated HF loaders (dataset scripts no longer supported) ---
    # Leipzig: deprecated + only 1M sentences max — skipped
    # CC-100 (statmt/cc100): deprecated — download manually from statmt.org
    # OpenSubtitles (Helsinki-NLP/open_subtitles): deprecated
    # Tatoeba (Helsinki-NLP/tatoeba_mt): deprecated
    # FLORES (facebook/flores): deprecated
    # XL-Sum (csebuetnlp/xlsum): deprecated
    # KazNERD (issai/kaznerd): gated
    # --- Parallel corpora (extract Kazakh side) ---
    DataSource(
        name="kazparc",
        dataset_id="issai/kazparc",
        config="kazparc",
        text_column="source_lang",
        description="KazParC parallel corpus, Kazakh side (~372K natural sentences)",
    ),
    DataSource(
        name="kazparc_sync",
        dataset_id="issai/kazparc",
        config="sync",
        text_column="source_lang",
        description="KazParC synthetic parallel, Kazakh side (~1.8M sentences)",
    ),
    # --- Reviews ---
    DataSource(
        name="kazsandra",
        dataset_id="issai/kazsandra",
        config=None,
        text_column="text",
        description="KazSAnDRA: 180K Kazakh reviews with sentiment",
    ),
    # --- Benchmarks ---
    DataSource(
        name="belebele",
        dataset_id="facebook/belebele",
        config="kaz_Cyrl",
        text_column="flores_passage",
        split="test",
        description="Belebele reading comprehension (900 questions, Kazakh)",
    ),
    DataSource(
        name="sib200",
        dataset_id="Davlan/sib200",
        config="kaz_Cyrl",
        text_column="text",
        description="SIB-200 topic classification (Kazakh)",
    ),
    DataSource(
        name="wikiann",
        dataset_id="unimelb-nlp/wikiann",
        config="kk",
        text_column="tokens",
        description="WikiANN NER for Kazakh",
    ),
]

SOURCES: dict[str, DataSource] = {s.name: s for s in WAVE1 + WAVE2}
