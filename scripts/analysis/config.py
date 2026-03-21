"""Shared analysis configuration for SLM paper figures and tables.

Defines model families, colors, ordering, matplotlib academic style,
and shared constants used by all analysis modules.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Model family colors
# ---------------------------------------------------------------------------

FAMILY_COLORS: dict[str, str] = {
    "sozkz": "#1f77b4",   # blue
    "gemma": "#2ca02c",   # green
    "llama": "#ff7f0e",   # orange
    "qwen": "#d62728",    # red
    "mistral": "#9467bd", # purple
    "gpt-oss": "#7f7f7f", # grey
}


def get_family(model_key: str) -> str:
    """Return the family prefix for a model key."""
    for prefix in FAMILY_COLORS:
        if model_key.startswith(prefix):
            return prefix
    return "other"


def get_color(model_key: str) -> str:
    """Return the color for a model key based on its family."""
    return FAMILY_COLORS.get(get_family(model_key), "#333333")


# ---------------------------------------------------------------------------
# Model ordering (own models first by size, then competitors by size)
# ---------------------------------------------------------------------------

MODEL_ORDER: list[str] = [
    "sozkz-50m",
    "sozkz-150m",
    "sozkz-300m",
    "sozkz-600m",
    "qwen-0.5b",
    "llama-3-1b",
    "qwen-1.5b",
    "gemma-2b",
    "llama-3-3b",
]

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS: list[str] = ["mc_qa", "belebele", "sib200"]

TASK_DISPLAY_NAMES: dict[str, str] = {
    "mc_qa": "MC QA",
    "belebele": "Belebele",
    "sib200": "SIB-200",
}

# ---------------------------------------------------------------------------
# Parameter sizes (millions)
# ---------------------------------------------------------------------------

PARAM_SIZES: dict[str, int] = {
    "sozkz-50m": 50,
    "sozkz-150m": 150,
    "sozkz-300m": 300,
    "sozkz-600m": 600,
    "qwen-0.5b": 500,
    "llama-3-1b": 1000,
    "qwen-1.5b": 1500,
    "gemma-2b": 2000,
    "llama-3-3b": 3000,
}

# ---------------------------------------------------------------------------
# Tokenizer map (one per family, for fertility analysis)
# ---------------------------------------------------------------------------

TOKENIZER_MAP: dict[str, str] = {
    "SozKZ (50K)": "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
    "Gemma": "google/gemma-2-2b",
    "Llama 3": "meta-llama/Llama-3.2-1B",
    "Qwen 2.5": "Qwen/Qwen2.5-0.5B",
    "Mistral": "mistralai/Mistral-7B-v0.3",
}

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

OUTPUT_DIRS: dict[str, str] = {
    "figures": "paper/figures",
    "tables": "paper/tables",
    "macros": "paper",
}

# ---------------------------------------------------------------------------
# Academic matplotlib style
# ---------------------------------------------------------------------------


def setup_academic_style() -> None:
    """Configure matplotlib for publication-ready figures with Type 1 fonts."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "mathtext.fontset": "cm",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Summary loader
# ---------------------------------------------------------------------------


def load_summary(results_dir: str = "paper/results") -> dict:
    """Load the aggregated summary.json from the results directory.

    Args:
        results_dir: Path to results directory containing summary.json.

    Returns:
        Parsed summary dict with keys: timestamp, tasks, models.

    Raises:
        FileNotFoundError: If summary.json does not exist.
    """
    path = os.path.join(results_dir, "summary.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Summary file not found at {path}. "
            f"Run 'python scripts/eval/aggregate_results.py' first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
