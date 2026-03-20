"""Grouped bar chart generation for model comparison across tasks.

Generates a publication-ready grouped bar chart showing all models
across accuracy tasks with family-based coloring.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.config import (
    MODEL_ORDER,
    TASK_DISPLAY_NAMES,
    get_color,
    get_family,
    setup_academic_style,
)


def generate_comparison_bar(
    summary: dict, output_dir: str = "paper/figures"
) -> str:
    """Generate grouped bar chart comparing models across accuracy tasks.

    X-axis = accuracy tasks, bars grouped by model, colored by family.
    BPB is excluded (inverse scale).

    Args:
        summary: Parsed summary.json dict with "models" key.
        output_dir: Directory to save output files.

    Returns:
        Path to saved PDF file.
    """
    setup_academic_style()

    models_data = summary["models"]

    # Accuracy tasks only (exclude BPB -- inverse scale)
    acc_tasks = ["mc_qa", "sentiment", "belebele", "ner", "sib200"]

    # Filter to models present in summary, ordered by MODEL_ORDER
    ordered_models = [m for m in MODEL_ORDER if m in models_data]

    n_tasks = len(acc_tasks)
    n_models = len(ordered_models)

    if n_models == 0:
        raise ValueError("No models found in summary data")

    # Bar positioning
    x = np.arange(n_tasks)
    total_width = 0.8
    bar_width = total_width / n_models

    fig, ax = plt.subplots(figsize=(10.0, 5.0))

    # Track families for legend (deduplicated)
    legend_entries: dict[str, plt.Rectangle] = {}

    for i, model in enumerate(ordered_models):
        values = []
        for task in acc_tasks:
            val = models_data[model].get(task)
            # Convert to percentage (0-100)
            values.append(val * 100 if val is not None else 0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        color = get_color(model)
        bars = ax.bar(x + offset, values, bar_width, color=color, edgecolor="white",
                      linewidth=0.3, label=model)

        # Add to legend (one entry per family)
        family = get_family(model)
        if family not in legend_entries:
            legend_entries[family] = bars[0]

    # Labels and formatting
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY_NAMES.get(t, t) for t in acc_tasks])
    ax.set_ylim(0, 105)

    # Legend with family names
    family_display = {
        "sozkz": "SozKZ",
        "gemma": "Gemma",
        "llama": "Llama 3",
        "qwen": "Qwen 2.5",
        "mistral": "Mistral",
        "gpt-oss": "GPT-OSS",
    }
    ax.legend(
        [legend_entries[f] for f in legend_entries],
        [family_display.get(f, f) for f in legend_entries],
        loc="upper right",
        framealpha=0.9,
    )

    ax.set_title("Model Comparison Across Kazakh NLP Tasks")

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "comparison_bar.pdf")
    png_path = os.path.join(output_dir, "comparison_bar.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path
