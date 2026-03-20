"""Scaling curve generation for SLM paper.

Generates two plots:
1. Own-model scaling curve with power-law fit and 95% confidence band.
2. All-models scatter plot colored by family with log x-axis.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from scripts.analysis.config import (
    FAMILY_COLORS,
    MODEL_ORDER,
    PARAM_SIZES,
    get_color,
    get_family,
    setup_academic_style,
)

# Accuracy tasks (exclude BPB -- inverse scale)
_ACC_TASKS = ["mc_qa", "sentiment", "belebele", "ner", "sib200"]


def _power_law(x: float, a: float, b: float) -> float:
    """Simple power law: y = a * x^b."""
    return a * np.power(x, b)


def _avg_accuracy(model_scores: dict[str, float]) -> float | None:
    """Compute mean accuracy across accuracy tasks (exclude BPB).

    Returns None if no accuracy tasks have values.
    """
    vals = [
        model_scores[task]
        for task in _ACC_TASKS
        if task in model_scores and model_scores[task] is not None
    ]
    if not vals:
        return None
    return float(np.mean(vals))


def generate_scaling_own(
    summary: dict, output_dir: str = "paper/figures"
) -> str:
    """Generate own-model scaling curve with power-law fit and confidence band.

    Plots SozKZ models (50M-600M) with a fitted power law y = a * x^b
    and 95% confidence band from the covariance matrix.

    Args:
        summary: Parsed summary.json dict with "models" key.
        output_dir: Directory to save output files.

    Returns:
        Path to saved PDF file.
    """
    setup_academic_style()

    models_data = summary["models"]

    # Extract own models (keys starting with "sozkz")
    own_keys = [k for k in MODEL_ORDER if k.startswith("sozkz") and k in models_data]

    if len(own_keys) < 2:
        raise ValueError(
            f"Need at least 2 own models for curve fitting, found {len(own_keys)}"
        )

    params = np.array([PARAM_SIZES[k] for k in own_keys], dtype=float)
    scores = np.array(
        [_avg_accuracy(models_data[k]) * 100 for k in own_keys], dtype=float
    )

    # Fit power law
    popt, pcov = curve_fit(_power_law, params, scores, p0=[1.0, 0.3], maxfev=10000)
    perr = np.sqrt(np.diag(pcov))

    # Smooth curve for plotting
    x_smooth = np.logspace(np.log10(30), np.log10(1000), 100)
    y_fit = _power_law(x_smooth, *popt)
    y_upper = _power_law(
        x_smooth, popt[0] + 1.96 * perr[0], popt[1] + 1.96 * perr[1]
    )
    y_lower = _power_law(
        x_smooth, popt[0] - 1.96 * perr[0], popt[1] - 1.96 * perr[1]
    )

    color = FAMILY_COLORS["sozkz"]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    # Confidence band
    ax.fill_between(x_smooth, y_lower, y_upper, alpha=0.2, color=color)

    # Fitted curve
    ax.plot(x_smooth, y_fit, "--", color=color, linewidth=1.5, label="Power-law fit")

    # Scatter points
    ax.scatter(params, scores, s=80, color=color, zorder=5, label="SozKZ models")

    # Label each point
    for key, px, py in zip(own_keys, params, scores):
        label = key.replace("sozkz-", "").upper()
        ax.annotate(
            label,
            (px, py),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    # Fit equation annotation
    eq_text = f"y = {popt[0]:.2f} * x^{{{popt[1]:.3f}}}"
    ax.text(
        0.05,
        0.95,
        eq_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontstyle="italic",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("SozKZ Scaling Behavior")
    ax.legend(loc="lower right", fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "scaling_own.pdf")
    png_path = os.path.join(output_dir, "scaling_own.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path


def generate_scaling_all(
    summary: dict, output_dir: str = "paper/figures"
) -> str:
    """Generate all-models scatter plot colored by family.

    Plots all models (excluding gpt-oss-120b to avoid compressing x-axis)
    as scatter points with log x-axis. No curve fitting.

    Args:
        summary: Parsed summary.json dict with "models" key.
        output_dir: Directory to save output files.

    Returns:
        Path to saved PDF file.
    """
    setup_academic_style()

    models_data = summary["models"]

    # Exclude gpt-oss-120b (compresses x-axis too much)
    plot_keys = [
        k
        for k in MODEL_ORDER
        if k in models_data and k != "gpt-oss-120b"
    ]

    if not plot_keys:
        raise ValueError("No models found in summary data")

    # Family display names for legend
    family_display = {
        "sozkz": "SozKZ",
        "gemma": "Gemma",
        "llama": "Llama 3",
        "qwen": "Qwen 2.5",
        "mistral": "Mistral",
        "gpt-oss": "GPT-OSS",
    }

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Track families for legend (deduplicated)
    legend_handles: dict[str, plt.Line2D] = {}

    for key in plot_keys:
        avg = _avg_accuracy(models_data[key])
        if avg is None:
            continue
        px = PARAM_SIZES[key]
        py = avg * 100
        color = get_color(key)
        family = get_family(key)

        handle = ax.scatter(px, py, s=80, color=color, zorder=5)
        if family not in legend_handles:
            legend_handles[family] = handle

        # Short label
        short = key.replace("sozkz-", "").replace("llama-3-", "L").replace(
            "qwen-", "Q"
        ).replace("gemma-", "G").replace("mistral-", "M")
        ax.annotate(
            short,
            (px, py),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Performance vs Model Size")

    # Legend
    ax.legend(
        [legend_handles[f] for f in legend_handles],
        [family_display.get(f, f) for f in legend_handles],
        loc="lower right",
        framealpha=0.9,
    )

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "scaling_all.pdf")
    png_path = os.path.join(output_dir, "scaling_all.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path
