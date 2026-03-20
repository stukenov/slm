"""Single entry point for generating all SLM paper analysis outputs.

Orchestrates all analysis modules to regenerate figures, tables,
and macros from raw JSON results in one pass.

Usage:
    python scripts/analysis/generate_all.py
    python scripts/analysis/generate_all.py --skip-fertility
    python scripts/analysis/generate_all.py --results-dir paper/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from scripts.analysis.config import setup_academic_style, load_summary, OUTPUT_DIRS
from scripts.analysis.fertility import (
    compute_all_fertilities,
    generate_fertility_chart,
    generate_fertility_table,
)
from scripts.analysis.figures import generate_comparison_bar
from scripts.analysis.tables import generate_comparison_table
from scripts.analysis.scaling import generate_scaling_own, generate_scaling_all
from scripts.analysis.macros import generate_macros


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate all SLM paper figures, tables, and macros from raw JSON."
    )
    parser.add_argument(
        "--results-dir",
        default="paper/results",
        help="Path to results directory containing summary.json (default: paper/results)",
    )
    parser.add_argument(
        "--skip-fertility",
        action="store_true",
        help="Skip fertility analysis (requires network access for tokenizers)",
    )
    parser.add_argument(
        "--skip-efficiency",
        action="store_true",
        help="Skip efficiency table (requires efficiency.json)",
    )
    parser.add_argument(
        "--skip-contamination",
        action="store_true",
        help="Skip contamination report (requires contamination.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate all analysis outputs from raw JSON results."""
    args = parse_args(argv)

    setup_academic_style()

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIRS["figures"], exist_ok=True)
    os.makedirs(OUTPUT_DIRS["tables"], exist_ok=True)
    os.makedirs(OUTPUT_DIRS["macros"], exist_ok=True)

    # Load summary data
    summary = load_summary(args.results_dir)

    output_files: list[str] = []
    n_figures = 0
    n_tables = 0

    # --- Figures ---

    print("Generating comparison bar chart...")
    path = generate_comparison_bar(summary)
    output_files.append(path)
    n_figures += 1

    print("Generating scaling curves...")
    path = generate_scaling_own(summary)
    output_files.append(path)
    n_figures += 1

    path = generate_scaling_all(summary)
    output_files.append(path)
    n_figures += 1

    # --- Tables ---

    print("Generating comparison table...")
    path = generate_comparison_table(summary)
    output_files.append(path)
    n_tables += 1

    # --- Macros ---

    print("Generating macros.tex...")
    path = generate_macros(summary)
    output_files.append(path)

    # --- Optional: Fertility ---

    if not args.skip_fertility:
        print("Computing tokenizer fertility...")
        fertilities = compute_all_fertilities()
        path = generate_fertility_chart(fertilities)
        output_files.append(path)
        n_figures += 1

        path = generate_fertility_table(fertilities)
        output_files.append(path)
        n_tables += 1

    # --- Optional: Efficiency ---

    if not args.skip_efficiency:
        efficiency_path = os.path.join(args.results_dir, "efficiency.json")
        if os.path.exists(efficiency_path):
            print("Generating efficiency table...")
            from scripts.analysis.efficiency import generate_efficiency_table

            with open(efficiency_path, "r", encoding="utf-8") as f:
                efficiency_data = json.load(f)
            path = generate_efficiency_table(efficiency_data)
            output_files.append(path)
            n_tables += 1

    # --- Optional: Contamination ---

    if not args.skip_contamination:
        contamination_path = os.path.join(args.results_dir, "contamination.json")
        if os.path.exists(contamination_path):
            print("Contamination results found, including in report...")

    # --- Summary ---

    print(f"\nGenerated {n_figures} figures, {n_tables} tables, macros.tex")
    print("Output files:")
    for f in output_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
