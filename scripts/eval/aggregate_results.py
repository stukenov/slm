"""Aggregate per-model per-task JSON results into a summary matrix.

Scans paper/results/ subdirectories for individual benchmark JSONs
and produces paper/results/summary.json with all models x tasks.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import glob


TASKS = ["bpb", "mc_qa", "sentiment", "belebele", "ner", "sib200"]

# Primary metric for each task
PRIMARY_METRIC = {
    "bpb": "bpb",
    "mc_qa": "accuracy",
    "sentiment": "accuracy",
    "belebele": "accuracy",
    "ner": "accuracy",
    "sib200": "accuracy",
}


def collect_results(results_dir: str) -> dict[str, dict[str, float]]:
    """Scan results directory and build model -> task -> metric mapping.

    Args:
        results_dir: Root results directory (e.g. paper/results/).

    Returns:
        Dict of {model_short: {task: primary_metric_value}}.
    """
    models: dict[str, dict[str, float]] = {}

    # Scan each task subdirectory
    for task in TASKS:
        task_dir = os.path.join(results_dir, task)
        if not os.path.isdir(task_dir):
            continue

        for json_path in glob.glob(os.path.join(task_dir, "*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Warning: skipping {json_path}: {e}")
                continue

            model_short = data.get("model_short", "unknown")
            file_task = data.get("task", task)
            metrics = data.get("metrics", {})

            metric_key = PRIMARY_METRIC.get(file_task, "accuracy")
            value = metrics.get(metric_key)

            if value is not None:
                if model_short not in models:
                    models[model_short] = {}
                models[model_short][file_task] = value

    return models


def print_table(models: dict[str, dict[str, float]]) -> None:
    """Print a formatted results table to stdout."""
    if not models:
        print("No results found.")
        return

    # Column widths
    model_width = max(len(m) for m in models) + 2
    col_width = 10

    # Header
    header = f"{'Model':<{model_width}}"
    for task in TASKS:
        header += f"{task:>{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows sorted by model name
    for model_short in sorted(models):
        row = f"{model_short:<{model_width}}"
        for task in TASKS:
            val = models[model_short].get(task)
            if val is None:
                row += f"{'--':>{col_width}}"
            elif task == "bpb":
                row += f"{val:>{col_width}.3f}"
            else:
                row += f"{val * 100:>{col_width}.1f}%"
        print(row)

    print(f"\n{len(models)} models, {len(TASKS)} tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results into summary matrix"
    )
    parser.add_argument(
        "--results-dir",
        default="paper/results",
        help="Root results directory (default: paper/results)",
    )
    parser.add_argument(
        "--output",
        default="paper/results/summary.json",
        help="Output summary JSON path (default: paper/results/summary.json)",
    )
    args = parser.parse_args()

    print(f"Scanning {args.results_dir}/ ...")
    models = collect_results(args.results_dir)

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tasks": TASKS,
        "models": models,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {args.output}")

    print()
    print_table(models)


if __name__ == "__main__":
    main()
