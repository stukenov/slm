"""Utilities: config loading, seed, whitepaper formatting."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML config with inheritance support.

    If the config contains ``inherits: base``, the base config is loaded first
    from ``configs/base.yaml`` (relative to the project root) and the
    experiment config is merged on top.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    inherits = config.pop("inherits", None)
    if inherits:
        # Search for the base config: first check sibling, then walk up
        search_dir = config_path.parent
        while search_dir != search_dir.parent:
            candidate = search_dir / f"{inherits}.yaml"
            if candidate.exists():
                base_config = load_config(candidate)
                base_config.update(config)
                return base_config
            search_dir = search_dir.parent
        raise FileNotFoundError(f"Could not find base config '{inherits}.yaml'")

    return config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_whitepaper_entry(
    experiment_name: str,
    config: dict,
    metrics: dict,
    samples: list[str] | None = None,
) -> str:
    """Format an experiment result entry for WHITEPAPER.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"### {experiment_name}",
        "",
        f"**Date**: {now}",
        f"**Model**: `{config.get('model_name', 'N/A')}`",
        f"**Dataset**: `{config.get('dataset_name', 'N/A')}`",
        f"**Epochs**: {config.get('num_train_epochs', 'N/A')}",
        f"**LR**: {config.get('learning_rate', 'N/A')}",
        f"**Block size**: {config.get('block_size', 'N/A')}",
        f"**Batch size**: {config.get('per_device_train_batch_size', 'N/A')} "
        f"x {config.get('gradient_accumulation_steps', 1)} (grad accum)",
        "",
        "**Metrics**:",
        "",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")

    if samples:
        lines.extend(["", "**Generation samples**:", ""])
        for s in samples:
            lines.append(f"> {s}")
            lines.append("")

    return "\n".join(lines)
