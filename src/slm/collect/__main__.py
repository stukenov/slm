"""CLI entrypoint: python -m slm.collect"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import collect, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Collect Kazakh text from public HuggingFace datasets",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated source names (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/collected",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HF repo ID to push merged dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to collect.yaml config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be collected without downloading",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load config
    clean_cfg: dict = {}
    if args.config:
        cfg = load_config(args.config)
        clean_cfg = cfg.get("cleaning", {})
        # Allow config to set defaults
        if not args.output_dir or args.output_dir == "data/collected":
            args.output_dir = cfg.get("output_dir", args.output_dir)
        if not args.push_to_hub:
            args.push_to_hub = cfg.get("push_to_hub", None)

    sources = args.sources.split(",") if args.sources else None

    collect(
        sources=sources,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        dry_run=args.dry_run,
        clean_cfg=clean_cfg,
    )


if __name__ == "__main__":
    main()
