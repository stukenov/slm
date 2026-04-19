#!/usr/bin/env python3
"""Fine-tune NLLB-200-distilled-600M for Kazakh GEC."""
from __future__ import annotations

import argparse
import logging

import yaml

from gecpaper.models.nllb_gec import train_nllb_gec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/round1_nllb_baseline.yaml")
    parser.add_argument("--data_path", default=None, help="Override data_path in config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.data_path:
        config["data_path"] = args.data_path

    train_nllb_gec(config)


if __name__ == "__main__":
    main()
