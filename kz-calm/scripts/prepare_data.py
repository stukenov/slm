"""Data preparation: download, segment, QC, create manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare TTS data")
    parser.add_argument("--input_dir", required=True, help="Directory with raw audio + transcripts")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--min_snr", type=float, default=15.0)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: implement full pipeline
    # 1. Load audio files + transcripts
    # 2. Segment long files (VAD-based)
    # 3. Run QC (SNR, clipping, duration)
    # 4. Normalize text
    # 5. Write JSONL manifest

    logger.info("Data preparation pipeline placeholder. Implement per dataset format.")


if __name__ == "__main__":
    main()
