"""Evaluation suite for KZ-CALM TTS."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_asr_backcheck(generated_dir: str, reference_texts: list[str]) -> dict:
    """Run ASR on generated audio and compute WER/CER against reference texts.

    Requires a separate ASR model (e.g., Whisper).
    """
    # TODO: implement ASR backcheck
    logger.info("ASR backcheck not yet implemented")
    return {"wer": None, "cer": None}


def evaluate_artifacts(generated_dir: str) -> dict:
    """Detect audio artifacts (clipping, noise bursts, silence gaps)."""
    # TODO: implement artifact detection
    logger.info("Artifact detection not yet implemented")
    return {"artifact_fraction": None}


def main():
    parser = argparse.ArgumentParser(description="Evaluate KZ-CALM TTS")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_set", required=True, help="Path to test phrases file")
    parser.add_argument("--output_dir", default="eval_results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    test_set = Path(args.test_set)
    phrases = test_set.read_text().strip().splitlines()
    logger.info(f"Loaded {len(phrases)} test phrases from {test_set}")

    # TODO: generate audio for each phrase, then evaluate
    logger.info("Full evaluation pipeline not yet implemented. Placeholder.")


if __name__ == "__main__":
    main()
