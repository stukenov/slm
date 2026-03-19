#!/bin/bash
# Launch TTS data preparation on vast.ai
#
# The script runs as --pre-cmd, then the "training" phase is a no-op (true).
# After completion, the instance stays alive (upload phase will fail since no model).
# Destroy manually: python -m slm.cloud destroy --instance-id <ID>
#
# Usage:
#   bash kz-calm/scripts/launch_data_prep.sh [--dry-run]

set -euo pipefail

EXTRA_ARGS=""
if [[ "${1:-}" == "--dry-run" ]]; then
    EXTRA_ARGS="--dry-run"
fi

PYTHONPATH=src python -m slm.cloud launch \
    --config kz-calm/configs/experiments/exp001_sanity.yaml \
    --hf-repo stukenov/kzcalm-tts-kk-v1 \
    --pre-cmd "pip install librosa soundfile && python kz-calm/scripts/prepare_tts_data_remote.py --hf-repo stukenov/kzcalm-tts-kk-v1" \
    --train-module slm.train \
    --max-price 0.30 \
    --disk 80 \
    --num-gpus 1 \
    $EXTRA_ARGS \
    -- --max_steps 1
