#!/bin/bash
# Submission B (aggressive): SLOT-24 + Pre-quant TTT
# Usage: run_aggressive.sh [seed] (default: 1337)
# Run on 8xH100 inside /workspace/parameter-golf/
set -e
SEED="${1:-1337}"
echo "=== Submission B (aggressive): seed=$SEED ==="
SEED=$SEED TTT_ENABLED=1 TTT_EPOCHS=6 TTT_FREEZE_BLOCKS=2 \
SLOT_ENABLED=1 SLOT_STEPS=24 SLOT_LR=0.024 SLOT_LR_MIN=0.001 SLOT_STRIDE=96 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "/workspace/aggressive_seed${SEED}.log"
