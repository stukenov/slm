#!/bin/bash
# Submission A (safe): Pre-quant TTT, NO SLOT — guaranteed legal
# Usage: run_safe.sh [seed] (default: 1337)
# Run on 8xH100 inside /workspace/parameter-golf/
set -e
SEED="${1:-1337}"
echo "=== Submission A (safe): seed=$SEED ==="
SEED=$SEED TTT_ENABLED=1 TTT_EPOCHS=6 TTT_FREEZE_BLOCKS=2 \
SLOT_ENABLED=0 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "/workspace/safe_seed${SEED}.log"
