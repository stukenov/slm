#!/usr/bin/env bash
# Run all experiments sequentially.
# Usage: ./scripts/run_all.sh [exp001_dapt_pythia14m exp002_dapt_pythia31m ...]
# If no args given, runs all experiments in order.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=(
        exp001_dapt_pythia14m
        exp002_dapt_pythia31m
        exp003_custom_tok_14m
        exp004_scratch_14m
    )
fi

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
TOTAL=${#EXPERIMENTS[@]}

echo "=== Running $TOTAL experiments sequentially ==="
echo "GPUs: $NUM_GPUS"
echo "Experiments: ${EXPERIMENTS[*]}"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NUM=$((i + 1))
    CONFIG="configs/experiments/${EXP}.yaml"

    echo "============================================"
    echo "[$NUM/$TOTAL] Starting: $EXP"
    echo "Time: $(date)"
    echo "============================================"

    if [ ! -f "$CONFIG" ]; then
        echo "SKIP: Config not found: $CONFIG"
        continue
    fi

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" -m slm.train --config "$CONFIG" 2>&1 | tee -a "logs/${EXP}.log"
    else
        python -m slm.train --config "$CONFIG" 2>&1 | tee -a "logs/${EXP}.log"
    fi

    echo ""
    echo "[$NUM/$TOTAL] Finished: $EXP at $(date)"
    echo ""
done

echo "============================================"
echo "=== ALL $TOTAL EXPERIMENTS COMPLETE ==="
echo "Time: $(date)"
echo "============================================"
