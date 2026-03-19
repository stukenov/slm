#!/usr/bin/env bash
# Run an experiment by config name.
# Usage: ./scripts/run_experiment.sh exp001_dapt_pythia14m [--max_steps 10]
set -euo pipefail

CONFIG_NAME="${1:?Usage: $0 <config_name> [extra args]}"
shift

CONFIG_PATH="configs/experiments/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

echo "=== Running experiment: $CONFIG_NAME ==="
echo "Config: $CONFIG_PATH"
echo "GPUs available: $NUM_GPUS"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using torchrun with $NUM_GPUS GPUs"
    torchrun --nproc_per_node="$NUM_GPUS" -m slm.train --config "$CONFIG_PATH" "$@"
else
    python -m slm.train --config "$CONFIG_PATH" "$@"
fi
