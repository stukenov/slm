#!/usr/bin/env bash
# Evaluate a trained model.
# Usage: ./scripts/evaluate.sh exp001_dapt_pythia14m
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment_name>}"
MODEL_PATH="outputs/${EXPERIMENT}/final"
PROMPTS="eval/prompts_kk.txt"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found: $MODEL_PATH"
    exit 1
fi

echo "=== Evaluating: $EXPERIMENT ==="
python -m slm.evaluate \
    --model_path "$MODEL_PATH" \
    --prompts "$PROMPTS" \
    --output "outputs/${EXPERIMENT}/eval_results.json"
