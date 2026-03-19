#!/usr/bin/env bash
# Publish a model to HuggingFace Hub.
# Usage: ./scripts/publish.sh exp001_dapt_pythia14m sakentukenov/slm-kk-14m-dapt-v1
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment_name> <repo_name> [base_model]}"
REPO_NAME="${2:?Usage: $0 <experiment_name> <repo_name> [base_model]}"
BASE_MODEL="${3:-}"
MODEL_PATH="outputs/${EXPERIMENT}/final"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found: $MODEL_PATH"
    exit 1
fi

echo "=== Publishing: $EXPERIMENT -> $REPO_NAME ==="
ARGS="--model_path $MODEL_PATH --repo_name $REPO_NAME"
if [ -n "$BASE_MODEL" ]; then
    ARGS="$ARGS --base_model $BASE_MODEL"
fi

python -m slm.publish $ARGS
