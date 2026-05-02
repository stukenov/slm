#!/bin/bash
# Run on the RunPod pod itself. Sets up env and launches the pipeline.
set -euo pipefail

echo "=== exp027: setup ==="

# Install dependencies
pip install -q datasets tokenizers huggingface_hub fasttext-wheel numpy tqdm datasketch

# HF token
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

# Huggingface login
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

echo "=== Starting pipeline ==="
mkdir -p /workspace/exp027

# Run in background with nohup
nohup python3 /workspace/prepare_bilingual_data.py --step all \
    > /workspace/exp027.log 2>&1 &

echo "Pipeline PID: $!"
echo "Log: tail -f /workspace/exp027.log"
