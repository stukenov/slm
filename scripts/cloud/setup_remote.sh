#!/bin/bash
# Setup script that runs on the vast.ai instance
set -euo pipefail

echo "=== Setting up OmniAudio v2 training ==="

# Install system deps
apt-get update -qq && apt-get install -y -qq screen rsync > /dev/null 2>&1

# Setup venv
cd /workspace
python -m venv .venv
source .venv/bin/activate

# Install PyTorch + deps
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers datasets huggingface_hub numpy torchaudio

# Install omniaudio package
cd /workspace/slm
pip install -q -e omniaudio/

# Login to HF for gated dataset
if [ -n "${HF_TOKEN:-}" ]; then
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
    echo "HF login OK"
else
    echo "WARNING: HF_TOKEN not set. Login manually or set token."
fi

echo "=== Setup complete ==="
