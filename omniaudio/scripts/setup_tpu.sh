#!/bin/bash
# Setup script for OmniAudio CTC training on TPU VM.
# Run on TPU VM after SCP'ing train_ctc_tpu.py.
#
# Usage: bash setup_tpu.sh

set -e
echo "=== OmniAudio TPU Setup ==="

# Install deps
pip install --quiet torch torchvision
pip install --quiet 'torch_xla[tpu]'
pip install --quiet transformers datasets huggingface_hub numpy

# HF auth (token should be passed via env or pre-configured)
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "HF auth: OK"
else
    echo "WARNING: HF_TOKEN not set. Upload will fail."
fi

# Verify
python3 -c "
import torch
import torch_xla
import torch_xla.runtime as xr
print(f'torch: {torch.__version__}')
print(f'torch_xla: {torch_xla.__version__}')
print(f'TPU devices: {xr.world_size()}')
" 2>/dev/null || echo "WARNING: torch_xla import failed — check versions"

echo "=== Setup complete ==="
