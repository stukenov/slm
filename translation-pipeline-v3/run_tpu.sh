#!/bin/bash
# ============================================================================
# Translation Pipeline v3 — Run on TPU VM
#
# Usage (on TPU VM):
#   bash run_tpu.sh [--benchmark]
#   bash run_tpu.sh [--smoke-test]
#   bash run_tpu.sh --start-chunk auto
# ============================================================================

set -eo pipefail

export PJRT_DEVICE=TPU
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

echo "=== Translation Pipeline v3 (TPU) ==="
echo "Working directory: $WORKDIR"

# Step 1: Install dependencies
echo "Installing dependencies..."
pip install --quiet 'torch==2.9.0' 'torch_xla==2.9.0' 2>&1 | tail -3 || true
pip install --quiet 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html 2>&1 | tail -3 || true
pip install --quiet transformers sentencepiece datasets huggingface_hub xxhash pyarrow 2>&1 | tail -3 || true

# Step 2: Verify TPU
python3 -c "
import torch_xla.core.xla_model as xm
dev = xm.xla_device()
print(f'TPU device: {dev}')
import torch_xla.runtime as xr
print(f'TPU chips: {xr.world_size()}')
" || { echo "ERROR: TPU not available"; exit 1; }

# Step 3: Run pipeline
echo "Starting pipeline..."
PYTHONUNBUFFERED=1 python3 pipeline.py "$@"
