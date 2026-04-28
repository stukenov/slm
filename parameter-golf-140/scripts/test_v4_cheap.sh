#!/bin/bash
# Test v4 on a cheap single GPU (4090/A6000/A40)
# Run 500 steps to verify code works end-to-end:
# training -> EMA -> SWA -> quant -> eval -> sliding window -> TTT
#
# Usage on RunPod:
#   1. Launch pod: python scripts/runpod_launch.py launch
#   2. SCP this + train_gpt_v4.py to pod
#   3. SSH in, run this script in screen
#
# Expected: ~15-20 min on 1x4090, BPB ~1.50 (500 steps, not meaningful)
# What matters: code runs end-to-end without errors, artifact < 16MB

set -euo pipefail

cd /workspace/parameter-golf 2>/dev/null || {
    echo "Not on RunPod. Trying local parameter-golf dir..."
    cd "$(dirname "$0")/.." || exit 1
}

# Download data if not present
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "=== Downloading data (1 shard for smoke test) ==="
    pip install sentencepiece datasets huggingface-hub zstandard tiktoken 2>/dev/null
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

# Copy v4 script as train_gpt.py for torchrun
if [ -f "train_gpt_v4.py" ]; then
    cp train_gpt_v4.py train_gpt.py
fi

echo "=== Starting v4 test (500 steps, 1 GPU) ==="
echo "=== XSA_LAST_N=11, LeakyReLU(0.5)^2, SDPA fallback ==="

# 500 steps, no wallclock cap, enable TTT for end-to-end test
SEED=1337 \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=500 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
XSA_LAST_N=11 \
TTT_ENABLED=1 \
TTT_EPOCHS=1 \
TTT_FREEZE_BLOCKS=0 \
TTT_CHUNK_TOKENS=32768 \
RUN_ID=v4_test_$(date +%Y%m%d_%H%M) \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee /tmp/v4_test.log

echo ""
echo "=== Test complete ==="
echo "Log: /tmp/v4_test.log"
grep -E "final_int6|legal_ttt|Total submission|artifact" /tmp/v4_test.log 2>/dev/null || true
