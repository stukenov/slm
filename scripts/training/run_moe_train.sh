#!/bin/bash
set -e

# Activate conda env if exists
if command -v conda &>/dev/null; then
    conda activate soz-kz-moe 2>/dev/null || true
fi

# Clean up before start
echo "Disk before:"
df -h .

# Set HF cache to local dir to avoid filling home
export HF_HOME="./hf_cache"
export TMPDIR="./tmp"
mkdir -p "$HF_HOME" "$TMPDIR"

# Run training in background with nohup
nohup python train_moe_3b.py \
    --batch-size 8 \
    --grad-accum 64 \
    > train_moe.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor: tail -f train_moe.log"
