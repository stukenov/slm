#!/bin/bash
# One-shot pod setup: install deps + download data
# Run once after SSH-ing into the pod
set -ex

cd /workspace

# Clone repo if needed
if [ ! -d parameter-golf ]; then
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf

# Install dependencies
pip install -q sentencepiece datasets huggingface-hub zstandard tiktoken numpy tqdm

# Download competition data (~2min)
python data/cached_challenge_fineweb.py

# Verify data
ls -lh data/datasets/fineweb10B_sp1024/
echo "=== Data shards ==="
ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l
ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin | wc -l

echo ""
echo "=== SETUP COMPLETE ==="
echo "Now scp your train_gpt.py and run:"
echo "  python train_gpt.py   # single GPU"
echo "  torchrun --nproc_per_node=N train_gpt.py   # multi-GPU"
