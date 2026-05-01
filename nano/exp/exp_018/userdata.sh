#!/bin/bash
set -euo pipefail
exec > /var/log/exp018.log 2>&1

echo "=== exp_018: Morpheme BPE 256K Tokenizer ==="
echo "Started: $(date)"

apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git > /dev/null

mkdir -p /root/exp018
cd /root/exp018

python3 -m venv .venv
source .venv/bin/activate

pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet datasets tokenizers transformers tqdm huggingface_hub pyarrow

mkdir -p /root/.cache/huggingface
echo "$HF_TOKEN" > /root/.cache/huggingface/token
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

echo "[1] Downloading experiment code from HF..."
pip install --quiet huggingface_hub

python3 -c "
from huggingface_hub import hf_hub_download
import os, shutil

# We'll download directly from the git repo via raw files
# Instead, let's just write the files inline
print('Files will be uploaded via SCP')
"

echo "[1] Done downloading"
echo "=== Setup complete, waiting for SCP upload ==="
