#!/bin/bash
set -e

cd /root/slm

# Clone CosyVoice
if [ ! -d "CosyVoice" ]; then
    echo "=== Cloning CosyVoice ==="
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice
    git submodule update --init --recursive
    cd ..
fi

# Install dependencies
echo "=== Installing dependencies ==="
cd CosyVoice
source /root/slm/.venv/bin/activate
pip install -r requirements.txt 2>&1 | tail -5
cd ..

# Download pretrained model
echo "=== Downloading CosyVoice2-0.5B ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='/root/slm/CosyVoice/pretrained_models/CosyVoice2-0.5B')
print('Model downloaded!')
"

echo "=== Setup complete ==="
