#!/bin/bash
set -e

LOG="/root/gpt_oss_setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== GPT-OSS-120B Setup started at $(date) ==="

# 1. Install build deps
echo ">>> Installing build dependencies..."
apt-get update -qq && apt-get install -y -qq cmake build-essential git curl 2>/dev/null || true

# 2. Build llama.cpp with CUDA (A10 = sm_86)
echo ">>> Building llama.cpp..."
cd /root
if [ ! -d llama.cpp ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
else
    cd llama.cpp && git pull && cd /root
fi
cd llama.cpp
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
echo ">>> llama.cpp built successfully"
ls -la build/bin/llama-server build/bin/llama-cli build/bin/llama-bench 2>/dev/null

# 3. Download GPT-OSS-120B GGUF (mxfp4 format, ~80GB)
echo ">>> Downloading GPT-OSS-120B GGUF..."
cd /root
mkdir -p models
# Use huggingface-cli if available, otherwise curl
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download ggml-org/gpt-oss-120b-GGUF \
        --include "gpt-oss-120b-mxfp4*" \
        --local-dir /root/models/gpt-oss-120b
else
    pip install -q huggingface-hub
    huggingface-cli download ggml-org/gpt-oss-120b-GGUF \
        --include "gpt-oss-120b-mxfp4*" \
        --local-dir /root/models/gpt-oss-120b
fi

echo ">>> Download complete"
ls -lah /root/models/gpt-oss-120b/

echo "=== Setup completed at $(date) ==="
echo ""
echo "To run inference (optimized for A10 24GB + 1TB RAM):"
echo ""
echo "  /root/llama.cpp/build/bin/llama-server \\"
echo "    -m /root/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf \\"
echo "    -fa 1 -ncmoe 23 -ngl 99 \\"
echo "    -ub 4096 -b 4096 \\"
echo "    -c 8192 -t 8 \\"
echo "    --jinja --host 0.0.0.0 --port 8080"
