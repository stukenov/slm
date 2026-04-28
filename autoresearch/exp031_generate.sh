#!/bin/bash
# ============================================================================
# exp031: Generate GEC v2 dataset using GPT-OSS-120B on 2x4090
#
# Steps:
# 1. Install vLLM
# 2. Launch GPT-OSS-120B AWQ on 2 GPUs (tensor parallel)
# 3. Wait for server ready
# 4. Run generation script
# 5. Upload to HF
# ============================================================================
set -e

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export TG_TOKEN="8620178354:AAGVRqNqAaGM_6JN_dQHbTWEbVsBvqJB6Xk"
export TG_CHAT_ID="47474471"

tg() {
    python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ.get('TG_CHAT_ID',''),
    'text': '[exp031/gec-gen] ' + '''$1''',
}).encode()
try: urllib.request.urlopen('https://api.telegram.org/bot' + os.environ.get('TG_TOKEN','') + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    echo "[TG] $1"
}

tg "Started GEC generation on $(nvidia-smi -L | wc -l)x GPU"
T0=$(date +%s)

# Step 1: Install vLLM
echo "=== Installing vLLM ==="
pip install -q vllm 2>&1 | tail -5
tg "vLLM installed"

# Step 2: Launch vLLM server with AWQ model
MODEL="twhitworth/gpt-oss-120b-awq-w4a16"
echo "=== Launching vLLM server: $MODEL ==="

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --port 8000 \
    --trust-remote-code \
    > /root/vllm_server.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server (PID=$VLLM_PID)..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        tg "vLLM server ready with $MODEL"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server died! Check /root/vllm_server.log"
        tg "vLLM FAILED — check logs"
        tail -30 /root/vllm_server.log
        exit 1
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "  [$i/120] waiting for model load..."
    fi
    sleep 5
done

# Step 3: Generate GEC data
echo "=== Generating GEC pairs ==="
tg "Starting generation..."

python3 /root/gec_generate_v2.py \
    --vllm_url http://localhost:8000 \
    --output /root/gec_v2.jsonl \
    --batch_size 10 \
    --max_seeds 2000 \
    2>&1 | tee /root/gec_generate.log

PAIRS=$(wc -l < /root/gec_v2.jsonl)
T1=$(date +%s)
tg "Generation done: $PAIRS pairs in $(( (T1-T0)/60 ))min"

# Step 4: Upload to HF
echo "=== Uploading to HF ==="
python3 << 'UPLOADEOF'
import os, json
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN")
repo = "stukenov/sozkz-corpus-synthetic-kk-gec-v2"

create_repo(repo, token=token, exist_ok=True, repo_type="dataset")
api = HfApi()
api.upload_file(
    path_or_fileobj="/root/gec_v2.jsonl",
    path_in_repo="data/train.jsonl",
    repo_id=repo,
    repo_type="dataset",
    token=token,
)
print(f"Uploaded to {repo}")
UPLOADEOF

tg "Uploaded to HF. Total: $PAIRS pairs, $(( (T1-T0)/60 ))min"

# Kill vLLM
kill $VLLM_PID 2>/dev/null

T_END=$(date +%s)
tg "Total time: $(( (T_END-T0)/60 ))min"
