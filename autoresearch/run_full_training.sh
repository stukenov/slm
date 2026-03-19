#!/bin/bash
# ============================================================================
# exp020: Full Training Pipeline — 300M Llama on 9B Kazakh tokens
# Downloads data -> Trains -> Uploads to HF -> Destroys instance -> Telegram logs
#
# Usage: bash run_full_training.sh
# Required env vars: TG_TOKEN, TG_CHAT_ID, HF_TOKEN, VAST_API_KEY, VAST_INSTANCE_ID
# ============================================================================
# NO set -e — we handle errors manually to prevent premature self-destruct

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXPERIMENT="exp020_llama_300m"
HF_REPO="stukenov/sozkz-core-llama-300m-kk-base-v1"
WORKDIR="/root/autoresearch"

# --- Telegram helper (injection-safe) ---
tg() {
    local msg="$1"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
        TG_MSG="[$EXPERIMENT] $msg" python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ['TG_CHAT_ID'],
    'text': os.environ['TG_MSG'],
    'parse_mode': 'HTML'
}).encode()
try:
    urllib.request.urlopen('https://api.telegram.org/bot' + os.environ['TG_TOKEN'] + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    fi
    echo "[TG] $msg"
}

# --- Step 0: Verify environment ---
tg "Started on ${NUM_GPUS}x RTX 4090 (Croatia)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "GPUs: $NUM_GPUS"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"

# --- Step 1: Install dependencies ---
tg "Installing dependencies..."
cd $WORKDIR
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv > /dev/null 2>&1; then
    python3 -c "
import urllib.request, subprocess, os
url = 'https://astral.sh/uv/install.sh'
script = urllib.request.urlopen(url).read()
open('/tmp/uv_install.sh', 'wb').write(script)
subprocess.run(['bash', '/tmp/uv_install.sh'], check=True)
"
fi
uv add setuptools 2>/dev/null || true
uv sync 2>&1 | tail -3
tg "Dependencies installed"

# --- Step 2: Download data ---
tg "Downloading 9B tokens (44 shards)..."
T0=$(date +%s)
PYTHONUNBUFFERED=1 uv run prepare.py --num-shards 44 --num-val-shards 1 2>&1 | tail -5
T1=$(date +%s)
DATA_MIN=$(( (T1 - T0) / 60 ))
TRAIN_SIZE=$(du -sh ~/.cache/autoresearch-kazakh/data/train.bin 2>/dev/null | awk '{print $1}')
tg "Data ready: ${TRAIN_SIZE} in ${DATA_MIN}min"

# --- Step 3: Train ---
tg "Starting training: 300M, ${NUM_GPUS} GPUs"
T2=$(date +%s)
PYTHONUNBUFFERED=1 uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_300m_ddp.py 2>&1 | tee train.log | while IFS= read -r line; do
        echo "$line"
        # Send progress to Telegram every 500 steps
        if echo "$line" | grep -qE "step\s+[0-9]"; then
            STEP=$(echo "$line" | grep -oP 'step\s+\K[0-9]+' || true)
            if [ -n "$STEP" ] && [ $(( STEP % 500 )) -eq 0 ] && [ "$STEP" != "0" ]; then
                tg "$(echo $line | sed 's/^  //')"
            fi
        fi
        if echo "$line" | grep -q "val_bpb:"; then
            tg "$(echo $line | xargs)"
        fi
        if echo "$line" | grep -q "Training done:"; then
            tg "$(echo $line | xargs)"
        fi
    done
TRAIN_EXIT=${PIPESTATUS[0]}
T3=$(date +%s)
TRAIN_HOURS=$(python3 -c "print(f'{($T3-$T2)/3600:.1f}')")

if [ "$TRAIN_EXIT" -ne 0 ]; then
    tg "ERROR: Training FAILED (exit $TRAIN_EXIT) after ${TRAIN_HOURS}h. Instance kept alive!"
    # Don't proceed to upload or self-destruct
    exit 1
fi

# Double-check training output exists
if [ ! -f "/root/checkpoints/final/model.pt" ] || [ ! -f "/root/checkpoints/final/results.json" ]; then
    tg "ERROR: Training exited OK but checkpoints missing! Instance kept alive!"
    exit 1
fi

tg "Training done: ${TRAIN_HOURS}h"

# --- Step 4: Upload to HuggingFace ---
tg "Uploading to HF: ${HF_REPO}"
UPLOAD_OK=0
HF_REPO=$HF_REPO HF_TOKEN=$HF_TOKEN uv run python3 upload_to_hf.py 2>&1 | tee upload.log | tail -20
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    # Verify repo actually exists on HF
    HTTP_CODE=$(python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('https://huggingface.co/api/models/${HF_REPO}', timeout=15)
    print(r.getcode())
except Exception as e:
    print(getattr(e, 'code', 0))
")
    if [ "$HTTP_CODE" = "200" ]; then
        UPLOAD_OK=1
        tg "Model uploaded: https://huggingface.co/${HF_REPO}"
    else
        tg "ERROR: Upload script exited OK but repo not found (HTTP $HTTP_CODE). Instance kept alive!"
    fi
else
    tg "ERROR: upload_to_hf.py failed (exit ${PIPESTATUS[0]}). Instance kept alive! Check upload.log"
fi

# --- Step 5: Self-destruct (ONLY if upload confirmed) ---
if [ "$UPLOAD_OK" = "1" ]; then
    T_END=$(date +%s)
    TOTAL_HOURS=$(python3 -c "print(f'{($T_END-$T0)/3600:.1f}')")
    HOURLY_RATE=${HOURLY_RATE:-6.44}
    TOTAL_COST=$(python3 -c "print(f'{($T_END-$T0)/3600*$HOURLY_RATE:.1f}')")
    tg "Total: ${TOTAL_HOURS}h, ~USD${TOTAL_COST}"
    tg "Self-destructing instance..."
    if [ -n "$VAST_API_KEY" ] && [ -n "$VAST_INSTANCE_ID" ]; then
        python3 -c "
import urllib.request
req = urllib.request.Request(
    'https://console.vast.ai/api/v0/instances/$VAST_INSTANCE_ID/',
    method='DELETE',
    headers={'Authorization': 'Bearer $VAST_API_KEY'}
)
try:
    urllib.request.urlopen(req, timeout=30)
    print('Instance destroyed')
except Exception as e:
    print(f'Destroy failed: {e}')
"
        tg "Instance destroyed. Pipeline complete!"
    fi
elif [ -f "/root/checkpoints/final/results.json" ]; then
    tg "Training OK but upload FAILED. Instance kept alive for manual recovery!"
else
    tg "ERROR: Training NOT complete. Instance kept alive. Check logs!"
fi
