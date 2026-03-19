#!/bin/bash
# ============================================================================
# exp024: SFT Pipeline — 600M Llama instruct on vast.ai
# 1x RTX 4090, AmanMussa/kazakh-instruction-v2 (52K), 3 epochs
# ============================================================================

export TG_TOKEN="5159241157:AAGksR3Dm_5DwxHZStjC2mNq7Z3iNZOxO68"
export TG_CHAT_ID="47474471"
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key 2>/dev/null)
export VAST_INSTANCE_ID="33096468"

EXPERIMENT="exp024_sft_600m"
HF_REPO="stukenov/sozkz-core-llama-600m-kk-instruct-v1"
WORKDIR="/root/slm"

tg() {
    local msg="$1"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
        TG_MSG="[$EXPERIMENT] $msg" python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ['TG_CHAT_ID'],
    'text': os.environ['TG_MSG'],
}).encode()
try:
    urllib.request.urlopen('https://api.telegram.org/bot' + os.environ['TG_TOKEN'] + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    fi
    echo "[TG] $msg"
}

# --- Step 0: Environment ---
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
tg "Started on $GPU_NAME"

# --- Step 1: Install deps ---
tg "Installing dependencies..."
cd $WORKDIR
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv > /dev/null 2>&1; then
    python3 -c "
import urllib.request, subprocess
script = urllib.request.urlopen('https://astral.sh/uv/install.sh').read()
open('/tmp/uv_install.sh', 'wb').write(script)
subprocess.run(['bash', '/tmp/uv_install.sh'], check=True)
"
fi
uv sync 2>&1 | tail -3
tg "Dependencies installed"

# --- Step 2: Train SFT ---
tg "Starting SFT: 600M, 52K examples, 3 epochs"
T0=$(date +%s)

PYTHONUNBUFFERED=1 uv run python3 -m slm.train_sft \
    --config configs/experiments/exp024_sft_alpaca_600m.yaml 2>&1 | tee sft.log | while IFS= read -r line; do
        echo "$line"
        if echo "$line" | grep -qE "'loss':|eval_loss"; then
            tg "$(echo $line | sed 's/^  //' | head -c 200)"
        fi
        if echo "$line" | grep -q "SFT Training complete"; then
            tg "$(echo $line | xargs)"
        fi
    done
TRAIN_EXIT=${PIPESTATUS[0]}
T1=$(date +%s)
TRAIN_HOURS=$(python3 -c "print(f'{($T1-$T0)/3600:.1f}')")

if [ "$TRAIN_EXIT" -ne 0 ]; then
    tg "ERROR: SFT FAILED (exit $TRAIN_EXIT) after ${TRAIN_HOURS}h. Instance kept alive!"
    exit 1
fi
tg "SFT done: ${TRAIN_HOURS}h"

# --- Step 3: Upload to HuggingFace ---
tg "Uploading to HF: $HF_REPO"

FINAL_DIR=$(find outputs/exp024_sft_alpaca_600m -name "final" -type d 2>/dev/null | head -1)
if [ -z "$FINAL_DIR" ]; then
    tg "ERROR: No final dir found! Instance kept alive!"
    exit 1
fi

UPLOAD_OK=0
python3 -c "
import os
os.environ['HF_TOKEN'] = '$HF_TOKEN'
from huggingface_hub import HfApi
api = HfApi()
repo_id = '$HF_REPO'
api.create_repo(repo_id, exist_ok=True, private=False)
api.upload_folder(folder_path='$FINAL_DIR', repo_id=repo_id, commit_message='Upload SFT model exp024')
api.update_repo_settings(repo_id=repo_id, gated='manual')
print('Upload complete')
" 2>&1 | tee upload.log | tail -10

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    HTTP_CODE=$(python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('https://huggingface.co/api/models/$HF_REPO', timeout=15)
    print(r.getcode())
except Exception as e:
    print(getattr(e, 'code', 0))
")
    if [ "$HTTP_CODE" = "200" ]; then
        UPLOAD_OK=1
        tg "Model uploaded: https://huggingface.co/$HF_REPO"
    else
        tg "ERROR: Upload script OK but repo not found (HTTP $HTTP_CODE). Instance kept alive!"
    fi
else
    tg "ERROR: Upload failed. Instance kept alive!"
fi

# --- Step 4: Self-destruct ---
if [ "$UPLOAD_OK" = "1" ]; then
    T_END=$(date +%s)
    TOTAL_HOURS=$(python3 -c "print(f'{($T_END-$T0)/3600:.1f}')")
    TOTAL_COST=$(python3 -c "print(f'{($T_END-$T0)/3600*0.269:.2f}')")
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
else
    tg "Training OK but upload FAILED. Instance kept alive!"
fi
