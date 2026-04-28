#!/bin/bash
# ============================================================================
# exp030 Final: Full fine-tune on all 4 GPUs with torchrun DDP
#
# UPDATE BEST_LR, BEST_CLEAN, BEST_EPOCHS from Wave 1+2 eval before running!
# ============================================================================
set -e

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export TG_TOKEN="8620178354:AAGVRqNqAaGM_6JN_dQHbTWEbVsBvqJB6Xk"
export TG_CHAT_ID="47474471"

# ── UPDATE THESE FROM WAVE 1+2 RESULTS ──
BEST_LR=${BEST_LR:-1e-5}
BEST_CLEAN=${BEST_CLEAN:-0.8}
BEST_EPOCHS=${BEST_EPOCHS:-3}
# ──────────────────────────────────────────

HF_REPO="stukenov/sozkz-core-llama-1b-kk-gec-v1"

tg() {
    local msg="$1"
    python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ.get('TG_CHAT_ID',''),
    'text': '[exp030/final] ' + '''$msg''',
}).encode()
try: urllib.request.urlopen('https://api.telegram.org/bot' + os.environ.get('TG_TOKEN','') + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    echo "[TG] $msg"
}

NUM_GPUS=$(nvidia-smi -L | wc -l)
tg "Final training started: ${NUM_GPUS}x GPU, LR=$BEST_LR, clean=$BEST_CLEAN, epochs=$BEST_EPOCHS"
T0=$(date +%s)

cd /root

EXP_ID=030final METHOD=full LR=$BEST_LR CLEAN_RATIO=$BEST_CLEAN EPOCHS=$BEST_EPOCHS \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=$NUM_GPUS exp030_train.py 2>&1 | tee exp030_final.log

T1=$(date +%s)
tg "Training done in $(( (T1-T0)/60 ))min"

# Upload to HuggingFace
tg "Uploading to $HF_REPO..."
python3 << 'UPLOADEOF'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN")
repo = os.environ.get("HF_REPO", "stukenov/sozkz-core-llama-1b-kk-gec-v1")
fd = "/root/exp030_030final/final"

if not os.path.exists(fd):
    print(f"ERROR: {fd} not found!")
    exit(1)

create_repo(repo, token=token, exist_ok=True, private=True)
model = AutoModelForCausalLM.from_pretrained(fd)
model.push_to_hub(repo, token=token)
tokenizer = AutoTokenizer.from_pretrained(fd)
tokenizer.push_to_hub(repo, token=token)

# Upload results
api = HfApi()
results_path = f"{fd}/results.json"
if os.path.exists(results_path):
    api.upload_file(path_or_fileobj=results_path, path_in_repo="results.json",
                    repo_id=repo, token=token)

print(f"Uploaded to {repo}")
UPLOADEOF

# Verify upload
UPLOAD_OK=0
python3 -c "
import urllib.request, sys, os
try:
    token = os.environ.get('HF_TOKEN', '')
    req = urllib.request.Request(
        'https://huggingface.co/api/models/$HF_REPO',
        headers={'Authorization': 'Bearer ' + token}
    )
    r = urllib.request.urlopen(req, timeout=30)
    if r.status == 200: print('HF OK'); sys.exit(0)
except: pass
sys.exit(1)
" && UPLOAD_OK=1

[ "$UPLOAD_OK" = "1" ] && tg "Uploaded: $HF_REPO" || tg "Upload FAILED — DO NOT DESTROY POD"

T_END=$(date +%s)
tg "Total: $(( (T_END-T0)/60 ))min. Upload: $([ $UPLOAD_OK = 1 ] && echo OK || echo FAILED)"
