#!/bin/bash
# ============================================================================
# exp030 Wave 1: 4 parallel experiments (one per GPU)
#   GPU 0: 030a — LoRA r=16, baseline
#   GPU 1: 030b — LoRA r=64
#   GPU 2: 030c — LoRA r=16, clean=50%
#   GPU 3: 030d — Full FT, 1 epoch
# ============================================================================
set -e

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export TG_TOKEN="8620178354:AAGVRqNqAaGM_6JN_dQHbTWEbVsBvqJB6Xk"
export TG_CHAT_ID="47474471"

tg() {
    local msg="$1"
    python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ.get('TG_CHAT_ID',''),
    'text': '[exp030/wave1] ' + '''$msg''',
}).encode()
try: urllib.request.urlopen('https://api.telegram.org/bot' + os.environ.get('TG_TOKEN','') + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    echo "[TG] $msg"
}

tg "Wave 1 started on $(nvidia-smi -L | wc -l)x GPU"
T0=$(date +%s)

cd /root

# GPU 0: 030a — LoRA r=16, clean=80% (baseline)
CUDA_VISIBLE_DEVICES=0 EXP_ID=030a METHOD=lora LORA_RANK=16 LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=0.8 EPOCHS=3 \
  PYTHONUNBUFFERED=1 python3 exp030_train.py > exp030_030a.log 2>&1 &
PID_A=$!

# GPU 1: 030b — LoRA r=64
CUDA_VISIBLE_DEVICES=1 EXP_ID=030b METHOD=lora LORA_RANK=64 LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=0.8 EPOCHS=3 \
  PYTHONUNBUFFERED=1 python3 exp030_train.py > exp030_030b.log 2>&1 &
PID_B=$!

# GPU 2: 030c — LoRA r=16, clean=50% (aggressive correction)
CUDA_VISIBLE_DEVICES=2 EXP_ID=030c METHOD=lora LORA_RANK=16 LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=0.5 EPOCHS=3 \
  PYTHONUNBUFFERED=1 python3 exp030_train.py > exp030_030c.log 2>&1 &
PID_C=$!

# GPU 3: 030d — Full FT, 1 epoch (quick full FT baseline)
CUDA_VISIBLE_DEVICES=3 EXP_ID=030d METHOD=full LR=1e-5 CLEAN_RATIO=0.8 \
  EPOCHS=1 \
  PYTHONUNBUFFERED=1 python3 exp030_train.py > exp030_030d.log 2>&1 &
PID_D=$!

echo "Launched: 030a=$PID_A, 030b=$PID_B, 030c=$PID_C, 030d=$PID_D"
echo "Waiting for all to complete..."

wait $PID_A && echo "030a done" || echo "030a FAILED"
wait $PID_B && echo "030b done" || echo "030b FAILED"
wait $PID_C && echo "030c done" || echo "030c FAILED"
wait $PID_D && echo "030d done" || echo "030d FAILED"

T1=$(date +%s)
tg "Wave 1 complete in $(( (T1-T0)/60 ))min. Run exp030_eval_all.sh next."
