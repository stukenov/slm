#!/bin/bash
# ============================================================================
# exp030 Wave 2: 4 more experiments based on Wave 1 winners
#   GPU 0: 030e — All linear LoRA modules
#   GPU 1: 030f — 5 epochs
#   GPU 2: 030g — 90% clean (conservative)
#   GPU 3: 030h — Full FT lr=5e-5, 3 epochs
#
# UPDATE BEST_RANK and BEST_CLEAN from Wave 1 eval before running!
# ============================================================================
set -e

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export TG_TOKEN="8620178354:AAGVRqNqAaGM_6JN_dQHbTWEbVsBvqJB6Xk"
export TG_CHAT_ID="47474471"

# ── UPDATE THESE FROM WAVE 1 RESULTS ──
BEST_RANK=${BEST_RANK:-64}
BEST_CLEAN=${BEST_CLEAN:-0.8}
# ───────────────────────────────────────

tg() {
    local msg="$1"
    python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ.get('TG_CHAT_ID',''),
    'text': '[exp030/wave2] ' + '''$msg''',
}).encode()
try: urllib.request.urlopen('https://api.telegram.org/bot' + os.environ.get('TG_TOKEN','') + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    echo "[TG] $msg"
}

tg "Wave 2 started (rank=$BEST_RANK, clean=$BEST_CLEAN)"
T0=$(date +%s)

cd /root

# GPU 0: 030e — All linear LoRA modules
CUDA_VISIBLE_DEVICES=0 EXP_ID=030e METHOD=lora LORA_RANK=$BEST_RANK LORA_MODULES=all \
  LR=2e-4 CLEAN_RATIO=$BEST_CLEAN EPOCHS=3 \
  PYTHONUNBUFFERED=1 python3 exp030_train.py > exp030_030e.log 2>&1 &
PID_E=$!

# GPU 1: 030f — 5 epochs
CUDA_VISIBLE_DEVICES=1 EXP_ID=030f METHOD=lora LORA_RANK=$BEST_RANK LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=$BEST_CLEAN EPOCHS=5 \
  PYTHONUNBUFFERED=1 python3 exp030_train.py > exp030_030f.log 2>&1 &
PID_F=$!

# GPU 2: 030g — 90% clean (conservative correction)
CUDA_VISIBLE_DEVICES=2 EXP_ID=030g METHOD=lora LORA_RANK=$BEST_RANK LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=0.9 EPOCHS=3 \
  PYTHONUNBUFFERED=1 python3 exp030_train.py > exp030_030g.log 2>&1 &
PID_G=$!

# GPU 3: 030h — Full FT lr=5e-5, 3 epochs
CUDA_VISIBLE_DEVICES=3 EXP_ID=030h METHOD=full LR=5e-5 CLEAN_RATIO=$BEST_CLEAN \
  EPOCHS=3 \
  PYTHONUNBUFFERED=1 python3 exp030_train.py > exp030_030h.log 2>&1 &
PID_H=$!

echo "Launched: 030e=$PID_E, 030f=$PID_F, 030g=$PID_G, 030h=$PID_H"
echo "Waiting for all to complete..."

wait $PID_E && echo "030e done" || echo "030e FAILED"
wait $PID_F && echo "030f done" || echo "030f FAILED"
wait $PID_G && echo "030g done" || echo "030g FAILED"
wait $PID_H && echo "030h done" || echo "030h FAILED"

T1=$(date +%s)
tg "Wave 2 complete in $(( (T1-T0)/60 ))min. Run exp030_eval_all.sh next."
