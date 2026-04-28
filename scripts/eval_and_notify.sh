#!/bin/bash
# Evaluate latest E2E checkpoint on GPU 1, send WER/CER to Telegram
set -euo pipefail
cd /root/slm
. .venv/bin/activate

TG_TOKEN="8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g"
TG_CHAT="47474471"
tg() { curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" -d chat_id="$TG_CHAT" -d parse_mode="HTML" -d text="$1" > /dev/null 2>&1 || true; }

CKPT="outputs/omniaudio_v2_e2e_from_ctc/checkpoint-best/model.pt"
CONFIG="omniaudio/configs/v2_scratch_sozkz_mels_e2e_from_ctc.yaml"

if [ ! -f "$CKPT" ]; then
    tg "⚠️ No E2E checkpoint-best yet"
    exit 0
fi

tg "🔍 Running WER/CER eval on kzcalm test (200 samples)..."

# Run on GPU 1 (training uses GPU 0), eval on kzcalm raw audio
RESULT=$(CUDA_VISIBLE_DEVICES=1 python -m omniaudio.evaluate_v2 \
    --config "$CONFIG" \
    --model-path "$CKPT" \
    --dataset kzcalm \
    --max-samples 50 2>&1)

WER=$(echo "$RESULT" | grep "WER:" | awk '{print $2}')
CER=$(echo "$RESULT" | grep "CER:" | awk '{print $2}')

# Get 3 random examples
EXAMPLES=$(echo "$RESULT" | grep -A1 "^REF:" | head -9)

tg "📊 <b>OmniAudio v2 E2E Eval</b>
WER: <b>${WER}</b>
CER: <b>${CER}</b>
Samples: 200 (kzcalm test)

${EXAMPLES}"
