#!/bin/bash
# Training script for vast.ai (2× GPU DDP)
set -euo pipefail

cd /workspace/slm
source /workspace/.venv/bin/activate

TG_TOKEN="8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g"
TG_CHAT="47474471"

tg() {
    curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        -d chat_id="$TG_CHAT" -d parse_mode="HTML" -d text="$1" > /dev/null 2>&1 || true
}

NUM_GPUS=$(nvidia-smi -L | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
tg "🚀 <b>OmniAudio v2 Training Started</b>
GPUs: ${NUM_GPUS}× ${GPU_NAME}
Stage 1: CTC pretrain (7 epochs)
Stage 2: E2E fine-tune (10 epochs)"

mkdir -p logs

# ==========================================
#  STAGE 1: CTC Pre-train Encoder
# ==========================================
tg "📡 Stage 1: CTC pretrain starting..."
STAGE1_START=$(date +%s)

if [ "$NUM_GPUS" -gt 1 ]; then
    STAGE1_CMD="torchrun --nproc_per_node=$NUM_GPUS -m omniaudio.train_v2 --config omniaudio/configs/v2_scratch_sozkz_mels_ctc_cloud.yaml"
else
    STAGE1_CMD="python -m omniaudio.train_v2 --config omniaudio/configs/v2_scratch_sozkz_mels_ctc_cloud.yaml"
fi

if $STAGE1_CMD 2>&1 | tee logs/ctc_cloud.log; then
    STAGE1_END=$(date +%s)
    STAGE1_MINS=$(( (STAGE1_END - STAGE1_START) / 60 ))
    BEST_VAL=$(grep "Val loss" logs/ctc_cloud.log | tail -1 | grep -oP 'Val loss: \K[0-9.]+')
    tg "✅ <b>Stage 1 Complete</b> (${STAGE1_MINS}min)
Best val loss: ${BEST_VAL}"
else
    LAST_LOG=$(tail -5 logs/ctc_cloud.log 2>/dev/null)
    tg "❌ <b>Stage 1 CRASHED</b>
${LAST_LOG}"
    exit 1
fi

# ==========================================
#  STAGE 2: E2E Fine-tune
# ==========================================
tg "🎯 Stage 2: E2E fine-tune starting..."
STAGE2_START=$(date +%s)

if [ "$NUM_GPUS" -gt 1 ]; then
    STAGE2_CMD="torchrun --nproc_per_node=$NUM_GPUS -m omniaudio.train_v2 --config omniaudio/configs/v2_scratch_sozkz_mels_e2e_cloud.yaml"
else
    STAGE2_CMD="python -m omniaudio.train_v2 --config omniaudio/configs/v2_scratch_sozkz_mels_e2e_cloud.yaml"
fi

if $STAGE2_CMD 2>&1 | tee logs/e2e_cloud.log; then
    STAGE2_END=$(date +%s)
    STAGE2_MINS=$(( (STAGE2_END - STAGE2_START) / 60 ))
    BEST_VAL=$(grep "Val loss" logs/e2e_cloud.log | tail -1 | grep -oP 'Val loss: \K[0-9.]+')
    tg "✅ <b>Stage 2 Complete</b> (${STAGE2_MINS}min)
Best val loss: ${BEST_VAL}"
else
    LAST_LOG=$(tail -5 logs/e2e_cloud.log 2>/dev/null)
    tg "❌ <b>Stage 2 CRASHED</b>
${LAST_LOG}"
    exit 1
fi

# ==========================================
#  UPLOAD TO HUGGINGFACE
# ==========================================
tg "📤 Uploading checkpoints to HuggingFace..."

if python scripts/cloud/upload_checkpoint.py; then
    TOTAL_END=$(date +%s)
    TOTAL_MINS=$(( (TOTAL_END - STAGE1_START) / 60 ))
    tg "🏁 <b>ALL DONE</b> (${TOTAL_MINS}min total)
Checkpoints uploaded to HF.
⚠️ Don't forget to destroy the instance!"
else
    tg "⚠️ Upload failed, but training completed. Checkpoints are on disk."
fi
