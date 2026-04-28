#!/bin/bash
# Two-stage training: CTC pretrain → E2E, all 4 models on all 4 GPUs
# GPU 0 = 50M, GPU 1 = 150M, GPU 2 = 600M, GPU 3 = 1B
set -e

cd /root/slm
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
LOGS=./logs
mkdir -p $LOGS

echo "============================================"
echo "STAGE 1: CTC Pretrain (4 encoders parallel)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm50m_ctc.yaml \
    > $LOGS/omniaudio_v2_llm50m.log 2>&1 &
PID0=$!
echo "[CTC] 50M encoder on GPU 0, PID=$PID0"

CUDA_VISIBLE_DEVICES=1 python -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm150m_ctc.yaml \
    > $LOGS/omniaudio_v2_llm150m.log 2>&1 &
PID1=$!
echo "[CTC] 150M encoder on GPU 1, PID=$PID1"

CUDA_VISIBLE_DEVICES=2 python -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm600m_ctc.yaml \
    > $LOGS/omniaudio_v2_llm600m.log 2>&1 &
PID2=$!
echo "[CTC] 600M encoder on GPU 2, PID=$PID2"

CUDA_VISIBLE_DEVICES=3 python -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm1b_ctc.yaml \
    > $LOGS/omniaudio_v2_llm1b.log 2>&1 &
PID3=$!
echo "[CTC] 1B encoder on GPU 3, PID=$PID3"

wait $PID0; echo "[CTC] 50M done (exit $?)"
wait $PID1; echo "[CTC] 150M done (exit $?)"
wait $PID2; echo "[CTC] 600M done (exit $?)"
wait $PID3; echo "[CTC] 1B done (exit $?)"

echo "============================================"
echo "STAGE 2: E2E Fine-tune (encoder+LLM parallel)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm50m_h100.yaml \
    > $LOGS/omniaudio_v2_llm50m.log 2>&1 &
PID0=$!
echo "[E2E] 50M on GPU 0, PID=$PID0"

CUDA_VISIBLE_DEVICES=1 python -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm150m_h100.yaml \
    > $LOGS/omniaudio_v2_llm150m.log 2>&1 &
PID1=$!
echo "[E2E] 150M on GPU 1, PID=$PID1"

CUDA_VISIBLE_DEVICES=2 python -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm600m_h100.yaml \
    > $LOGS/omniaudio_v2_llm600m.log 2>&1 &
PID2=$!
echo "[E2E] 600M on GPU 2, PID=$PID2"

CUDA_VISIBLE_DEVICES=3 python -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm1b_h100.yaml \
    > $LOGS/omniaudio_v2_llm1b.log 2>&1 &
PID3=$!
echo "[E2E] 1B on GPU 3, PID=$PID3"

wait $PID0; echo "[E2E] 50M done (exit $?)"
wait $PID1; echo "[E2E] 150M done (exit $?)"
wait $PID2; echo "[E2E] 600M done (exit $?)"
wait $PID3; echo "[E2E] 1B done (exit $?)"

echo "============================================"
echo "ALL DONE"
echo "============================================"
