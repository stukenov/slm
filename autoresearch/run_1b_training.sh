#!/bin/bash
# ============================================================================
# exp028v2: Full Training Pipeline — 1.08B Llama (GQA+Zloss, NO QK-Norm)
# on ~16.2B cleaned Kazakh+FineWeb-Edu tokens
#
# SPOT-SAFE: frequent checkpoints (500 steps), auto-resume, SIGTERM handling.
# Autonomous: downloads data, benchmarks, trains, uploads to HF.
# Telegram notifications at every stage.
# NO self-destruct — user confirms pod destruction manually.
#
# Usage: bash run_1b_training.sh [--resume]
# Required env: HF_TOKEN (or ~/.cache/huggingface/token)
# ============================================================================

export TG_TOKEN="8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g"
export TG_CHAT_ID="47474471"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-unknown}"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
EXPERIMENT="exp028v2_llama_1b"
HF_REPO="stukenov/sozkz-core-llama-1b-kk-base-v1"
WORKDIR="/root/autoresearch"
HOURLY_RATE="${HOURLY_RATE:-14.00}"  # spot rate estimate
CKPT_BASE="/root/checkpoints/exp028_1b"

# --- Telegram helper ---
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

T_START=$(date +%s)

# ============================================================
# Step 0: Verify environment
# ============================================================
tg "Starting on ${NUM_GPUS}x GPU (pod: $RUNPOD_POD_ID)"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
DISK_FREE=$(df -h / | tail -1 | awk '{print $4}')
echo "GPU: $NUM_GPUS x $GPU_NAME ($GPU_MEM MiB)"
echo "Disk: $DISK_FREE free"

if [ "$NUM_GPUS" -lt 2 ]; then
    tg "ERROR: Need at least 2 GPUs, found $NUM_GPUS. Keeping pod alive."
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    tg "ERROR: HF_TOKEN not set. Keeping pod alive."
    exit 1
fi

tg "Env OK: ${NUM_GPUS}x $GPU_NAME, ${DISK_FREE} free"

# ============================================================
# Step 1: Install dependencies
# ============================================================
tg "Installing dependencies..."
cd $WORKDIR
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv > /dev/null 2>&1; then
    python3 -c "
import urllib.request, subprocess
url = 'https://astral.sh/uv/install.sh'
script = urllib.request.urlopen(url).read()
open('/tmp/uv_install.sh', 'wb').write(script)
subprocess.run(['bash', '/tmp/uv_install.sh'], check=True)
"
fi
uv add setuptools 2>/dev/null || true
uv sync 2>&1 | tail -3
tg "Dependencies installed"

# ============================================================
# Step 2: Download data (70% KK + 30% ENKK)
# ============================================================
tg "Downloading ~16.2B tokens (KK + ENKK, 70/30 mix)..."
T_DATA=$(date +%s)
PYTHONUNBUFFERED=1 uv run prepare_1b.py 2>&1 | tail -10
T_DATA_END=$(date +%s)
DATA_MIN=$(( (T_DATA_END - T_DATA) / 60 ))

TRAIN_SIZE=$(du -sh ~/.cache/autoresearch-1b/data/train.bin 2>/dev/null | awk '{print $1}')
VAL_SIZE=$(du -sh ~/.cache/autoresearch-1b/data/val.bin 2>/dev/null | awk '{print $1}')

if [ ! -f "$HOME/.cache/autoresearch-1b/data/train.bin" ]; then
    tg "ERROR: Data download FAILED. train.bin missing. Keeping pod alive."
    exit 1
fi

tg "Data ready: train=${TRAIN_SIZE}, val=${VAL_SIZE} in ${DATA_MIN}min"

# ============================================================
# Step 3: Find latest checkpoint (for resume)
# ============================================================
RESUME_ARG=""
if [ -d "$CKPT_BASE" ]; then
    LATEST_CKPT=$(ls -d ${CKPT_BASE}/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/model.pt" ]; then
        RESUME_STEP=$(basename "$LATEST_CKPT" | sed 's/step_//')
        RESUME_ARG="--resume $LATEST_CKPT"
        tg "Found checkpoint at step $RESUME_STEP — RESUMING"
    fi
fi

# ============================================================
# Step 4: Speed benchmark (50 steps) — only if fresh start
# ============================================================
if [ -z "$RESUME_ARG" ]; then
    tg "Running speed benchmark (50 steps)..."
    PYTHONUNBUFFERED=1 uv run torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_1b_ddp.py --max-steps 50 2>&1 | tee /tmp/bench.log &
    BENCH_PID=$!

    # Wait for benchmark
    sleep 120
    kill $BENCH_PID 2>/dev/null; wait $BENCH_PID 2>/dev/null

    BENCH_LINE=$(grep "tok/s" /tmp/bench.log | tail -1)
    if [ -n "$BENCH_LINE" ]; then
        TPS=$(echo "$BENCH_LINE" | grep -oP '[0-9]+ tok/s' | grep -oP '[0-9]+')
        if [ -n "$TPS" ] && [ "$TPS" -gt 0 ]; then
            ETA_HOURS=$(python3 -c "print(f'{16200000000 / $TPS / 3600:.1f}')")
            EST_COST=$(python3 -c "print(f'{16200000000 / $TPS / 3600 * $HOURLY_RATE:.0f}')")
            tg "Benchmark: ${TPS} tok/s, ETA: ${ETA_HOURS}h, est. cost: USD${EST_COST}"
        else
            tg "WARNING: Could not parse tok/s. Continuing anyway."
        fi
    fi

    # Smoke: loss decreasing?
    FIRST_LOSS=$(grep "loss" /tmp/bench.log | head -1 | grep -oP 'loss [0-9.]+' | grep -oP '[0-9.]+')
    LAST_LOSS=$(grep "loss" /tmp/bench.log | tail -1 | grep -oP 'loss [0-9.]+' | grep -oP '[0-9.]+')
    if [ -n "$FIRST_LOSS" ] && [ -n "$LAST_LOSS" ]; then
        LOSS_OK=$(python3 -c "print('yes' if float('$LAST_LOSS') < float('$FIRST_LOSS') * 1.1 else 'no')")
        if [ "$LOSS_OK" = "no" ]; then
            tg "WARNING: Loss not decreasing ($FIRST_LOSS -> $LAST_LOSS). Continuing cautiously."
        else
            tg "Smoke test OK: loss $FIRST_LOSS -> $LAST_LOSS"
        fi
    fi

    # Clean up benchmark processes
    sleep 5
    ps aux | grep train_1b_ddp | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
    sleep 3

    # Remove benchmark checkpoint (incomplete training)
    rm -rf ${CKPT_BASE}/step_* 2>/dev/null
    rm -rf ${CKPT_BASE}/final 2>/dev/null
fi

# ============================================================
# Step 5: Full training (with checkpoint resume support)
# ============================================================
tg "Starting FULL training: 1.08B, ${NUM_GPUS} GPUs, GQA+Zloss (NO QK-Norm) $RESUME_ARG"
T_TRAIN=$(date +%s)

PUBLISH_EVERY=${PUBLISH_EVERY:-2000}
FIRST_PUBLISH=${FIRST_PUBLISH:-50}

PYTHONUNBUFFERED=1 uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_1b_ddp.py $RESUME_ARG \
    --publish-every $PUBLISH_EVERY \
    --first-publish $FIRST_PUBLISH 2>&1 | tee train.log | while IFS= read -r line; do
        echo "$line"
        # Telegram progress every 2000 steps (more frequent for spot monitoring)
        if echo "$line" | grep -qE "step\s+[0-9]"; then
            STEP=$(echo "$line" | grep -oP 'step\s+\K[0-9]+' || true)
            if [ -n "$STEP" ] && [ $(( STEP % 2000 )) -eq 0 ] && [ "$STEP" != "0" ]; then
                tg "$(echo $line | sed 's/^  //')"
            fi
        fi
        if echo "$line" | grep -q "val_bpb:"; then
            tg "$(echo $line | xargs)"
        fi
        if echo "$line" | grep -q "Training done:"; then
            tg "$(echo $line | xargs)"
        fi
        if echo "$line" | grep -q "Saved checkpoint:"; then
            tg "Checkpoint: $(echo $line | xargs)"
        fi
        if echo "$line" | grep -q "Emergency checkpoint"; then
            tg "SPOT PREEMPTION — emergency checkpoint saved!"
        fi
        if echo "$line" | grep -q "Publishing step"; then
            tg "$(echo $line | xargs)"
        fi
    done
TRAIN_EXIT=${PIPESTATUS[0]}
T_TRAIN_END=$(date +%s)
TRAIN_HOURS=$(python3 -c "print(f'{($T_TRAIN_END - $T_TRAIN)/3600:.1f}')")

if [ "$TRAIN_EXIT" -ne 0 ]; then
    # Check if it was a SIGTERM (spot preemption)
    if [ -d "$CKPT_BASE" ]; then
        LATEST=$(ls -d ${CKPT_BASE}/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$LATEST" ]; then
            tg "Training interrupted (exit $TRAIN_EXIT) after ${TRAIN_HOURS}h. Last checkpoint: $LATEST. Resume with --resume."
        else
            tg "ERROR: Training FAILED (exit $TRAIN_EXIT) after ${TRAIN_HOURS}h. No checkpoints found. Pod kept alive!"
        fi
    else
        tg "ERROR: Training FAILED (exit $TRAIN_EXIT) after ${TRAIN_HOURS}h. Pod kept alive!"
    fi
    exit 1
fi

if [ ! -f "${CKPT_BASE}/final/model.pt" ] || [ ! -f "${CKPT_BASE}/final/results.json" ]; then
    tg "ERROR: Training exited OK but checkpoints missing! Pod kept alive!"
    exit 1
fi

tg "Training complete: ${TRAIN_HOURS}h"

# ============================================================
# Step 6: Final upload to HF main branch (with inference verification)
# ============================================================
# Intermediate checkpoints were auto-published as branches during training.
# Now publish the FINAL model to main branch with full inference verification.
tg "Publishing final model to HF main branch: ${HF_REPO} (with inference verification)"
UPLOAD_OK=0

HF_REPO=$HF_REPO HF_TOKEN=$HF_TOKEN uv run python3 upload_1b_to_hf.py 2>&1 | tee upload.log | tail -30
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    UPLOAD_OK=1
    tg "Final model uploaded + inference verified: https://huggingface.co/${HF_REPO}"
else
    tg "WARNING: Final upload failed but checkpoints are already on HF branches. Check upload.log."
fi

# ============================================================
# Step 7: Final report (NO self-destruct — user confirms manually)
# ============================================================
T_END=$(date +%s)
TOTAL_HOURS=$(python3 -c "print(f'{($T_END - $T_START)/3600:.1f}')")
TOTAL_COST=$(python3 -c "print(f'{($T_END - $T_START)/3600 * $HOURLY_RATE:.0f}')")

RESULTS=$(cat ${CKPT_BASE}/final/results.json 2>/dev/null)

if [ "$UPLOAD_OK" = "1" ]; then
    tg "PIPELINE COMPLETE!
Total: ${TOTAL_HOURS}h, ~USD${TOTAL_COST}
Model: https://huggingface.co/${HF_REPO}
Results: ${RESULTS}

Pod $RUNPOD_POD_ID is still alive. Reply 'destroy' to confirm pod destruction."
else
    tg "Training OK but upload FAILED.
Total: ${TOTAL_HOURS}h
Checkpoints at ${CKPT_BASE}/
Pod kept alive for manual recovery."
fi

# Keep pod alive — do NOT self-destruct
echo "Pipeline finished. Pod kept alive for manual verification."
echo "Checkpoints: ${CKPT_BASE}/"
echo "To destroy pod, run: runpod stop pod $RUNPOD_POD_ID"
