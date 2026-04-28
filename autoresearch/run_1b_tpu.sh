#!/bin/bash
# ============================================================================
# exp029: Training Pipeline — 1.08B Llama (GQA+Zloss+FSDP) on TPU
# Google TRC Grant, spot instance.
#
# FSDP: model sharded across all TPU chips.
# SPOT-SAFE: consolidated checkpoints to GCS, portable across chip counts.
# Autonomous: installs deps, downloads data, benchmarks, trains.
# Telegram notifications at every stage.
#
# Usage (run ON the TPU VM — single host or --worker=all for pods):
#   bash run_1b_tpu.sh [--resume]
#
# Required env: HF_TOKEN (for dataset download)
# ============================================================================

set -eo pipefail

export PJRT_DEVICE=TPU
export TG_TOKEN="8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g"
export TG_CHAT_ID="47474471"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"
export GCS_CHECKPOINT_BUCKET="${GCS_CHECKPOINT_BUCKET:-gs://sozkz-trc-checkpoints/exp029}"

EXPERIMENT="exp029_llama_1b_tpu"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_CKPT="/tmp/checkpoints/exp029_1b"

# --- Telegram helper ---
tg() {
    local msg="$1"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
        python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ.get('TG_CHAT_ID',''),
    'text': '[$EXPERIMENT] ' + '''$msg''',
    'parse_mode': 'HTML'
}).encode()
try:
    urllib.request.urlopen('https://api.telegram.org/bot' + os.environ.get('TG_TOKEN','') + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    fi
    echo "[TG] $msg"
}

T_START=$(date +%s)

# ============================================================
# Step 0: Detect TPU
# ============================================================
TPU_TYPE="unknown"
NUM_CHIPS=0

if python3 -c "import torch_xla; import torch_xla.runtime as xr; print(f'chips={xr.world_size()}')" 2>/dev/null; then
    NUM_CHIPS=$(python3 -c "import torch_xla.runtime as xr; print(xr.world_size())" 2>/dev/null || echo 0)
    TPU_TYPE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type" -H "Metadata-Flavor: Google" 2>/dev/null || echo "unknown")
    tg "TPU detected: $TPU_TYPE, $NUM_CHIPS chips"
else
    tg "torch_xla not available, will install..."
fi

DISK_FREE=$(df -h / | tail -1 | awk '{print $4}')
tg "Starting on TPU $TPU_TYPE ($NUM_CHIPS chips), disk: $DISK_FREE free"

# ============================================================
# Step 1: Install dependencies
# ============================================================
tg "Installing dependencies..."
cd "$WORKDIR"

# Install PyTorch/XLA (TPU-specific) — pin versions
pip install --quiet 'torch==2.9.0' 'torch_xla==2.9.0' 2>&1 | tail -5 || true
pip install --quiet 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html 2>&1 | tail -5 || true

# Install data deps
pip install --quiet numpy pyarrow requests huggingface_hub 2>&1 | tail -3 || true

# Verify torch_xla + FSDP
python3 -c "
import torch
import torch_xla
import torch_xla.runtime as xr
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
dev = torch_xla.device()
t = torch.randn(2, 2, device=dev)
print(f'torch_xla OK: device={dev}, chips={xr.world_size()}, FSDP available')
" || { tg "ERROR: torch_xla/FSDP verification FAILED"; exit 1; }

NUM_CHIPS=$(python3 -c "import torch_xla.runtime as xr; print(xr.world_size())" 2>/dev/null || echo 1)
tg "Dependencies OK, $NUM_CHIPS TPU chips, FSDP ready"

# ============================================================
# Step 2: Download data (70% KK + 30% ENKK)
# ============================================================
tg "Downloading ~16.2B tokens (KK + ENKK, 70/30 mix)..."
T_DATA=$(date +%s)
cd "$WORKDIR"
PYTHONUNBUFFERED=1 python3 prepare_1b.py 2>&1 | tail -10
T_DATA_END=$(date +%s)
DATA_MIN=$(( (T_DATA_END - T_DATA) / 60 ))

TRAIN_BIN="$HOME/.cache/autoresearch-1b/data/train.bin"
if [ ! -f "$TRAIN_BIN" ]; then
    tg "ERROR: Data download FAILED. train.bin missing."
    exit 1
fi

TRAIN_SIZE=$(du -sh "$TRAIN_BIN" 2>/dev/null | awk '{print $1}')
tg "Data ready: train=$TRAIN_SIZE in ${DATA_MIN}min"

# ============================================================
# Step 3: Find latest checkpoint (for resume)
# ============================================================
RESUME_ARG=""

# Check GCS for consolidated checkpoints first (portable across chip counts)
if command -v gsutil &>/dev/null; then
    GCS_LATEST=$(gsutil ls "$GCS_CHECKPOINT_BUCKET/step_*/meta.json" 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
    if [ -n "$GCS_LATEST" ]; then
        GCS_DIR=$(dirname "$GCS_LATEST")
        RESUME_STEP=$(echo "$GCS_DIR" | grep -oP 'step_\K[0-9]+')
        LOCAL_RESUME="$LOCAL_CKPT/step_$RESUME_STEP"
        mkdir -p "$LOCAL_RESUME"
        gsutil -m cp -r "${GCS_DIR}/*" "$LOCAL_RESUME/" 2>/dev/null || true
        if [ -f "$LOCAL_RESUME/meta.json" ]; then
            RESUME_ARG="--resume $LOCAL_RESUME"
            tg "Resuming from GCS checkpoint step $RESUME_STEP"
        fi
    fi
fi

# Check local checkpoints
if [ -z "$RESUME_ARG" ] && [ -d "$LOCAL_CKPT" ]; then
    LATEST_CKPT=$(ls -d ${LOCAL_CKPT}/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/meta.json" ]; then
        RESUME_STEP=$(basename "$LATEST_CKPT" | sed 's/step_//')
        RESUME_ARG="--resume $LATEST_CKPT"
        tg "Resuming from local checkpoint step $RESUME_STEP"
    fi
fi

# ============================================================
# Step 4: Smoke test (10 steps)
# ============================================================
if [ -z "$RESUME_ARG" ]; then
    tg "Running FSDP smoke test (10 steps, $NUM_CHIPS chips)..."
    cd "$WORKDIR"
    PYTHONUNBUFFERED=1 python3 train_1b_tpu.py --max-steps 10 2>&1 | tee /tmp/smoke.log | tail -30

    SMOKE_EXIT=$?
    if [ "$SMOKE_EXIT" -ne 0 ]; then
        tg "ERROR: Smoke test FAILED (exit $SMOKE_EXIT). Check /tmp/smoke.log"
        exit 1
    fi

    LAST_LINE=$(grep "tok/s" /tmp/smoke.log | tail -1)
    if [ -n "$LAST_LINE" ]; then
        TPS=$(echo "$LAST_LINE" | grep -oP '[0-9]+ tok/s' | grep -oP '[0-9]+' || true)
        LOSS=$(echo "$LAST_LINE" | grep -oP 'loss [0-9.]+' | grep -oP '[0-9.]+' || true)
        if [ -n "$TPS" ] && [ "$TPS" -gt 0 ]; then
            ETA_HOURS=$(python3 -c "print(f'{16200000000 / int($TPS) / 3600:.1f}')")
            tg "Smoke test OK: ${TPS} tok/s, loss=$LOSS, full ETA: ${ETA_HOURS}h (FSDP, $NUM_CHIPS chips)"
        else
            tg "Smoke test completed but could not parse throughput"
        fi
    fi

    # Clean smoke test checkpoints
    rm -rf ${LOCAL_CKPT}/step_* 2>/dev/null
    rm -rf ${LOCAL_CKPT}/final 2>/dev/null
fi

# ============================================================
# Step 5: Full training
# ============================================================
tg "Starting FULL training: 1.08B FSDP, $NUM_CHIPS TPU chips, GQA+Zloss $RESUME_ARG"
T_TRAIN=$(date +%s)

cd "$WORKDIR"
PYTHONUNBUFFERED=1 python3 train_1b_tpu.py $RESUME_ARG 2>&1 | tee train_tpu.log | while IFS= read -r line; do
    echo "$line"
    # Telegram progress every 2000 steps
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
done
TRAIN_EXIT=${PIPESTATUS[0]}
T_TRAIN_END=$(date +%s)
TRAIN_HOURS=$(python3 -c "print(f'{($T_TRAIN_END - $T_TRAIN)/3600:.1f}')")

if [ "$TRAIN_EXIT" -ne 0 ]; then
    if [ -d "$LOCAL_CKPT" ]; then
        LATEST=$(ls -d ${LOCAL_CKPT}/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$LATEST" ]; then
            tg "Training interrupted (exit $TRAIN_EXIT) after ${TRAIN_HOURS}h. Last checkpoint: $LATEST. Will auto-resume on next TPU."
        else
            tg "ERROR: Training FAILED (exit $TRAIN_EXIT) after ${TRAIN_HOURS}h. No checkpoints."
        fi
    else
        tg "ERROR: Training FAILED (exit $TRAIN_EXIT) after ${TRAIN_HOURS}h."
    fi
    exit 1
fi

# ============================================================
# Step 6: Verify results
# ============================================================
RESULTS_FILE="$LOCAL_CKPT/final/results.json"
if [ -f "$RESULTS_FILE" ]; then
    RESULTS=$(cat "$RESULTS_FILE")
    tg "TRAINING COMPLETE!
Time: ${TRAIN_HOURS}h
Results: $RESULTS

Checkpoints synced to GCS: $GCS_CHECKPOINT_BUCKET
Consolidated checkpoint ready for HF upload.
TPU VM still alive. Upload to HF manually, then delete TPU."
else
    tg "Training exited OK but results.json missing. Check $LOCAL_CKPT/"
fi

T_END=$(date +%s)
TOTAL_HOURS=$(python3 -c "print(f'{($T_END - $T_START)/3600:.1f}')")
tg "Total pipeline time: ${TOTAL_HOURS}h"

echo ""
echo "============================================"
echo "  Pipeline finished. TPU VM still alive."
echo "  Checkpoints: $LOCAL_CKPT/"
echo "  GCS: $GCS_CHECKPOINT_BUCKET/"
echo "  Next: upload to HF, then DELETE the TPU!"
echo "============================================"
