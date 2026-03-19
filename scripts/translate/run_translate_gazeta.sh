#!/usr/bin/env bash
# Run full gazeta translation on 2 GPUs, merge, and upload to HuggingFace.
set -euo pipefail

cd /root/slm
export PYTHONUNBUFFERED=1
VENV=".venv/bin/python"
SCRIPT="scripts/translate_gazeta.py"
OUTPUT="./gazeta_kk"
COMMON_ARGS="--bf16 --num-beams 1 --batch-size 16 --chunk-batch-size 64 --no-compile --output-dir $OUTPUT"

echo "=== Starting full gazeta translation ==="
echo "$(date): Launching 2 GPU workers..."

# Clean previous test runs
rm -rf ./gazeta_kk_test

# Launch GPU 0 (first half)
$VENV $SCRIPT --gpu-id 0 --split-id 0 --num-splits 2 $COMMON_ARGS > logs/translate_gpu0.log 2>&1 &
PID0=$!
echo "GPU 0: PID=$PID0"

# Launch GPU 1 (second half)
$VENV $SCRIPT --gpu-id 1 --split-id 1 --num-splits 2 $COMMON_ARGS > logs/translate_gpu1.log 2>&1 &
PID1=$!
echo "GPU 1: PID=$PID1"

# Wait for both
echo "$(date): Waiting for both workers..."
wait $PID0
EXIT0=$?
echo "$(date): GPU 0 finished (exit=$EXIT0)"

wait $PID1
EXIT1=$?
echo "$(date): GPU 1 finished (exit=$EXIT1)"

if [ $EXIT0 -ne 0 ] || [ $EXIT1 -ne 0 ]; then
    echo "ERROR: One or both workers failed! Check logs/translate_gpu*.log"
    exit 1
fi

# Merge
echo ""
echo "=== Merging results ==="
$VENV $SCRIPT --merge --output-dir $OUTPUT

# Preview
echo ""
echo "=== Translation examples ==="
$VENV $SCRIPT --preview --output-dir $OUTPUT

# Upload to HuggingFace
echo ""
echo "=== Uploading to HuggingFace ==="
$VENV $SCRIPT --upload --output-dir $OUTPUT --hf-repo saken-tukenov/sozkz-corpus-raw-kk-gazeta-v1

echo ""
echo "=== ALL DONE ==="
echo "$(date): Dataset published at https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-raw-kk-gazeta-v1"
