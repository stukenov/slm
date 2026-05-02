#!/bin/bash
pkill -f prepare_bilingual 2>/dev/null || true
pkill -f annotate_kk 2>/dev/null || true
pkill -f add_russian 2>/dev/null || true
mkdir -p /workspace/exp027
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
nohup bash /workspace/run_annotate.sh > /workspace/exp027.log 2>&1 &
echo "Started. PID: $!"
sleep 5
head -15 /workspace/exp027.log
