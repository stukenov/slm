#!/bin/bash
# Kill old process, set token, restart pipeline
pkill -f prepare_bilingual || true
export HF_TOKEN="REDACTED_HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
rm -f /workspace/exp027.log
nohup python3 /workspace/prepare_bilingual_data.py --step all > /workspace/exp027.log 2>&1 &
echo "Restarted with HF_TOKEN. PID: $!"
sleep 3
head -5 /workspace/exp027.log
