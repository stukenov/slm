#!/bin/bash
# exp032: Deploy and run token counting on RunPod pod
# Usage: bash autoresearch/exp032_deploy.sh
set -euo pipefail

POD_INFO="/tmp/exp032_pod.json"
if [ ! -f "$POD_INFO" ]; then
    echo "ERROR: $POD_INFO not found. Run exp032_launch_pod.py first."
    exit 1
fi

SSH_IP=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['ssh_ip'])")
SSH_PORT=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['ssh_port'])")
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 root@${SSH_IP} -p ${SSH_PORT}"
SCP="scp -o StrictHostKeyChecking=no -P ${SSH_PORT}"

echo "=== Connecting to pod at ${SSH_IP}:${SSH_PORT} ==="
$SSH "echo 'Connected OK'" || { echo "SSH failed"; exit 1; }

echo "=== Installing dependencies ==="
$SSH "pip install -q datasets transformers huggingface-hub wandb 2>&1 | tail -5"

echo "=== Copying token counting script ==="
$SCP autoresearch/exp032_count_tokens.py root@${SSH_IP}:/root/exp032_count_tokens.py

echo "=== Launching token counting in screen ==="
$SSH "screen -dmS exp032 bash -c '
    cd /root && \
    python exp032_count_tokens.py \
        --num-proc 8 \
        --output /root/exp032_token_stats.json \
    2>&1 | tee /root/exp032.log; \
    echo DONE >> /root/exp032.log
'"

echo ""
echo "=== LAUNCHED ==="
echo "Monitor:  $SSH \"tail -f /root/exp032.log\""
echo "Results:  $SSH \"cat /root/exp032_token_stats.json\""
echo "Download: $SCP root@${SSH_IP}:/root/exp032_token_stats.json /tmp/exp032_token_stats.json"
