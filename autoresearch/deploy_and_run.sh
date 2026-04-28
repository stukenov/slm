#!/bin/bash
# ============================================================================
# exp028: Deploy autoresearch to RunPod pod and start training.
# Reads connection info from /tmp/exp028_pod.json (written by launch_1b_pod.py)
#
# Usage:
#   1. python3 launch_1b_pod.py   (creates pod, saves /tmp/exp028_pod.json)
#   2. bash deploy_and_run.sh     (deploys code, starts training in screen)
# ============================================================================
set -e

POD_INFO="/tmp/exp028_pod.json"
if [ ! -f "$POD_INFO" ]; then
    echo "ERROR: $POD_INFO not found. Run launch_1b_pod.py first."
    exit 1
fi

SSH_IP=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['ssh_ip'])")
SSH_PORT=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['ssh_port'])")
POD_ID=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['pod_id'])")
API_KEY=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['api_key'])")

SSH="ssh root@$SSH_IP -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30"
SCP="scp -P $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30"

echo "=== Deploying to pod $POD_ID ($SSH_IP:$SSH_PORT) ==="

# Wait for SSH
echo "Waiting for SSH..."
for i in $(seq 1 30); do
    if $SSH "echo ok" 2>/dev/null; then
        echo "SSH ready!"
        break
    fi
    echo "  [$i/30] waiting..."
    sleep 10
done

# Create workspace
$SSH "mkdir -p /root/autoresearch"

# Copy files
echo "Copying autoresearch files..."
$SCP -r /Users/sakentukenov/slm/autoresearch/* root@$SSH_IP:/root/autoresearch/

# Set HF token
$SSH "mkdir -p ~/.cache/huggingface && echo 'REDACTED_HF_TOKEN' > ~/.cache/huggingface/token"

# Start training in detached screen
echo "Starting training in screen..."
$SSH "cd /root/autoresearch && \
    export RUNPOD_POD_ID=$POD_ID && \
    export RUNPOD_API_KEY=$API_KEY && \
    export HF_TOKEN=\$(cat ~/.cache/huggingface/token) && \
    export HOURLY_RATE=26.32 && \
    screen -dmS exp028 bash run_1b_training.sh"

echo ""
echo "=== Training launched! ==="
echo "Pod: $POD_ID"
echo "SSH: $SSH"
echo "Monitor: $SSH 'tail -f /root/autoresearch/train.log'"
echo "Screen: $SSH 'screen -r exp028'"
echo ""
echo "Training will notify via Telegram and self-destruct when done."
