#!/bin/bash
# Deploy all files to a RunPod pod and run setup
# Usage: bash scripts/deploy_to_pod.sh SSH_HOST SSH_PORT
#   e.g.: bash scripts/deploy_to_pod.sh 194.68.245.27 22340
set -e

SSH_HOST="${1:?Usage: deploy_to_pod.sh SSH_HOST SSH_PORT}"
SSH_PORT="${2:?Usage: deploy_to_pod.sh SSH_HOST SSH_PORT}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 root@${SSH_HOST} -p ${SSH_PORT}"
SCP="scp -o StrictHostKeyChecking=no -P ${SSH_PORT}"

DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying to root@${SSH_HOST}:${SSH_PORT} ==="

# Upload files
echo "Uploading train_gpt.py..."
$SCP "$DIR/train_gpt.py" "root@${SSH_HOST}:/workspace/train_gpt.py"

echo "Uploading scripts..."
$SCP "$DIR/scripts/setup_pod.sh" "root@${SSH_HOST}:/workspace/setup_pod.sh"
$SCP "$DIR/scripts/run_train.sh" "root@${SSH_HOST}:/workspace/run_train.sh"

echo "Uploading config..."
$SCP "$DIR/configs/v1_11L_wd04.env" "root@${SSH_HOST}:/workspace/config.env"

# Run setup (clone repo + download data)
echo ""
echo "=== Running setup on pod ==="
$SSH "mkdir -p /workspace/logs && bash /workspace/setup_pod.sh"

# Copy our train script into the repo
$SSH "cp /workspace/train_gpt.py /workspace/parameter-golf/train_gpt.py"

echo ""
echo "=== DEPLOY COMPLETE ==="
echo "SSH in and run:"
echo "  $SSH"
echo "  bash /workspace/run_train.sh quick    # 500-step test"
echo "  bash /workspace/run_train.sh single   # full 600s run"
