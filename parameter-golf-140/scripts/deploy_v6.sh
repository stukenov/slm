#!/bin/bash
# Deploy v6 to RunPod pod and run smoke test
# Usage: bash deploy_v6.sh <ssh_host> <ssh_port>
# Example: bash deploy_v6.sh root@1.2.3.4 12345
set -e

SSH_HOST="${1:?Usage: deploy_v6.sh <ssh_host> <ssh_port>}"
SSH_PORT="${2:?Usage: deploy_v6.sh <ssh_host> <ssh_port>}"
SSH="ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_HOST"
SCP="scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no -P $SSH_PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/reference_prs/pr1263_train_gpt.py"

echo "=== Step 1: Setup pod ==="
$SCP "$SCRIPT_DIR/scripts/setup_pod.sh" "$SSH_HOST:/workspace/setup_pod.sh"
$SSH "bash /workspace/setup_pod.sh" 2>&1 | tail -5

echo ""
echo "=== Step 2: Upload train_gpt.py (PR #1263 base) ==="
$SCP "$TRAIN_SCRIPT" "$SSH_HOST:/workspace/parameter-golf/train_gpt.py"
$SSH "wc -l /workspace/parameter-golf/train_gpt.py"

echo ""
echo "=== Step 3: Install flash-attn ==="
$SSH "pip install flash-attn --no-build-isolation 2>&1 | tail -3" || echo "FA3 install may take a while, continuing..."

echo ""
echo "=== Step 4: Smoke test (50 steps, 1 GPU) ==="
$SSH "cd /workspace/parameter-golf && \
  ITERATIONS=50 MAX_WALLCLOCK_SECONDS=120 VAL_LOSS_EVERY=0 SLOT_ENABLED=0 SEED=42 \
  python train_gpt.py 2>&1 | tail -20"

echo ""
echo "=== Step 5: Full 1xH100 run (600s, SLOT eval) ==="
echo "To run full training + SLOT eval:"
echo "  $SSH \"cd /workspace/parameter-golf && screen -dmS train bash -c 'SEED=1337 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee /workspace/run_v6.log'\""
echo "  $SSH \"tail -f /workspace/run_v6.log\""
