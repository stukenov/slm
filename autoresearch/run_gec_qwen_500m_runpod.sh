#!/bin/bash
# exp039: Launch GEC Qwen 500M fine-tuning on RunPod
# Usage: bash autoresearch/run_gec_qwen_500m_runpod.sh [POD_ID]
#
# If POD_ID is provided, deploys to existing pod.
# Otherwise creates a new pod first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POD_ID="${1:-}"
GPU_TYPE="NVIDIA RTX A6000"
GPU_COUNT=1
DOCKER_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')}"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN or have ~/.cache/huggingface/token"
    exit 1
fi

# --- Create pod if needed ---
if [ -z "$POD_ID" ]; then
    echo "Creating RunPod: ${GPU_COUNT}x ${GPU_TYPE}..."
    POD_ID=$(runpodctl create pod \
        --name "exp039-gec-qwen500m" \
        --gpuType "${GPU_TYPE}" \
        --gpuCount ${GPU_COUNT} \
        --imageName "${DOCKER_IMAGE}" \
        --containerDiskSize 80 \
        --ports "22/tcp" \
        --args "" 2>&1 | sed -n 's/.*pod "\([^"]*\)".*/\1/p' || true)

    if [ -z "$POD_ID" ]; then
        echo "Failed to create pod. Trying NVIDIA L40S..."
        POD_ID=$(runpodctl create pod \
            --name "exp039-gec-qwen500m" \
            --gpuType "NVIDIA L40S" \
            --gpuCount 1 \
            --imageName "${DOCKER_IMAGE}" \
            --containerDiskSize 80 \
            --ports "22/tcp" \
            --args "" 2>&1 | sed -n 's/.*pod "\([^"]*\)".*/\1/p' || true)
    fi

    if [ -z "$POD_ID" ]; then
        echo "FAILED: No GPU available"
        exit 1
    fi
    echo "Pod created: $POD_ID"
    echo "Waiting for pod to be ready..."
    sleep 30
fi

# --- Get SSH info ---
echo "Getting SSH info for pod $POD_ID..."
SSH_INFO=$(runpodctl ssh info "$POD_ID" 2>&1 || true)
echo "$SSH_INFO"

SSH_CMD=$(echo "$SSH_INFO" | grep 'ssh ' | head -1 || true)
if [ -z "$SSH_CMD" ]; then
    echo "Waiting for SSH to be ready..."
    for i in $(seq 1 12); do
        sleep 10
        SSH_INFO=$(runpodctl ssh info "$POD_ID" 2>&1 || true)
        SSH_CMD=$(echo "$SSH_INFO" | grep 'ssh ' | head -1 || true)
        if [ -n "$SSH_CMD" ]; then break; fi
        echo "  attempt $i/12..."
    done
fi

if [ -z "$SSH_CMD" ]; then
    echo "FAILED: Could not get SSH info"
    exit 1
fi

echo "SSH: $SSH_CMD"

# Extract host and port
SSH_HOST=$(echo "$SSH_CMD" | sed -n 's/.*@\([^ :]*\).*/\1/p')
SSH_PORT=$(echo "$SSH_CMD" | sed -n 's/.*-p \([0-9]*\).*/\1/p' || echo "22")
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $SSH_PORT root@$SSH_HOST"
SCP="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

# --- Deploy ---
echo "Deploying training script..."
$SCP "$SCRIPT_DIR/exp039_gec_qwen_500m.py" "root@$SSH_HOST:/root/exp039_gec_qwen_500m.py"

echo "Installing dependencies..."
$SSH "pip install -q peft trl datasets huggingface_hub transformers accelerate torch && echo 'DEPS OK'"

echo "Setting up HF token..."
$SSH "mkdir -p ~/.cache/huggingface && echo '$HF_TOKEN' > ~/.cache/huggingface/token && huggingface-cli whoami"

# --- Verify ---
echo "Running smoke test (dry-run 5 steps)..."
$SSH "cd /root && HF_TOKEN=$HF_TOKEN python3 exp039_gec_qwen_500m.py --dry-run 2>&1 | tail -20"

echo ""
echo "============================================"
echo " Smoke test passed! Ready to launch training"
echo " Pod ID: $POD_ID"
echo " SSH: $SSH_CMD"
echo "============================================"
echo ""
echo "To launch full training:"
echo "  $SSH \"screen -dmS gec_train bash -c 'HF_TOKEN=$HF_TOKEN python3 /root/exp039_gec_qwen_500m.py --epochs 3 --lr 2e-4 --stage 7 2>&1 | tee /root/gec_train.log'\""
echo ""
echo "To monitor:"
echo "  $SSH \"tail -f /root/gec_train.log\""
echo ""
echo "To destroy after training:"
echo "  runpodctl delete pod $POD_ID"
