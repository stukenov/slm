#!/bin/bash
# ============================================================================
# exp030: Deploy GEC 1B experiment scripts to RunPod pod
# Reads connection info from /tmp/exp030_pod.json
# ============================================================================
set -e

POD_INFO="/tmp/exp030_pod.json"
if [ ! -f "$POD_INFO" ]; then
    echo "ERROR: $POD_INFO not found. Run exp030_launch_pod.py first."
    exit 1
fi

SSH_IP=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['ssh_ip'])")
SSH_PORT=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['ssh_port'])")
POD_ID=$(python3 -c "import json; print(json.load(open('$POD_INFO'))['pod_id'])")

SSH="ssh root@$SSH_IP -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30"
SCP="scp -P $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30"

echo "=== Deploying exp030 to pod $POD_ID ($SSH_IP:$SSH_PORT) ==="

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
$SSH "mkdir -p /root/exp030"

# Copy scripts
echo "Copying experiment scripts..."
$SCP /Users/sakentukenov/slm/autoresearch/exp030_train.py root@$SSH_IP:/root/exp030_train.py
$SCP /Users/sakentukenov/slm/autoresearch/exp030_eval.py root@$SSH_IP:/root/exp030_eval.py
$SCP /Users/sakentukenov/slm/autoresearch/exp030_wave1.sh root@$SSH_IP:/root/exp030_wave1.sh
$SCP /Users/sakentukenov/slm/autoresearch/exp030_wave2.sh root@$SSH_IP:/root/exp030_wave2.sh
$SCP /Users/sakentukenov/slm/autoresearch/exp030_final.sh root@$SSH_IP:/root/exp030_final.sh
$SCP /Users/sakentukenov/slm/autoresearch/exp030_eval_all.sh root@$SSH_IP:/root/exp030_eval_all.sh

# Install deps
echo "Installing dependencies..."
$SSH "pip install -q torch transformers datasets huggingface-hub safetensors accelerate peft 2>&1 | tail -5"

# Set HF token
$SSH "mkdir -p ~/.cache/huggingface && echo "$HF_TOKEN" > ~/.cache/huggingface/token"

# Verify
echo "Verifying setup..."
$SSH "python3 -c 'import peft; print(\"peft OK:\", peft.__version__)'"
$SSH "python3 -c 'from huggingface_hub import HfApi; print(\"HF auth:\", HfApi().whoami()[\"name\"])'"
$SSH "nvidia-smi -L"

echo ""
echo "=== Deploy complete ==="
echo "Pod: $POD_ID"
echo "SSH: $SSH"
echo ""
echo "Next steps:"
echo "  1. Smoke test:  $SSH 'cd /root && CUDA_VISIBLE_DEVICES=0 EXP_ID=smoke METHOD=lora LORA_RANK=8 LR=2e-4 CLEAN_RATIO=0.8 EPOCHS=1 MAX_STEPS=10 HF_TOKEN=\$(cat ~/.cache/huggingface/token) python3 exp030_train.py'"
echo "  2. Wave 1:      $SSH 'cd /root && screen -dmS wave1 bash exp030_wave1.sh'"
