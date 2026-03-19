"""Generate the bash script that runs on the remote vast.ai instance."""

from __future__ import annotations


def generate_run_script(
    *,
    config_path: str,
    hf_repo: str,
    experiment_name: str,
    num_gpus: int = 1,
    extra_train_args: str = "",
    instance_id: int | None = None,
    vast_api_key: str = "",
    train_module: str = "slm.train",
    pre_train_cmd: str = "",
) -> str:
    """Return a bash script that trains, publishes, and self-destructs.

    Safety: if training or upload fails, the instance is NOT destroyed
    so the user can SSH in and debug.
    """
    # Determine train command based on GPU count
    if num_gpus > 1:
        train_cmd = (
            f"torchrun --nproc_per_node={num_gpus} "
            f"-m {train_module} --config {config_path}"
        )
    else:
        train_cmd = f"python -m {train_module} --config {config_path}"

    if extra_train_args:
        train_cmd += f" {extra_train_args}"

    script = f"""\
#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
LOGFILE="/root/slm/logs/cloud_{experiment_name}.log"
mkdir -p /root/slm/logs

exec > >(tee -a "$LOGFILE") 2>&1

# datasets library downloads parquets + generates arrow cache (~60GB total)
# Keep everything on overlay (must have enough space, ~100GB+)
echo "Disk space:"
df -h /

echo "=== Cloud training started: $(date -u) ==="
echo "Instance: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\\n' ', ')"
echo "GPU count: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Config: {config_path}"
echo "HF repo: {hf_repo}"
echo ""

cd /root/slm
{f"""
# ---- Phase 0: Pre-training command ----
echo ">>> Phase 0: Running pre-training command..."
if ! {pre_train_cmd}; then
    echo "!!! Pre-training command FAILED. Instance kept alive for debugging."
    exit 1
fi
echo ">>> Pre-training command complete."
""" if pre_train_cmd else ""}
# ---- Phase 1: Training ----
echo ">>> Phase 1/4: Training..."
if ! {train_cmd}; then
    echo "!!! Training FAILED. Instance kept alive for debugging."
    exit 1
fi
echo ">>> Training complete."

# Find the final model directory (train.py saves to outputs/<name>/final)
if [ -d "/root/slm/outputs/{experiment_name}/final" ]; then
    MODEL_DIR="/root/slm/outputs/{experiment_name}/final"
else
    MODEL_DIR=$(ls -td /root/slm/outputs/{experiment_name}/checkpoint-* 2>/dev/null | head -1)
    if [ -z "$MODEL_DIR" ]; then
        MODEL_DIR="/root/slm/outputs/{experiment_name}"
    fi
fi
echo ">>> Model dir: $MODEL_DIR"

# ---- Phase 2: Upload to HuggingFace ----
echo ">>> Phase 2/4: Uploading to HuggingFace ({hf_repo})..."
UPLOAD_OK=0
for attempt in 1 2 3; do
    echo "  Upload attempt $attempt/3..."
    if python -m slm.publish --model_path "$MODEL_DIR" --repo_name "{hf_repo}"; then
        UPLOAD_OK=1
        break
    fi
    echo "  Upload attempt $attempt failed, retrying in 30s..."
    sleep 30
done

if [ $UPLOAD_OK -ne 1 ]; then
    echo "!!! Upload to HF FAILED after 3 attempts. Instance kept alive for debugging."
    exit 1
fi
echo ">>> Upload complete."

# ---- Phase 3: Verify upload ----
echo ">>> Phase 3/4: Verifying upload..."
if python -c "from huggingface_hub import HfApi; info = HfApi().model_info('{hf_repo}'); print(f'Verified: {{info.id}}, size={{info.siblings.__len__()}} files')"; then
    echo ">>> Verification passed."
else
    echo "!!! Verification failed — model may not have uploaded correctly."
    echo "!!! Instance kept alive for debugging."
    exit 1
fi

# ---- Phase 4: Self-destruct ----
echo ">>> Phase 4/4: Self-destruct in 60s..."
echo "=== Cloud training finished successfully: $(date -u) ==="
sleep 60

# Best-effort self-destruct via vast.ai REST API
INSTANCE_ID="{instance_id or ''}"
VAST_KEY="{vast_api_key}"
if [ -n "$INSTANCE_ID" ] && [ -n "$VAST_KEY" ]; then
    echo ">>> Destroying instance $INSTANCE_ID..."
    curl -s -X PUT -H "Authorization: Bearer $VAST_KEY" \
        "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" \
        -d '{{"state": "stopped"}}' || echo "WARNING: self-destruct API call failed"
    curl -s -X DELETE -H "Authorization: Bearer $VAST_KEY" \
        "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" \
        || echo "WARNING: self-destruct delete failed"
    echo ">>> Self-destruct request sent."
else
    echo "WARNING: Missing instance ID or API key for self-destruct."
fi
"""
    return script


def generate_onstart_script(hf_token: str) -> str:
    """Generate the onstart command that runs when the instance boots.

    Installs dependencies and configures the environment.
    """
    return f"""\
#!/bin/bash
set -e
echo "=== Instance onstart: $(date -u) ==="

# Configure HF token
mkdir -p ~/.cache/huggingface
echo '{hf_token}' > ~/.cache/huggingface/token
export HF_TOKEN='{hf_token}'

# Install vast.ai CLI for self-destruct
pip install -q vastai 2>/dev/null || true

echo "=== Onstart complete ==="
"""
