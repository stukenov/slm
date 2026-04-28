#!/bin/bash
set -euo pipefail

HOST="${1:?Usage: $0 HOST PORT [sample|full] [N]}"
PORT="${2:?Usage: $0 HOST PORT [sample|full] [N]}"
MODE="${3:-sample}"   # sample (default for first run) or full
SAMPLE_N="${4:-5}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUNPOD_KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
KEY_OPT=""
if [[ -f "$RUNPOD_KEY" ]]; then
  KEY_OPT="-i $RUNPOD_KEY"
fi
SSH="ssh $KEY_OPT -o ConnectTimeout=30 -o StrictHostKeyChecking=no root@${HOST} -p ${PORT}"
RSYNC="rsync -az --delete -e 'ssh $KEY_OPT -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p ${PORT}'"

echo "=== Syncing repo to ${HOST}:${PORT} ==="
${SSH} "mkdir -p /workspace/slm"
eval "${RSYNC} \
  --exclude '.git' \
  --exclude '.venv*' \
  --exclude '__pycache__' \
  --exclude 'data/' \
  --exclude 'outputs/' \
  '${ROOT_DIR}/' 'root@${HOST}:/workspace/slm/'"

# Copy HF token if available
HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
if [[ -f "$HF_TOKEN_PATH" ]]; then
  ${SSH} "mkdir -p /root/.cache/huggingface"
  scp $KEY_OPT -o StrictHostKeyChecking=no -P "${PORT}" \
    "$HF_TOKEN_PATH" "root@${HOST}:/root/.cache/huggingface/token"
  echo "HF token copied."
fi

echo "=== Starting exp038 in mode=${MODE} ==="
${SSH} "cd /workspace/slm && bash scripts/exp038/run_on_pod.sh ${MODE} ${SAMPLE_N}"
