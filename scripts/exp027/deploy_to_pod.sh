#!/bin/bash
# Deploy exp027 data prep pipeline to a RunPod pod.
#
# Usage: bash scripts/exp027/deploy_to_pod.sh <HOST> <PORT>
# Example: bash scripts/exp027/deploy_to_pod.sh 194.68.245.22 22345

set -euo pipefail

HOST="${1:?Usage: $0 HOST PORT}"
PORT="${2:?Usage: $0 HOST PORT}"
SSH="ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no root@${HOST} -p ${PORT}"
SCP="scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no -P ${PORT}"

echo "=== Deploying exp027 to ${HOST}:${PORT} ==="

# 1. Copy scripts
echo "Copying scripts..."
${SCP} scripts/exp027/prepare_bilingual_data.py "root@${HOST}:/workspace/prepare_bilingual_data.py"
${SCP} scripts/exp027/run_on_pod.sh "root@${HOST}:/workspace/run_on_pod.sh"

# 2. Setup and run
echo "Starting setup + pipeline on pod..."
${SSH} "bash /workspace/run_on_pod.sh"

echo ""
echo "=== Pipeline running on pod ==="
echo "Monitor: ${SSH} 'tail -f /workspace/exp027.log'"
echo "Check:   ${SSH} 'cat /workspace/exp027.log | tail -50'"
