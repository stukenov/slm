#!/bin/bash
# Full deploy + run pipeline for 8xH100 RunPod pod
# Usage: deploy_and_run.sh <ssh_host> <ssh_port> <safe|aggressive>
set -e
HOST="${1:?Usage: deploy_and_run.sh <host> <port> <safe|aggressive>}"
PORT="${2:?Usage: deploy_and_run.sh <host> <port> <safe|aggressive>}"
MODE="${3:?Usage: deploy_and_run.sh <host> <port> <safe|aggressive>}"
SSH="ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $PORT root@$HOST"
SCP="scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $PORT"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== [1/4] Clone + deps ==="
$SSH "cd /workspace && \
  ([ -d parameter-golf ] || git clone --depth 1 https://github.com/openai/parameter-golf.git) && \
  pip install -q sentencepiece datasets huggingface-hub zstandard tiktoken numpy tqdm && \
  echo OK"

echo ""
echo "=== [2/4] Download data + install FA3 ==="
$SSH "cd /workspace/parameter-golf && \
  ([ -f data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ] || python3 data/cached_challenge_fineweb.py) && \
  pip install flash-attn --no-build-isolation 2>&1 | tail -3 && \
  echo OK"

echo ""
echo "=== [3/4] Upload scripts ==="
if [ "$MODE" = "safe" ]; then
    $SCP "$DIR/train_gpt_v6_safe.py" "root@$HOST:/workspace/parameter-golf/train_gpt.py"
else
    $SCP "$DIR/train_gpt_v6.py" "root@$HOST:/workspace/parameter-golf/train_gpt.py"
fi
$SCP "$DIR/scripts/run_safe.sh" "$DIR/scripts/run_aggressive.sh" "$DIR/scripts/run_3seeds.sh" \
    "root@$HOST:/workspace/parameter-golf/"
$SSH "chmod +x /workspace/parameter-golf/run_*.sh"
$SSH "wc -l /workspace/parameter-golf/train_gpt.py"

echo ""
echo "=== [4/4] Launch 3-seed run in background ==="
$SSH "cd /workspace/parameter-golf && \
  nohup bash run_3seeds.sh $MODE > /workspace/run_3seeds_${MODE}.log 2>&1 &
  echo 'LAUNCHED PID='\$!"

echo ""
echo "=== DEPLOYED ==="
echo "Monitor: $SSH \"tail -f /workspace/run_3seeds_${MODE}.log\""
echo "Check:   $SSH \"grep final_ /workspace/${MODE}_seed*.log\""
