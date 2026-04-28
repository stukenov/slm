#!/bin/bash
# Full pipeline: setup + train + SLOT eval on RunPod H100
# Usage: run_on_pod.sh <ssh_host> <ssh_port> [seed]
set -e
SSH_HOST="${1:?Usage: run_on_pod.sh <ssh_host> <ssh_port> [seed]}"
SSH_PORT="${2:?Usage: run_on_pod.sh <ssh_host> <ssh_port> [seed]}"
SEED="${3:-1337}"
SSH="ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_HOST"
SCP="scp -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_PORT"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== [1/5] Clone repo + install deps ==="
$SSH "cd /workspace && \
  ([ -d parameter-golf ] || git clone https://github.com/openai/parameter-golf.git) && \
  pip install -q sentencepiece datasets huggingface-hub zstandard tiktoken numpy tqdm && \
  echo 'deps OK'"

echo ""
echo "=== [2/5] Download data ==="
$SSH "cd /workspace/parameter-golf && \
  ([ -f data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ] || python3 data/cached_challenge_fineweb.py) && \
  echo 'data OK: shards=' && ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l"

echo ""
echo "=== [3/5] Install flash-attn-3 ==="
$SSH "pip install flash-attn --no-build-isolation 2>&1 | tail -3 || \
  pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu124_torch251 2>&1 | tail -3 || \
  echo 'FA3 failed, will use SDPA fallback'"

echo ""
echo "=== [4/5] Upload train_gpt.py ==="
$SCP "$SCRIPT_DIR/reference_prs/pr1263_train_gpt.py" "$SSH_HOST:/workspace/parameter-golf/train_gpt.py"
$SSH "wc -l /workspace/parameter-golf/train_gpt.py"

echo ""
echo "=== [5/5] Launch training (seed=$SEED, 1xH100, screen) ==="
$SSH "cd /workspace/parameter-golf && screen -dmS train bash -c '\
  SEED=$SEED MAX_WALLCLOCK_SECONDS=600 SLOT_ENABLED=1 SLOT_STEPS=16 \
  python3 train_gpt.py 2>&1 | tee /workspace/run_seed${SEED}.log; \
  echo DONE'"

echo ""
echo "=== LAUNCHED ==="
echo "Monitor: $SSH \"tail -f /workspace/run_seed${SEED}.log\""
echo "Attach:  $SSH \"screen -r train\""
