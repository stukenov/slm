#!/usr/bin/env bash
# RunPod launch script for FineWeb-Edu sample-100BT EN→KK translation.
#
# Recommended pod: 8× RTX 4090 ($3.04/hr) → ~36h → ~$108
# Alternative:     4× RTX 4090 ($1.48/hr) → ~73h → ~$108
#
# Usage on RunPod:
#   bash run_100bt.sh              # Full run, all GPUs
#   bash run_100bt.sh --smoke      # Smoke test only
#   bash run_100bt.sh --resume     # Resume from last checkpoint
#
# Multi-node (2 pods, each 4×4090):
#   Pod 1: bash run_100bt.sh --start 0 --end 50
#   Pod 2: bash run_100bt.sh --start 50 --end 100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse args
SMOKE=false
RESUME=false
START_CHUNK="0"
END_CHUNK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=true; shift ;;
        --resume) RESUME=true; START_CHUNK="auto"; shift ;;
        --start) START_CHUNK="$2"; shift 2 ;;
        --end) END_CHUNK="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Detect GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected!"
    exit 1
fi
echo "=== Detected $NUM_GPUS GPU(s) ==="
nvidia-smi -L

# Step 1: Install dependencies
echo ""
echo "=== Step 1: Install dependencies ==="
pip install -q ctranslate2 sentencepiece datasets huggingface_hub xxhash pyarrow tqdm

# Step 2: Download and convert model
echo ""
echo "=== Step 2: Setup translation model ==="
if [ -f "model_ct2/model.bin" ]; then
    echo "CT2 model already exists, skipping."
else
    bash setup_model.sh
fi

# Step 3: Verify HF auth
echo ""
echo "=== Step 3: Verify HuggingFace auth ==="
python -c "
from huggingface_hub import HfApi
api = HfApi()
info = api.whoami()
print(f'Logged in as: {info[\"name\"]}')
print('HF auth OK')
"

# Step 4: Smoke test
echo ""
echo "=== Step 4: Smoke test ==="
python pipeline_100bt.py --smoke-test

if [ "$SMOKE" = true ]; then
    echo ""
    echo "Smoke test passed! Exiting (--smoke mode)."
    exit 0
fi

# Step 5: Run translation
echo ""
echo "=== Step 5: Starting translation ==="
echo "GPUs: $NUM_GPUS"
echo "Start chunk: $START_CHUNK"
echo "End chunk: ${END_CHUNK:-auto (100)}"
echo ""

CMD="python pipeline_100bt.py --num-gpus $NUM_GPUS --start-chunk $START_CHUNK"
if [ -n "$END_CHUNK" ]; then
    CMD="$CMD --end-chunk $END_CHUNK"
fi

echo "Running: $CMD"
echo "Started at: $(date)"
echo ""

# Run in foreground so we see output. Use screen/tmux externally if needed.
$CMD

echo ""
echo "=== Translation complete ==="
echo "Finished at: $(date)"
echo ""

# Step 6: Final verification
echo "=== Step 6: Final verification ==="
python -c "
import json, os
prog_file = 'progress_100bt.json'
if os.path.exists(prog_file):
    with open(prog_file) as f:
        p = json.load(f)
    print(f'Chunks completed: {len(p[\"chunks_completed\"])}')
    print(f'Chunks verified:  {len(p[\"chunks_verified\"])}')
    print(f'Total rows:       {p[\"total_rows_translated\"]:,}')
    print(f'Total time:       {p[\"total_elapsed_sec\"]/3600:.1f}h')

    # Check for gaps
    completed = sorted(p['chunks_completed'])
    if completed:
        expected = set(range(completed[0], completed[-1]+1))
        missing = expected - set(completed)
        if missing:
            print(f'WARNING: Missing chunks: {sorted(missing)}')
        else:
            print(f'All chunks {completed[0]}-{completed[-1]} present, no gaps.')
else:
    print('No progress file found.')
"
