#!/bin/bash
set -euo pipefail
pkill -f annotate_kk 2>/dev/null || true
pkill -f add_russian 2>/dev/null || true
export HF_TOKEN="REDACTED_HF_TOKEN"

echo "=== Phase 1: Annotate Kazakh dataset ==="
python3 /workspace/annotate_kk_dataset.py 2>&1 | tee /workspace/exp027/phase1.log

echo ""
echo "=== Phase 2: Add Russian + Parallel ==="
python3 /workspace/add_russian_and_parallel.py 2>&1 | tee /workspace/exp027/phase2.log

echo ""
echo "=== ALL DONE ==="
