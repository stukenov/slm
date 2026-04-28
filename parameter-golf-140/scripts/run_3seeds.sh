#!/bin/bash
# Run 3-seed validation for a submission
# Usage: run_3seeds.sh <safe|aggressive>
# Run on 8xH100 inside /workspace/parameter-golf/
set -e
MODE="${1:?Usage: run_3seeds.sh <safe|aggressive>}"

if [ "$MODE" = "safe" ]; then
    SCRIPT="run_safe.sh"
elif [ "$MODE" = "aggressive" ]; then
    SCRIPT="run_aggressive.sh"
else
    echo "Unknown mode: $MODE (use safe or aggressive)"
    exit 1
fi

SCRIPT_DIR="$(dirname "$0")"

echo "=== 3-seed validation: $MODE ==="
echo ""

for SEED in 1337 42 2025; do
    echo "=========================================="
    echo "  SEED $SEED — $(date)"
    echo "=========================================="
    bash "$SCRIPT_DIR/$SCRIPT" "$SEED"
    echo ""
    echo "Seed $SEED complete. Cooling 10s..."
    sleep 10
done

echo ""
echo "=== ALL 3 SEEDS COMPLETE ==="
echo "Logs:"
if [ "$MODE" = "safe" ]; then
    ls -la /workspace/safe_seed*.log
else
    ls -la /workspace/aggressive_seed*.log
fi

echo ""
echo "Extract BPB results:"
for SEED in 1337 42 2025; do
    if [ "$MODE" = "safe" ]; then
        LOG="/workspace/safe_seed${SEED}.log"
    else
        LOG="/workspace/aggressive_seed${SEED}.log"
    fi
    echo "seed=$SEED:"
    grep -E 'final_int6_sliding_window_exact|final_slot_exact|final_causal_slot_exact' "$LOG" 2>/dev/null || echo "  (no final results found)"
done
