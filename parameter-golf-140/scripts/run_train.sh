#!/bin/bash
# Run training experiment on the pod
# Usage:
#   bash run_train.sh quick          # 500 steps, no wallclock limit
#   bash run_train.sh single         # full run, 1 GPU, 600s
#   bash run_train.sh 8gpu           # full run, 8xH100, 600s
#   bash run_train.sh 3seed          # 3-seed validation, 8xH100
set -e

MODE="${1:-quick}"
WORK="/workspace/parameter-golf"
cd "$WORK"

# Load config
if [ -f /workspace/config.env ]; then
    set -a; source /workspace/config.env; set +a
    echo "Loaded config from /workspace/config.env"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case "$MODE" in
    quick)
        echo "=== Quick ablation: 500 steps ==="
        ITERATIONS=500 \
        VAL_LOSS_EVERY=100 \
        TRAIN_LOG_EVERY=50 \
        MAX_WALLCLOCK_SECONDS=0 \
        python train_gpt.py 2>&1 | tee "/workspace/logs/quick_${TIMESTAMP}.log"
        ;;
    single)
        echo "=== Full run: 1 GPU, 600s ==="
        MAX_WALLCLOCK_SECONDS=600 \
        python train_gpt.py 2>&1 | tee "/workspace/logs/single_${TIMESTAMP}.log"
        ;;
    8gpu)
        echo "=== Full run: 8xH100, 600s ==="
        MAX_WALLCLOCK_SECONDS=600 \
        torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee "/workspace/logs/8gpu_${TIMESTAMP}.log"
        ;;
    3seed)
        echo "=== 3-seed validation ==="
        for S in 1337 42 2025; do
            echo "--- Seed $S ---"
            SEED=$S MAX_WALLCLOCK_SECONDS=600 \
            torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee "/workspace/logs/seed${S}_${TIMESTAMP}.log"
        done
        ;;
    *)
        echo "Usage: bash run_train.sh {quick|single|8gpu|3seed}"
        exit 1
        ;;
esac

# Show final results
echo ""
echo "=== Results ==="
grep -E "val_bpb|final_int8|submission size" /workspace/logs/*_${TIMESTAMP}.log 2>/dev/null || true
