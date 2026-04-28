#!/bin/bash
# Run v4 on 8xH100 — 3 seeds for submission
# Expected: ~20 min per seed (600s train + ~600s eval/TTT)
# Total: ~60 min = ~$20

set -euo pipefail
cd /workspace/parameter-golf

# Common env vars
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=20000
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=64
export VAL_LOSS_EVERY=4000
export TRAIN_LOG_EVERY=500
export WARMDOWN_ITERS=3500

# v4 features
export XSA_LAST_N=11
export GATED_ATTENTION=1
export VALUE_RESIDUAL=1
export CROWNQ_LAMBDA=0.01
export CROWNQ_WARMDOWN_ONLY=1
export RECUR_LAYERS="4,5"
export USE_MIXER=1
export MIXER_ETA=0.1
export LATE_QAT_THRESHOLD=0.15
export SWA_ENABLED=1

# TTT config
export TTT_ENABLED=1
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0
export TTT_CHUNK_TOKENS=32768
export TTT_LR=0.002
export TTT_MOMENTUM=0.9

for SEED_VAL in 1337 42 2025; do
    echo ""
    echo "================================================================"
    echo "=== SEED $SEED_VAL — $(date) ==="
    echo "================================================================"
    echo ""

    export SEED=$SEED_VAL
    export RUN_ID="v4_seed${SEED_VAL}"

    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "/workspace/v4_seed${SEED_VAL}.log"

    # Save artifacts
    cp final_model.int6.ptz "/workspace/final_model_seed${SEED_VAL}.int6.ptz" 2>/dev/null || true
    echo "=== SEED $SEED_VAL DONE — $(date) ==="
done

echo ""
echo "================================================================"
echo "=== ALL 3 SEEDS COMPLETE ==="
echo "================================================================"
echo ""
echo "Logs:"
for s in 1337 42 2025; do
    echo "  /workspace/v4_seed${s}.log"
    grep -E "legal_ttt_exact|final_int6_sliding_window_s64_exact|Total submission" "/workspace/v4_seed${s}.log" 2>/dev/null || true
done
