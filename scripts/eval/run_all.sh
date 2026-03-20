#!/usr/bin/env bash
# Run all evaluation benchmarks for all models in the registry.
#
# Usage:
#   ./scripts/eval/run_all.sh                          # all models, all tasks
#   ./scripts/eval/run_all.sh --models sozkz-50m,sozkz-150m
#   ./scripts/eval/run_all.sh --tasks bpb,mc_qa
#   ./scripts/eval/run_all.sh --limit 100              # limit samples per task
#   ./scripts/eval/run_all.sh --models sozkz-50m --tasks ner --limit 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All 14 models from MODEL_REGISTRY
ALL_MODELS=(
    sozkz-50m
    sozkz-150m
    sozkz-300m
    sozkz-600m
    gemma-2b
    gemma-9b
    llama-3-1b
    llama-3-3b
    llama-3-8b
    qwen-0.5b
    qwen-1.5b
    qwen-7b
    mistral-7b
    gpt-oss-120b
)

# All 6 eval scripts and their task subdirectory names
declare -A EVAL_SCRIPTS=(
    [eval_bpb.py]="bpb"
    [eval_mc_bench.py]="mc_qa"
    [eval_sentiment.py]="sentiment"
    [eval_belebele.py]="belebele"
    [eval_ner.py]="ner"
    [eval_sib200.py]="sib200"
)

# Models that require --quantize flag
QUANTIZE_MODELS="gemma-9b llama-3-8b qwen-7b mistral-7b"

# Parse arguments
SELECTED_MODELS=""
SELECTED_TASKS=""
LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            SELECTED_MODELS="$2"
            shift 2
            ;;
        --tasks)
            SELECTED_TASKS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--models m1,m2] [--tasks t1,t2] [--limit N]"
            exit 1
            ;;
    esac
done

# Resolve model list
if [[ -n "$SELECTED_MODELS" ]]; then
    IFS=',' read -ra MODELS <<< "$SELECTED_MODELS"
else
    MODELS=("${ALL_MODELS[@]}")
fi

# Resolve task/script list
if [[ -n "$SELECTED_TASKS" ]]; then
    IFS=',' read -ra TASK_FILTER <<< "$SELECTED_TASKS"
else
    TASK_FILTER=()
fi

echo "============================================"
echo "  SLM Evaluation Pipeline"
echo "============================================"
echo "Models: ${MODELS[*]}"
if [[ ${#TASK_FILTER[@]} -gt 0 ]]; then
    echo "Tasks:  ${TASK_FILTER[*]}"
else
    echo "Tasks:  all (${!EVAL_SCRIPTS[*]})"
fi
if [[ -n "$LIMIT" ]]; then
    echo "Limit:  $LIMIT samples per task"
fi
echo "============================================"
echo ""

TOTAL=0
PASSED=0
FAILED=0

for model in "${MODELS[@]}"; do
    echo "--- Model: $model ---"

    # Check if model needs quantization
    QUANT_FLAG=""
    if echo "$QUANTIZE_MODELS" | grep -qw "$model"; then
        QUANT_FLAG="--quantize"
    fi

    for script in "${!EVAL_SCRIPTS[@]}"; do
        task="${EVAL_SCRIPTS[$script]}"

        # Skip if task filter is set and this task isn't in it
        if [[ ${#TASK_FILTER[@]} -gt 0 ]]; then
            SKIP=true
            for t in "${TASK_FILTER[@]}"; do
                if [[ "$t" == "$task" ]]; then
                    SKIP=false
                    break
                fi
            done
            if $SKIP; then
                continue
            fi
        fi

        OUTPUT="paper/results/${task}/${model}.json"
        TOTAL=$((TOTAL + 1))

        echo -n "  [$task] $script ... "

        # Build command
        CMD="python $SCRIPT_DIR/$script --model $model --output $OUTPUT"
        if [[ -n "$QUANT_FLAG" ]]; then
            CMD="$CMD $QUANT_FLAG"
        fi
        if [[ -n "$LIMIT" ]]; then
            CMD="$CMD --limit $LIMIT"
        fi

        if $CMD 2>&1; then
            echo "OK"
            PASSED=$((PASSED + 1))
        else
            echo "FAILED"
            FAILED=$((FAILED + 1))
        fi
    done
    echo ""
done

echo "============================================"
echo "  Results: $PASSED/$TOTAL passed, $FAILED failed"
echo "============================================"

# Aggregate all results
echo ""
echo "Aggregating results..."
python "$SCRIPT_DIR/aggregate_results.py"

echo ""
echo "Done. Summary at paper/results/summary.json"
