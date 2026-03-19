#!/bin/bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

MODEL="/root/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
BENCH="/root/llama.cpp/build/bin/llama-bench"
LOG="/root/bench_results_v2.txt"

echo "=== V2 BENCHMARK $(date) ===" | tee $LOG

run_bench() {
    local label="$1"
    shift
    echo "" | tee -a $LOG
    echo ">>> $label" | tee -a $LOG
    $BENCH -m $MODEL "$@" 2>&1 | grep -E '(tg128|pp512|error)' | tee -a $LOG
    sleep 3
}

# Fine-tune ncmoe between 21-24 (2GPU)
run_bench "ncmoe=24, 2GPU, t=28" -fa 1 -ncmoe 24 -ngl 99 -ts 0.5,0.5 -t 28
run_bench "ncmoe=23, 2GPU, t=28" -fa 1 -ncmoe 23 -ngl 99 -ts 0.5,0.5 -t 28
run_bench "ncmoe=22, 2GPU, t=28" -fa 1 -ncmoe 22 -ngl 99 -ts 0.5,0.5 -t 28
run_bench "ncmoe=21, 2GPU, t=28" -fa 1 -ncmoe 21 -ngl 99 -ts 0.5,0.5 -t 28

# KV quant to free VRAM, then try lower ncmoe
run_bench "ncmoe=20, 2GPU, t=28, KV q8" -fa 1 -ncmoe 20 -ngl 99 -ts 0.5,0.5 -t 28 -ctk q8_0 -ctv q8_0
run_bench "ncmoe=18, 2GPU, t=28, KV q8" -fa 1 -ncmoe 18 -ngl 99 -ts 0.5,0.5 -t 28 -ctk q8_0 -ctv q8_0

# 1 GPU only (no tensor split overhead)
run_bench "ncmoe=25, 1GPU, t=28" -fa 1 -ncmoe 25 -ngl 99 -t 28
run_bench "ncmoe=30, 1GPU, t=28" -fa 1 -ncmoe 30 -ngl 99 -t 28

# Thread tuning with best ncmoe (25 as baseline)
run_bench "ncmoe=25, 2GPU, t=56" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 56
run_bench "ncmoe=25, 2GPU, t=40" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40
run_bench "ncmoe=25, 2GPU, t=20" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 20
run_bench "ncmoe=25, 2GPU, t=14" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 14

# Asymmetric split (A10 #0 might have less load)
run_bench "ncmoe=25, split 0.4/0.6, t=28" -fa 1 -ncmoe 25 -ngl 99 -ts 0.4,0.6 -t 28
run_bench "ncmoe=25, split 0.3/0.7, t=28" -fa 1 -ncmoe 25 -ngl 99 -ts 0.3,0.7 -t 28

# NUMA
run_bench "ncmoe=25, 2GPU, t=28, numa" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 28 --numa distribute

echo "" | tee -a $LOG
echo "=== V2 COMPLETE $(date) ===" | tee -a $LOG
