#!/bin/bash
# Benchmark GPT-OSS-120B with different configurations
# Server: 2x A10 24GB, 56-core Xeon Gold 6330N, 1TB RAM

export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

MODEL="/root/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
BENCH="/root/llama.cpp/build/bin/llama-bench"
LOG="/root/bench_results.txt"

echo "========================================" | tee $LOG
echo "GPT-OSS-120B Benchmark $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG

run_bench() {
    local label="$1"
    shift
    echo "" | tee -a $LOG
    echo ">>> TEST: $label" | tee -a $LOG
    echo ">>> ARGS: $@" | tee -a $LOG
    $BENCH -m $MODEL "$@" -r 3 2>&1 | tee -a $LOG
    echo "" | tee -a $LOG
    # cool down
    sleep 5
}

# === Test 1: Current config (baseline) ===
run_bench "BASELINE: ncmoe=30, 2GPU split, t=28" \
    -fa 1 -ncmoe 30 -ngl 99 --tensor-split 0.5,0.5 -t 28

# === Test 2: Lower ncmoe = more layers on GPU ===
run_bench "ncmoe=25, 2GPU split, t=28" \
    -fa 1 -ncmoe 25 -ngl 99 --tensor-split 0.5,0.5 -t 28

run_bench "ncmoe=20, 2GPU split, t=28" \
    -fa 1 -ncmoe 20 -ngl 99 --tensor-split 0.5,0.5 -t 28

run_bench "ncmoe=15, 2GPU split, t=28" \
    -fa 1 -ncmoe 15 -ngl 99 --tensor-split 0.5,0.5 -t 28

run_bench "ncmoe=10, 2GPU split, t=28" \
    -fa 1 -ncmoe 10 -ngl 99 --tensor-split 0.5,0.5 -t 28

# === Test 3: Single GPU (no split overhead) ===
run_bench "ncmoe=23, 1GPU only, t=28" \
    -fa 1 -ncmoe 23 -ngl 99 -t 28

# === Test 4: cmoe (minimal VRAM, 2GPU) ===
run_bench "cmoe (minimal VRAM), 2GPU, t=28" \
    -fa 1 -cmoe -ngl 99 --tensor-split 0.5,0.5 -t 28

# === Test 5: Thread count tuning ===
run_bench "ncmoe=20, 2GPU, t=56 (all cores)" \
    -fa 1 -ncmoe 20 -ngl 99 --tensor-split 0.5,0.5 -t 56

run_bench "ncmoe=20, 2GPU, t=16" \
    -fa 1 -ncmoe 20 -ngl 99 --tensor-split 0.5,0.5 -t 16

run_bench "ncmoe=20, 2GPU, t=8" \
    -fa 1 -ncmoe 20 -ngl 99 --tensor-split 0.5,0.5 -t 8

# === Test 6: NUMA optimization ===
run_bench "ncmoe=20, 2GPU, t=28, numa=distribute" \
    -fa 1 -ncmoe 20 -ngl 99 --tensor-split 0.5,0.5 -t 28 --numa distribute

echo "" | tee -a $LOG
echo "========================================" | tee -a $LOG
echo "BENCHMARK COMPLETE $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG
