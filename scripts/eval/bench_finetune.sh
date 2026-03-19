#!/bin/bash
# Fine-tuning around best config: ncmoe=25, t=40, 2GPU 0.5/0.5 = 42.4 t/s
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Kill dual instances first
screen -X -S gpt_gpu0 quit 2>/dev/null
screen -X -S gpt_gpu1 quit 2>/dev/null
screen -X -S gpt_infer quit 2>/dev/null
sleep 3

MODEL="/root/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
BENCH="/root/llama.cpp/build/bin/llama-bench"
LOG="/root/bench_finetune.txt"

echo "=== FINETUNE BENCHMARK $(date) ===" | tee $LOG

run() {
    local label="$1"; shift
    echo "" | tee -a $LOG
    echo ">>> $label" | tee -a $LOG
    $BENCH -m $MODEL "$@" 2>&1 | grep -E '(tg128|pp512)' | tee -a $LOG
    sleep 3
}

echo "--- PHASE 1: Thread fine-tuning (ncmoe=25) ---" | tee -a $LOG
run "t=36, ncmoe=25" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 36
run "t=38, ncmoe=25" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 38
run "t=40, ncmoe=25 (baseline)" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40
run "t=42, ncmoe=25" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 42
run "t=44, ncmoe=25" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 44
run "t=48, ncmoe=25" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 48

echo "" | tee -a $LOG
echo "--- PHASE 2: ncmoe tuning with KV q8 (frees ~150MB VRAM) ---" | tee -a $LOG
run "ncmoe=24, t=40, KV q8" -fa 1 -ncmoe 24 -ngl 99 -ts 0.5,0.5 -t 40 -ctk q8_0 -ctv q8_0
run "ncmoe=23, t=40, KV q8" -fa 1 -ncmoe 23 -ngl 99 -ts 0.5,0.5 -t 40 -ctk q8_0 -ctv q8_0
run "ncmoe=25, t=40, KV q8" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40 -ctk q8_0 -ctv q8_0
run "ncmoe=26, t=40" -fa 1 -ncmoe 26 -ngl 99 -ts 0.5,0.5 -t 40
run "ncmoe=27, t=40" -fa 1 -ncmoe 27 -ngl 99 -ts 0.5,0.5 -t 40
run "ncmoe=28, t=40" -fa 1 -ncmoe 28 -ngl 99 -ts 0.5,0.5 -t 40

echo "" | tee -a $LOG
echo "--- PHASE 3: Batch size tuning ---" | tee -a $LOG
run "ub=2048,b=2048, ncmoe=25, t=40" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40 -ub 2048 -b 2048
run "ub=512,b=512, ncmoe=25, t=40" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40 -ub 512 -b 512
run "ub=8192,b=8192, ncmoe=25, t=40" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40 -ub 8192 -b 8192

echo "" | tee -a $LOG
echo "--- PHASE 4: Context size ---" | tee -a $LOG
run "c=4096, ncmoe=25, t=40" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40

echo "" | tee -a $LOG
echo "--- PHASE 5: Split-mode row vs layer ---" | tee -a $LOG
run "split-mode=row, ncmoe=25, t=40" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40 -sm row

echo "" | tee -a $LOG
echo "--- PHASE 6: mmap ---" | tee -a $LOG
run "mmap=1, ncmoe=25, t=40" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 40 -mmp 1

echo "" | tee -a $LOG
echo "--- PHASE 7: Best combo candidates ---" | tee -a $LOG
# Combine best thread + KV quant + any improvements found
run "COMBO: ncmoe=24, t=42, KV q8" -fa 1 -ncmoe 24 -ngl 99 -ts 0.5,0.5 -t 42 -ctk q8_0 -ctv q8_0
run "COMBO: ncmoe=25, t=44, KV q8" -fa 1 -ncmoe 25 -ngl 99 -ts 0.5,0.5 -t 44 -ctk q8_0 -ctv q8_0
run "COMBO: ncmoe=24, t=44, KV q8" -fa 1 -ncmoe 24 -ngl 99 -ts 0.5,0.5 -t 44 -ctk q8_0 -ctv q8_0

echo "" | tee -a $LOG
echo "=== FINETUNE COMPLETE $(date) ===" | tee -a $LOG
