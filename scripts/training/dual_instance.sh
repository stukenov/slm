#!/bin/bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

MODEL="/root/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
SERVER="/root/llama.cpp/build/bin/llama-server"

# Kill existing server
screen -X -S gpt_infer quit 2>/dev/null
sleep 2

# Instance 1: GPU 0, port 15127
screen -dmS gpt_gpu0 bash -c "
CUDA_VISIBLE_DEVICES=0 $SERVER \
  -m $MODEL \
  -fa 1 -cmoe -ngl 99 \
  -ub 4096 -b 4096 \
  -c 8192 -t 28 \
  --jinja --host 0.0.0.0 --port 15127 \
  2>&1 | tee /root/gpt_gpu0.log
exec bash"

echo "Instance 1 (GPU 0, port 15127) started"

# Instance 2: GPU 1, port 15128
screen -dmS gpt_gpu1 bash -c "
CUDA_VISIBLE_DEVICES=1 $SERVER \
  -m $MODEL \
  -fa 1 -cmoe -ngl 99 \
  -ub 4096 -b 4096 \
  -c 8192 -t 28 \
  --jinja --host 0.0.0.0 --port 15128 \
  2>&1 | tee /root/gpt_gpu1.log
exec bash"

echo "Instance 2 (GPU 1, port 15128) started"
echo "Waiting for servers to load..."
sleep 45

# Check both are up
echo "=== GPU 0 status ==="
tail -3 /root/gpt_gpu0.log
echo "=== GPU 1 status ==="
tail -3 /root/gpt_gpu1.log
