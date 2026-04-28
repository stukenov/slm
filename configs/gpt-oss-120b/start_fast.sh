#!/bin/bash
# Start GPT-OSS-120B OPTIMIZED for GEC/instruct generation
# Key changes vs start.sh:
#   - Context 128K → 4K (GEC prompts are ~200 tokens)
#   - ncmoe 30 → 24 (more MoE on GPU = 42 t/s vs 37 t/s)
#   - 2 parallel slots (-np 2)
#   - No --jinja (skip reasoning overhead, use completion API)

export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

screen -X -S gpt_infer quit 2>/dev/null
sleep 3

screen -dmS gpt_infer bash -c "
export PATH=/usr/local/cuda-12.1/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:\$LD_LIBRARY_PATH
/root/llama.cpp/build/bin/llama-server \
  -m /root/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  --alias 'GPT-OSS-120B' \
  -fa 1 -ncmoe 24 -ngl 99 --tensor-split 0.5,0.5 -fit off \
  -t 44 -ctk q4_0 -ctv q4_0 \
  -ub 2048 -b 2048 -c 4096 \
  -np 2 \
  --host 0.0.0.0 --port 15127 \
  2>&1 | tee /root/gpt_infer.log
exec bash"

echo 'GPT-OSS-120B FAST server started in screen gpt_infer'
echo '  Context: 4K, ncmoe: 24, slots: 2, no reasoning'
echo '  API: http://164.138.46.36:15127/v1/chat/completions'
