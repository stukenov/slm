#!/bin/bash
# Start GPT-OSS-120B server on kaznu
# Usage: ssh kaznu "bash /root/slm/configs/gpt-oss-120b/start.sh"

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
  -fa 1 -ncmoe 30 -ngl 99 --tensor-split 0.5,0.5 -fit off \
  -t 44 -ctk q4_0 -ctv q4_0 \
  -ub 2048 -b 2048 -c 131072 \
  --jinja --chat-template-file /root/chat_template.jinja \
  --temp 0.5 --top-p 0.9 --min-p 0.05 --top-k 40 \
  --repeat-penalty 1.1 \
  --path /root/webui \
  --host 0.0.0.0 --port 15127 \
  2>&1 | tee /root/gpt_infer.log
exec bash"

echo "GPT-OSS-120B server started in screen 'gpt_infer'"
echo "GUI: http://164.138.46.36:15127"
echo "API: http://164.138.46.36:15127/v1/chat/completions"
