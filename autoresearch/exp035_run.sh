#!/bin/bash
# exp035: Full pipeline — Stage 1 (pretrain) then Stage 2 (SFT LoRA)
# Run on RunPod H100 80GB

set -e

echo "=== exp035 Stage 1: Continue Pretrain ==="
python3 /root/exp035_translate_pretrain.py 2>&1 | tee /root/exp035_stage1.log
echo "Stage 1 done."

echo ""
echo "=== exp035 Stage 2: SFT LoRA ==="
pip install peft 2>&1 | tail -1
python3 /root/exp035_translate_sft_lora.py 2>&1 | tee /root/exp035_stage2.log
echo "Stage 2 done."

echo ""
echo "=== ALL DONE ==="
