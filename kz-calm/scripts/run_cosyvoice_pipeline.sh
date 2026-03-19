#!/bin/bash
set -e
cd /root/slm/CosyVoice
source /root/slm/.venv/bin/activate
export PYTHONPATH=/root/slm/CosyVoice:$PYTHONPATH
export PYTHONPATH=/root/slm/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH

DATA_DIR=/root/slm/CosyVoice/data/kazakh-train
MODEL_DIR=/root/slm/CosyVoice/pretrained_models/CosyVoice2-0.5B

echo "=== Step 1: Extract speaker embeddings ==="
python /root/slm/kz-calm/scripts/cosyvoice_extract_with_sf.py embeddings \
    --dir $DATA_DIR \
    --onnx_path $MODEL_DIR/campplus.onnx
echo "Embeddings done!"

echo "=== Step 2: Extract speech tokens ==="
python /root/slm/kz-calm/scripts/cosyvoice_extract_with_sf.py tokens \
    --dir $DATA_DIR \
    --onnx_path $MODEL_DIR/speech_tokenizer_v2.onnx
echo "Speech tokens done!"

echo "=== Step 3: Make parquet ==="
python tools/make_parquet_list.py \
    --num_utts_per_parquet 1000 \
    --num_processes 4 \
    --src_dir $DATA_DIR \
    --des_dir $DATA_DIR/parquet
echo "Parquet done!"

echo "=== Step 4: Train LLM ==="
torchrun --nproc_per_node=2 cosyvoice/bin/train.py \
    --config conf/cosyvoice2.yaml \
    --train_data $DATA_DIR/data.list \
    --model llm \
    --checkpoint $MODEL_DIR/llm.pt \
    --model_dir /root/slm/outputs/cosyvoice_kk/llm \
    --num_workers 4 \
    --prefetch 100 \
    --pin_memory \
    --use_amp
echo "LLM training done!"
