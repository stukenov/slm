#!/usr/bin/env bash
# Download HPLT EN→KK translation model and convert to CTranslate2 format.
# Run this once before using the pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="HPLT/translate-en-kk-v2.0-hplt_opus"
CACHE_DIR="model_cache"
CT2_DIR="model_ct2"

echo "=== Step 1: Download model ==="
if [ -d "$CACHE_DIR" ] && [ -f "$CACHE_DIR/model.en-kk.spm" ]; then
    echo "Model cache already exists, skipping download."
else
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_NAME', local_dir='$CACHE_DIR')
print('Model downloaded to $CACHE_DIR')
"
fi

echo "=== Step 2: Fix vocab format ==="
VOCAB_FIXED="$CACHE_DIR/vocab_fixed.txt"
if [ -f "$VOCAB_FIXED" ]; then
    echo "Fixed vocab already exists, skipping."
else
    python << 'PYEOF'
import os
vocab = os.path.join('model_cache', 'model.en-kk.vocab')
fixed = os.path.join('model_cache', 'vocab_fixed.txt')
with open(vocab) as f, open(fixed, 'w') as out:
    for i, line in enumerate(f):
        token = line.strip().split('\t')[0]
        out.write(f'"{token}":{i}\n')
print(f'Fixed vocab written to {fixed}')
PYEOF
fi

echo "=== Step 3: Convert to CTranslate2 ==="
if [ -d "$CT2_DIR" ] && [ -f "$CT2_DIR/model.bin" ]; then
    echo "CT2 model already exists, skipping conversion."
else
    python -c "
from ctranslate2.converters.marian import MarianConverter
npz = '$CACHE_DIR/model.npz.best-chrf.npz'
vocab = '$CACHE_DIR/vocab_fixed.txt'
MarianConverter(npz, [vocab]).convert('$CT2_DIR')
print('CT2 model saved to $CT2_DIR')
"
fi

echo "=== Done ==="
echo "Model ready. You can now run the pipeline."
