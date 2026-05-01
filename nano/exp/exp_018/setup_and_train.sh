#!/bin/bash
set -euo pipefail

TG_TOKEN="8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g"
TG_CHAT="47474471"

tg() {
    curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        -d chat_id="$TG_CHAT" \
        -d parse_mode="Markdown" \
        -d text="$1" > /dev/null 2>&1 || true
}

tg "🚀 *exp018* started
Vocab: 256K | Backend: qazcorpora
Instance: c7i.4xlarge (16 vCPU, 32GB)"

echo "=== exp_018: Morpheme BPE 256K Tokenizer ==="
echo "Started: $(date)"
echo ""

cd /home/ubuntu/exp018

echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1

echo "[2/5] Creating venv and installing Python deps..."
python3 -m venv .venv
source .venv/bin/activate

pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet datasets tokenizers transformers tqdm huggingface_hub pyarrow

echo "[3/5] Configuring HuggingFace..."
mkdir -p ~/.cache/huggingface
cp /home/ubuntu/exp018/.hf_token ~/.cache/huggingface/token
HF_TOKEN=$(cat ~/.cache/huggingface/token)
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null || echo "unknown")
echo "  HF user: $HF_USER"

tg "📦 *exp018* deps installed, HF auth OK ($HF_USER)
Starting segmentation..."

echo "[4/5] Training tokenizer (256K vocab, qazcorpora backend)..."
echo ""

python3 train_tokenizer.py \
    --vocab-size 256000 \
    --backend qazcorpora \
    --save-corpus-hf stukenov/sozkz-corpus-segmented-kk-v1 \
    --push stukenov/sozkz-morphbpe-256k-kk-v1 2>&1 | while IFS= read -r line; do
    echo "$line"
    # Send progress every 500K docs
    if echo "$line" | grep -qE '\[seg\].*docs'; then
        tg "📊 $line"
    fi
    # Corpus upload done
    if echo "$line" | grep -q 'Upload.*Done'; then
        tg "✅ *Corpus uploaded to HF!*
$line"
    fi
    # Training done
    if echo "$line" | grep -q 'Training done'; then
        tg "🏋️ *BPE training done!*
$line"
    fi
    # Tokenizer pushed
    if echo "$line" | grep -q 'Upload complete'; then
        tg "✅ *Tokenizer uploaded to HF!*"
    fi
    # Fertility result
    if echo "$line" | grep -q 'fertility'; then
        tg "📏 $line"
    fi
done

EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -eq 0 ]; then
    tg "🎉 *exp018 COMPLETE!*
Tokenizer: stukenov/sozkz-morphbpe-256k-kk-v1
Corpus: stukenov/sozkz-corpus-segmented-kk-v1
⚠️ Don't forget to terminate the instance!"
else
    tg "❌ *exp018 FAILED!* (exit code $EXIT_CODE)
Check log: ssh -i ~/.ssh/exp018-tokenizer.pem ubuntu@100.24.19.102
tail /home/ubuntu/exp018/training.log"
fi

echo ""
echo "[5/5] Done! (exit code: $EXIT_CODE)"
echo "Finished: $(date)"
