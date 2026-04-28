#!/bin/bash
# Auto pipeline: wait for tokenizer training to finish, then tokenize dataset.
# Runs unattended in screen.
set -euo pipefail

export HF_TOKEN="REDACTED_HF_TOKEN"
cd /root/slm

echo "=== Auto Pipeline: waiting for tokenizer training ==="

# Wait for tokenizer screen to finish
while screen -ls | grep -q exp027_tokenizer; do
    echo "$(date): tokenizer still running..."
    sleep 60
done

echo "$(date): tokenizer training finished!"

# Check if tokenizer was actually uploaded
if grep -q "DONE.*ekitil-vocab" exp027/tokenizer.log 2>/dev/null; then
    echo "$(date): tokenizer upload confirmed, starting tokenization..."
    python3 exp027/tokenize_dataset.py 2>&1 | tee exp027/tokenize.log
    echo "$(date): tokenization complete!"
else
    echo "$(date): ERROR - tokenizer training may have failed. Check exp027/tokenizer.log"
    # Send telegram about failure
    python3 -c "
import urllib.request, urllib.parse
url = 'https://api.telegram.org/botREDACTED_TG_BOT_TOKEN/sendMessage?chat_id=47474471&text=' + urllib.parse.quote('❌ Tokenizer training failed! Check logs.')
try: urllib.request.urlopen(url, timeout=10)
except: pass
"
fi

echo "=== Pipeline finished ==="
