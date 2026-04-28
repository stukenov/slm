#!/bin/bash
set -euo pipefail

MODE="${1:-full}"   # "sample" or "full"
SAMPLE_N="${2:-5}"

cd /workspace/slm

apt-get update -qq
apt-get install -y ffmpeg git rsync curl

# Node.js 20 for yt-dlp n-challenge solver
if ! node --version 2>/dev/null | grep -q 'v2[0-9]'; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>/dev/null
  apt-get install -y nodejs -qq
  ln -sf /usr/bin/node /usr/local/bin/node 2>/dev/null || true
fi

python3 -m pip install blinker --ignore-installed -q
python3 -m pip install -e '.[audio_collect,review]' -q

if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

mkdir -p /workspace/logs

if [[ "$MODE" == "sample" ]]; then
  echo "=== Sample mode: processing ${SAMPLE_N} videos for manual review ==="
  # Run in foreground so you see progress
  python3 scripts/exp038/prepare_youtube_kk_audio.py \
    --config configs/experiments/exp038_youtube_recent_kk_audio.yaml \
    --step all \
    --sample "$SAMPLE_N" \
    2>&1 | tee /workspace/logs/exp038_sample.log

  echo ""
  echo "=== Sample done. Starting Review UI on port 8501 ==="
  nohup streamlit run scripts/exp038/review_ui.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > /workspace/logs/review_ui.log 2>&1 &

  echo "Review UI started."
  echo "Open in browser: https://\${RUNPOD_POD_ID}-8501.proxy.runpod.net"
  echo "Or SSH tunnel:   ssh -L 8501:localhost:8501 root@<host> -p <port>"

else
  echo "=== Full mode: inventory + process all videos (target 5000h) ==="
  screen -dmS exp038 bash -c "
    python3 scripts/exp038/prepare_youtube_kk_audio.py \
      --config configs/experiments/exp038_youtube_recent_kk_audio.yaml \
      --step all \
      2>&1 | tee /workspace/logs/exp038_full.log
  "
  echo "Started in screen session 'exp038'"
  echo "Monitor: screen -r exp038"
  echo "Log:     tail -f /workspace/logs/exp038_full.log"
fi
