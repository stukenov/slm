#!/bin/bash
# Parallel crawler + hourly HF upload
cd /root/zerokz_crawler

# Run parallel crawler (blocks until all sites done, handles HF upload internally)
# Upload runs in background every hour
upload_loop() {
    while true; do
        sleep 600
        echo "[$(date)] Upload to HF..."
        .venv/bin/python upload_to_hf.py 2>&1 | tee -a upload.log
    done
}
upload_loop &
UPLOAD_PID=$!

echo "[$(date)] Starting parallel crawler (${CRAWL_WORKERS:-6} workers)..."
CRAWL_WORKERS=${CRAWL_WORKERS:-6} .venv/bin/python crawl_parallel.py 2>&1 | tee -a crawl_all.log

# Final upload
echo "[$(date)] Final HF upload..."
.venv/bin/python upload_to_hf.py 2>&1 | tee -a upload.log

kill $UPLOAD_PID 2>/dev/null
echo "FINISHED at $(date)"
