#!/bin/bash
# Resilient OpenSLR-140 preprocessing — restarts on crash, skips bad samples.
# Each run processes until it hits a corrupt audio, saves what it has, exits.
# This script keeps restarting until no new data is produced (= dataset exhausted).

cd /root/slm
source .venv/bin/activate

OUTPUT_DIR="/root/slm/data/asr_mels"
MAX_RESTARTS=50
restart=0

while [ $restart -lt $MAX_RESTARTS ]; do
    restart=$((restart + 1))

    # Count existing shards to detect progress
    before=$(ls "$OUTPUT_DIR/openslr140/shard_"*_meta.json 2>/dev/null | wc -l)

    echo "=== Restart #$restart (existing shards: $before) ==="

    # Run preprocessing — it will process until crash, save shards, exit
    PYTHONUNBUFFERED=1 python scripts/preprocess_asr_mels.py \
        --output_dir "$OUTPUT_DIR" \
        --datasets openslr140 2>&1

    # Count shards after
    after=$(ls "$OUTPUT_DIR/openslr140/shard_"*_meta.json 2>/dev/null | wc -l)

    echo "=== Restart #$restart done. Shards: $before -> $after ==="

    # If no new shards were created, dataset is exhausted or permanently broken
    if [ "$after" -eq "$before" ]; then
        echo "No new data produced. Dataset likely exhausted."
        break
    fi

    # Remove the combine output (train/val/test) so next run doesn't fail
    rm -rf "$OUTPUT_DIR/train" "$OUTPUT_DIR/validation" "$OUTPUT_DIR/test" "$OUTPUT_DIR/metadata.json"

    sleep 5
done

echo "=== All restarts done. Total shards: $(ls "$OUTPUT_DIR/openslr140/shard_"*_meta.json 2>/dev/null | wc -l) ==="
echo "Run combine manually: python scripts/preprocess_asr_mels.py --output_dir $OUTPUT_DIR --datasets '' --push_to_hub"
