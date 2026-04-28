#!/bin/bash
# Monitor OmniAudio CTC training on TPUs
# Shows last few lines from each training log

echo "=== OmniAudio TPU Training Monitor — $(date) ==="
echo ""

check_training() {
    local name=$1 node=$2 zone=$3 screen=$4 log=$5
    echo "--- $name ($node in $zone) ---"
    # Check if TPU is alive
    state=$(gcloud compute tpus queued-resources list --zone="$zone" --project=sozkz-trc \
        --format="value(state)" --filter="name:*${node%%-node*}*" 2>/dev/null | head -1)

    if echo "$state" | grep -q "ACTIVE"; then
        # Get last lines from log
        gcloud compute tpus tpu-vm ssh "$node" --zone="$zone" --project=sozkz-trc \
            --command="tail -5 ~/$log 2>/dev/null || echo 'No log yet'; screen -ls | grep $screen || echo 'Screen not found'" 2>&1 | grep -v "^Using\|^SSH:\|^Propagat\|^Warning:\|^$"
    else
        echo "  TPU state: $state"
    fi
    echo ""
}

# 50m on v4-8
check_training "50m CTC" "sozkz-v4-8-spot-node" "us-central2-b" "ctc_50m" "ctc_50m.log"

# 150m on v6e-4
check_training "150m CTC" "sozkz-v6e4-euw4a-node" "europe-west4-a" "ctc_150m" "ctc_150m.log"

# 1b on v6e-8
check_training "1b CTC" "sozkz-v6e8-euw4a-node" "europe-west4-a" "ctc_1b" "ctc_1b.log"

# 600m — check if new TPU is ready
echo "--- 600m CTC (waiting for TPU) ---"
for zone in us-east1-d; do
    gcloud compute tpus queued-resources list --zone="$zone" --project=sozkz-trc \
        --format="table(name,state)" 2>/dev/null | grep -v "^$"
done

echo ""
echo "=== Done ==="
