#!/bin/bash
# Test TPU creation in all 6 TRC grant zones
# Создаёт минимальный TPU в каждой зоне, проверяет и удаляет
#
# Usage: bash ~/slm/trc/scripts/test_all_zones.sh

PROJECT="sozkz-trc"
RESULTS=()

test_zone() {
    local name="$1"
    local zone="$2"
    local accel="$3"
    local version="$4"
    local spot_flag="$5"
    local subnet_flag=""

    # us-central2 has custom subnet name
    if [[ "$zone" == us-central2-* ]]; then
        subnet_flag="--subnetwork=default-us-central2"
    fi

    echo ""
    echo "============================================"
    echo "Testing: $name ($accel) in $zone"
    echo "============================================"

    # Create
    echo "Creating..."
    if gcloud compute tpus tpu-vm create "$name" \
        --zone="$zone" \
        --accelerator-type="$accel" \
        --version="$version" \
        --project="$PROJECT" \
        $subnet_flag $spot_flag 2>&1; then
        echo "✅ CREATE OK: $name in $zone"
        RESULTS+=("✅ $name ($zone): OK")

        # Check status
        echo "Checking status..."
        gcloud compute tpus tpu-vm describe "$name" \
            --zone="$zone" --project="$PROJECT" \
            --format="table(name,state,acceleratorType)" 2>&1

        # Delete immediately
        echo "Deleting test TPU..."
        gcloud compute tpus tpu-vm delete "$name" \
            --zone="$zone" --project="$PROJECT" --quiet 2>&1
        echo "🗑️  Deleted: $name"
    else
        echo "❌ FAILED: $name in $zone"
        RESULTS+=("❌ $name ($zone): FAILED")
    fi
}

echo "=== SozKZ TRC Grant — TPU Zone Test ==="
echo "Project: $PROJECT"
echo "Testing smallest TPU in each granted zone..."
echo ""

# Test 1: v4 on-demand in us-central2-b
test_zone "test-v4-od" "us-central2-b" "v4-8" "tpu-vm-tf-2.17.0-pjrt" ""

# Test 2: v4 spot in us-central2-b
test_zone "test-v4-spot" "us-central2-b" "v4-8" "tpu-vm-tf-2.17.0-pjrt" "--spot"

# Test 3: v5e spot in us-central1-a
test_zone "test-v5e-us" "us-central1-a" "v5litepod-4" "v2-alpha-tpuv5-lite" "--spot"

# Test 4: v5e spot in europe-west4-b
test_zone "test-v5e-eu" "europe-west4-b" "v5litepod-4" "v2-alpha-tpuv5-lite" "--spot"

# Test 5: v6e spot in us-east1-d
test_zone "test-v6e-us" "us-east1-d" "v6e-4" "v2-alpha-tpuv6e" "--spot"

# Test 6: v6e spot in europe-west4-a
test_zone "test-v6e-eu" "europe-west4-a" "v6e-4" "v2-alpha-tpuv6e" "--spot"

echo ""
echo "============================================"
echo "         TEST RESULTS SUMMARY"
echo "============================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo "============================================"
echo ""

# Verify nothing is left running
echo "Verifying no TPUs left running..."
for zone in us-central2-b us-central1-a us-east1-d europe-west4-a europe-west4-b; do
    count=$(gcloud compute tpus tpu-vm list --zone="$zone" --project="$PROJECT" --format="value(name)" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "⚠️  $count TPU(s) still running in $zone!"
        gcloud compute tpus tpu-vm list --zone="$zone" --project="$PROJECT" 2>/dev/null
    fi
done
echo "✅ Cleanup verified"
