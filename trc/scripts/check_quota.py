#!/usr/bin/env python3
"""
Check TPU quota usage across all TRC grant zones.
Usage: python3 check_quota.py
"""

import subprocess
import json
from config import PROJECT_ID, ALLOCATIONS, ALL_ZONES

def check_tpu_quota(zone):
    """Check TPU quota in a zone."""
    try:
        result = subprocess.run(
            ["gcloud", "compute", "tpus", "tpu-vm", "list",
             "--zone", zone, "--project", PROJECT_ID, "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        tpus = []
        if result.returncode == 0 and result.stdout.strip():
            tpus = json.loads(result.stdout)
        return tpus
    except Exception as e:
        print(f"  Error: {e}")
        return []

def main():
    print("=" * 70)
    print(f"TPU Quota Usage — Project: {PROJECT_ID}")
    print("=" * 70)

    for alloc in ALLOCATIONS:
        zone = alloc["zone"]
        tpu_type = alloc["tpu_type"]
        chips = alloc["chips"]
        quota_type = alloc["quota_type"]

        tpus = check_tpu_quota(zone)

        # Count chips in use for this type
        used_chips = 0
        for tpu in tpus:
            accel = tpu.get("acceleratorType", "").split("/")[-1]
            if tpu_type in accel.lower():
                # Extract chip count from accelerator type (e.g., v4-32 -> 32)
                parts = accel.split("-")
                if len(parts) >= 2 and parts[-1].isdigit():
                    used_chips += int(parts[-1])

        status = "🟢" if used_chips == 0 else ("🟡" if used_chips < chips else "🔴")
        print(f"\n{status} {tpu_type} ({quota_type}) — {zone}")
        print(f"   Квота: {chips} чипов | Используется: {used_chips} | Свободно: {chips - used_chips}")

        if tpus:
            for tpu in tpus:
                name = tpu.get("name", "").split("/")[-1]
                state = tpu.get("state", "UNKNOWN")
                accel = tpu.get("acceleratorType", "").split("/")[-1]
                print(f"   └─ {name}: {accel} [{state}]")

    print(f"\n{'='*70}")
    print("Легенда: 🟢 свободно | 🟡 частично | 🔴 полностью занято")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
