#!/usr/bin/env python3
"""
List all TPU VMs and Queued Resources across all TRC grant zones.
Usage: python3 list_tpus.py
"""

import subprocess
import json
import sys
from config import PROJECT_ID, ALL_ZONES

def list_tpus_in_zone(zone):
    """List TPU VMs in a specific zone."""
    try:
        result = subprocess.run(
            ["gcloud", "compute", "tpus", "tpu-vm", "list",
             "--zone", zone, "--project", PROJECT_ID, "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return []
    except Exception as e:
        print(f"  Error listing TPUs in {zone}: {e}")
        return []

def list_queued_resources_in_zone(zone):
    """List Queued Resources in a specific zone."""
    try:
        result = subprocess.run(
            ["gcloud", "compute", "tpus", "queued-resources", "list",
             "--zone", zone, "--project", PROJECT_ID, "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return []
    except Exception as e:
        print(f"  Error listing QRs in {zone}: {e}")
        return []

def main():
    print(f"{'='*60}")
    print(f"TPU Resources in project: {PROJECT_ID}")
    print(f"{'='*60}")

    total_tpus = 0
    total_qrs = 0

    for zone in sorted(ALL_ZONES):
        print(f"\n--- Zone: {zone} ---")

        # TPU VMs
        tpus = list_tpus_in_zone(zone)
        if tpus:
            for tpu in tpus:
                name = tpu.get("name", "unknown").split("/")[-1]
                state = tpu.get("state", "UNKNOWN")
                accel = tpu.get("acceleratorType", "unknown").split("/")[-1]
                print(f"  TPU VM: {name} | {accel} | {state}")
                total_tpus += 1
        else:
            print("  No TPU VMs")

        # Queued Resources
        qrs = list_queued_resources_in_zone(zone)
        if qrs:
            for qr in qrs:
                name = qr.get("name", "unknown").split("/")[-1]
                state = qr.get("state", {}).get("state", "UNKNOWN")
                print(f"  Queued Resource: {name} | {state}")
                total_qrs += 1
        else:
            print("  No Queued Resources")

    print(f"\n{'='*60}")
    print(f"Total: {total_tpus} TPU VMs, {total_qrs} Queued Resources")
    print(f"{'='*60}")

    if total_tpus > 0:
        print("\n⚠️  Напоминание: неиспользуемые TPU занимают квоту!")
        print("    Удали ненужные: python3 delete_tpu.py --name NAME --zone ZONE")

if __name__ == "__main__":
    main()
