#!/usr/bin/env python3
"""
Delete TPU VMs safely with confirmation.

Usage:
  python3 delete_tpu.py --name test-v4-ondemand --zone us-central2-b
  python3 delete_tpu.py --all-tests    # Delete all test TPUs
  python3 delete_tpu.py --all          # Delete ALL TPUs (with confirmation)
"""

import argparse
import subprocess
import json
import sys
from config import PROJECT_ID, ALL_ZONES, TEST_ALLOCATIONS

def delete_tpu(name, zone, force=False):
    """Delete a single TPU VM."""
    if not force:
        confirm = input(f"Удалить TPU '{name}' в {zone}? (y/N): ")
        if confirm.lower() != 'y':
            print(f"  Пропущено: {name}")
            return 0

    print(f"Удаляю TPU: {name} в {zone}...")
    result = subprocess.run(
        ["gcloud", "compute", "tpus", "tpu-vm", "delete", name,
         "--zone", zone, "--project", PROJECT_ID, "--quiet"],
        timeout=120
    )
    if result.returncode == 0:
        print(f"  ✅ {name} удалён")
    else:
        print(f"  ❌ Ошибка удаления {name}")
    return result.returncode

def list_all_tpus():
    """Get all TPUs across all zones."""
    all_tpus = []
    for zone in ALL_ZONES:
        try:
            result = subprocess.run(
                ["gcloud", "compute", "tpus", "tpu-vm", "list",
                 "--zone", zone, "--project", PROJECT_ID, "--format=json"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                tpus = json.loads(result.stdout)
                for tpu in tpus:
                    name = tpu.get("name", "").split("/")[-1]
                    all_tpus.append({"name": name, "zone": zone, "tpu": tpu})
        except:
            pass
    return all_tpus

def main():
    parser = argparse.ArgumentParser(description="Delete TPU VM (TRC Grant)")
    parser.add_argument("--name", help="TPU name to delete")
    parser.add_argument("--zone", help="Zone of the TPU")
    parser.add_argument("--all-tests", action="store_true", help="Delete all test TPUs")
    parser.add_argument("--all", action="store_true", help="Delete ALL TPUs")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    if args.all_tests:
        print("Удаляю все тестовые TPU...")
        for t in TEST_ALLOCATIONS:
            delete_tpu(t["name"], t["zone"], force=args.force)
        return

    if args.all:
        print("Поиск всех TPU...")
        tpus = list_all_tpus()
        if not tpus:
            print("TPU не найдены.")
            return
        print(f"Найдено {len(tpus)} TPU:")
        for t in tpus:
            print(f"  - {t['name']} в {t['zone']}")
        if not args.force:
            confirm = input(f"\nУдалить все {len(tpus)} TPU? (y/N): ")
            if confirm.lower() != 'y':
                print("Отменено.")
                return
        for t in tpus:
            delete_tpu(t["name"], t["zone"], force=True)
        return

    if not args.name or not args.zone:
        parser.error("--name and --zone required (or use --all-tests / --all)")

    delete_tpu(args.name, args.zone, force=args.force)

if __name__ == "__main__":
    main()
