#!/usr/bin/env python3
"""
Create a TPU VM in one of the TRC grant zones.
Safety: only allows creation in allocated zones to prevent accidental charges.

Usage:
  python3 create_tpu.py --type v5e --zone us-central1-a --spot
  python3 create_tpu.py --type v4 --zone us-central2-b
  python3 create_tpu.py --type v6e --zone us-east1-d --spot --name my-worker
  python3 create_tpu.py --test  # Create smallest test TPU in each zone
"""

import argparse
import subprocess
import sys
from config import PROJECT_ID, ALLOCATIONS, TEST_ALLOCATIONS, RUNTIME_VERSIONS, get_subnet_flag

# Safety: only these zone+type combos are covered by the TRC grant
ALLOWED_COMBOS = set()
for a in ALLOCATIONS:
    ALLOWED_COMBOS.add((a["zone"], a["tpu_type"], a["quota_type"]))

def validate_request(zone, tpu_type, spot):
    """Check that the requested TPU is within our TRC grant allocation."""
    quota_type = "spot" if spot else "on-demand"
    if (zone, tpu_type, quota_type) not in ALLOWED_COMBOS:
        print(f"❌ ОШИБКА: Нет квоты для {tpu_type} ({quota_type}) в {zone}!")
        print(f"   Создание TPU за пределами квоты приведёт к РЕАЛЬНЫМ СПИСАНИЯМ.")
        print(f"\nДоступные комбинации:")
        for a in ALLOCATIONS:
            print(f"  - {a['tpu_type']} ({a['quota_type']}) в {a['zone']} — {a['chips']} чипов")
        return False
    return True

def create_tpu(name, accelerator_type, zone, runtime_version, spot=True):
    """Create a TPU VM."""
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "create", name,
        "--zone", zone,
        "--accelerator-type", accelerator_type,
        "--version", runtime_version,
        "--project", PROJECT_ID,
    ]
    if spot:
        cmd.append("--spot")

    subnet = get_subnet_flag(zone)
    if subnet:
        cmd.extend(["--subnetwork", subnet])

    spot_str = "spot" if spot else "on-demand"
    print(f"Создаю TPU: {name}")
    print(f"  Тип: {accelerator_type} ({spot_str})")
    print(f"  Зона: {zone}")
    print(f"  Runtime: {runtime_version}")
    print(f"  Команда: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, timeout=300)
    if result.returncode == 0:
        print(f"\n✅ TPU {name} создан успешно!")
        print(f"   Подключение: gcloud compute tpus tpu-vm ssh {name} --zone={zone}")
        print(f"   Удаление:    gcloud compute tpus tpu-vm delete {name} --zone={zone}")
    else:
        print(f"\n❌ Ошибка создания TPU {name}")
    return result.returncode

def run_tests():
    """Create smallest test TPUs in each zone."""
    print("🧪 Тестовое создание TPU во всех зонах гранта")
    print("=" * 50)
    results = []
    for t in TEST_ALLOCATIONS:
        print(f"\n--- {t['name']} ({t['accelerator_type']}) в {t['zone']} ---")
        rc = create_tpu(
            t["name"], t["accelerator_type"], t["zone"],
            t["runtime_version"], t.get("spot", True)
        )
        results.append((t["name"], t["zone"], rc))

    print(f"\n{'='*50}")
    print("Результаты тестов:")
    for name, zone, rc in results:
        status = "✅ OK" if rc == 0 else "❌ FAIL"
        print(f"  {name} ({zone}): {status}")

    # Cleanup
    print(f"\n⚠️  Не забудь удалить тестовые TPU:")
    print(f"   python3 delete_tpu.py --all-tests")

def main():
    parser = argparse.ArgumentParser(description="Create TPU VM (TRC Grant)")
    parser.add_argument("--type", choices=["v4", "v5e", "v6e"], help="TPU type")
    parser.add_argument("--zone", help="GCP zone")
    parser.add_argument("--spot", action="store_true", default=True, help="Use spot (default)")
    parser.add_argument("--on-demand", action="store_true", help="Use on-demand")
    parser.add_argument("--name", help="TPU name (auto-generated if not specified)")
    parser.add_argument("--chips", type=int, help="Number of chips (default: full allocation)")
    parser.add_argument("--test", action="store_true", help="Create test TPUs in all zones")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if not args.type or not args.zone:
        parser.error("--type and --zone are required (or use --test)")

    spot = not args.on_demand

    if not validate_request(args.zone, args.type, spot):
        sys.exit(1)

    # Find matching allocation
    matching = [a for a in ALLOCATIONS
                if a["tpu_type"] == args.type and a["zone"] == args.zone
                and a["quota_type"] == ("spot" if spot else "on-demand")]

    if not matching:
        print("❌ Не найдена подходящая квота")
        sys.exit(1)

    alloc = matching[0]
    name = args.name or f"sozkz-{args.type}-worker"
    accel = alloc["accelerator_type"]

    # Allow smaller chip count
    if args.chips and args.chips < alloc["chips"]:
        accel_base = args.type.replace("v5e", "v5litepod")
        if args.type == "v5e":
            accel = f"v5litepod-{args.chips}"
        elif args.type == "v6e":
            accel = f"v6e-{args.chips}"
        else:
            accel = f"v4-{args.chips}"

    runtime = RUNTIME_VERSIONS[args.type]
    rc = create_tpu(name, accel, args.zone, runtime, spot)
    sys.exit(rc)

if __name__ == "__main__":
    main()
