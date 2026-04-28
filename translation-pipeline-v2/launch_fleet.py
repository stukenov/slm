#!/usr/bin/env python3
"""
Launch a fleet of RunPod instances for FineWeb-Edu 100BT translation.

Strategy: try to launch pods from largest to smallest GPU count.
Whatever comes alive gets assigned chunk ranges proportional to GPU power.

Usage:
    python launch_fleet.py                    # Launch default fleet
    python launch_fleet.py --dry-run          # Show what would be launched
    python launch_fleet.py --max-pods 3       # Limit to 3 pods
    python launch_fleet.py --total-chunks 100 # Override chunk count
"""

import argparse
import json
import os
import sys
import time

try:
    import runpod
except ImportError:
    print("pip install runpod")
    sys.exit(1)

RUNPOD_API_KEY = json.load(open(os.path.expanduser("~/.runpod/config.json")))["api_key"]
runpod.api_key = RUNPOD_API_KEY

DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
FLEET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet_100bt.json")

# Pods to try launching, in order of preference (most powerful first)
# (label, gpu_type_id, gpu_count, relative_speed_per_gpu)
# Speed is in sents/sec per GPU from BENCHMARK.md
POD_CONFIGS = [
    ("8x4090",  "NVIDIA GeForce RTX 4090", 8, 3040),
    ("4x4090",  "NVIDIA GeForce RTX 4090", 4, 3040),
    ("2x4090",  "NVIDIA GeForce RTX 4090", 2, 3040),
    ("4x3090",  "NVIDIA GeForce RTX 3090", 4, 2190),
    ("2x3090",  "NVIDIA GeForce RTX 3090", 2, 2190),
    ("2xA5000", "NVIDIA RTX A5000",        2, 1800),
    ("1x4090",  "NVIDIA GeForce RTX 4090", 1, 3040),
    ("2xA4000", "NVIDIA RTX A4000",        2, 1301),
    ("1x3090",  "NVIDIA GeForce RTX 3090", 1, 2190),
    ("1xA4000", "NVIDIA RTX A4000",        1, 1301),
]


def launch_pod(label: str, gpu_type_id: str, gpu_count: int) -> dict | None:
    """Try to launch a pod. Returns pod info or None."""
    try:
        pod = runpod.create_pod(
            name=f"translate-100bt-{label}",
            image_name=DOCKER_IMAGE,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            volume_in_gb=0,
            container_disk_in_gb=80,
            ports="22/tcp",
            support_public_ip=True,
            cloud_type="COMMUNITY",
        )
        print(f"  [OK] {label}: pod {pod['id']} created")
        return pod
    except Exception as e:
        print(f"  [--] {label}: unavailable ({e})")
        return None


def wait_for_pod(pod_id: str, timeout: int = 600) -> dict | None:
    """Wait for pod to boot and return SSH info."""
    for i in range(timeout // 10):
        try:
            p = runpod.get_pod(pod_id)
            rt = p.get("runtime") or {}
            uptime = rt.get("uptimeInSeconds", 0)
            ports = rt.get("ports") or []
            gpus = rt.get("gpus") or []

            if uptime > 0 and ports:
                for port in ports:
                    if port.get("privatePort") == 22:
                        return {
                            "ssh_ip": port["ip"],
                            "ssh_port": port["publicPort"],
                            "gpu_count": len(gpus) if gpus else p.get("gpuCount", 0),
                        }
        except Exception:
            pass

        if i % 6 == 0:
            print(f"    [{i*10}s] waiting for {pod_id}...")
        time.sleep(10)

    print(f"    [TIMEOUT] {pod_id} didn't boot in {timeout}s")
    return None


def main():
    parser = argparse.ArgumentParser(description="Launch RunPod fleet for 100BT translation")
    parser.add_argument("--max-pods", type=int, default=5, help="Max pods to launch")
    parser.add_argument("--total-chunks", type=int, default=100, help="Total chunks in dataset")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without launching")
    parser.add_argument("--boot-timeout", type=int, default=600, help="Seconds to wait per pod")
    args = parser.parse_args()

    print(f"Fleet launcher: up to {args.max_pods} pods, {args.total_chunks} chunks")
    print(f"Docker image: {DOCKER_IMAGE}")
    print()

    if args.dry_run:
        print("DRY RUN — would try these configs:")
        for label, gpu_type, gpu_count, speed in POD_CONFIGS[:args.max_pods]:
            total_speed = speed * gpu_count
            hours = (args.total_chunks * 1_000_000 * 34.5) / total_speed / 3600  # ~34.5 sents per doc
            print(f"  {label}: {gpu_type} ×{gpu_count} → {total_speed} sents/sec → ~{hours:.0f}h solo")
        return

    # Phase 1: Launch pods
    print("=" * 60)
    print("PHASE 1: Launching pods")
    print("=" * 60)

    launched = []  # (label, pod_id, gpu_count, speed_per_gpu)
    for label, gpu_type, gpu_count, speed in POD_CONFIGS:
        if len(launched) >= args.max_pods:
            break

        print(f"\nTrying {label} ({gpu_type} ×{gpu_count})...")
        pod = launch_pod(label, gpu_type, gpu_count)
        if pod:
            launched.append((label, pod["id"], gpu_count, speed))

    if not launched:
        print("\nFATAL: No pods launched! Check RunPod availability.")
        sys.exit(1)

    print(f"\nLaunched {len(launched)} pods. Waiting for boot...")

    # Phase 2: Wait for all pods
    print()
    print("=" * 60)
    print("PHASE 2: Waiting for pods to boot")
    print("=" * 60)

    fleet = []  # Final fleet info
    for label, pod_id, gpu_count, speed_per_gpu in launched:
        print(f"\nWaiting for {label} (pod {pod_id})...")
        info = wait_for_pod(pod_id, args.boot_timeout)
        if info:
            actual_gpus = info["gpu_count"] or gpu_count
            total_speed = speed_per_gpu * actual_gpus
            ssh_cmd = f"ssh root@{info['ssh_ip']} -p {info['ssh_port']} -o StrictHostKeyChecking=no"
            fleet.append({
                "label": label,
                "pod_id": pod_id,
                "ssh_ip": info["ssh_ip"],
                "ssh_port": info["ssh_port"],
                "ssh_cmd": ssh_cmd,
                "gpu_count": actual_gpus,
                "speed_per_gpu": speed_per_gpu,
                "total_speed": total_speed,
            })
            print(f"  [READY] {label}: {actual_gpus} GPUs, {total_speed} sents/sec")
            print(f"          SSH: {ssh_cmd}")
        else:
            print(f"  [FAILED] {label} (pod {pod_id}) — will terminate")
            try:
                runpod.terminate_pod(pod_id)
            except Exception:
                pass

    if not fleet:
        print("\nFATAL: No pods booted!")
        sys.exit(1)

    # Phase 3: Assign chunk ranges proportional to speed
    print()
    print("=" * 60)
    print("PHASE 3: Assigning chunk ranges")
    print("=" * 60)

    total_speed = sum(p["total_speed"] for p in fleet)
    chunk_cursor = 0

    for i, pod in enumerate(fleet):
        fraction = pod["total_speed"] / total_speed
        num_chunks = max(1, round(fraction * args.total_chunks))

        # Last pod gets remainder
        if i == len(fleet) - 1:
            num_chunks = args.total_chunks - chunk_cursor

        start = chunk_cursor
        end = min(chunk_cursor + num_chunks, args.total_chunks)
        chunk_cursor = end

        pod["start_chunk"] = start
        pod["end_chunk"] = end
        pod["num_chunks"] = end - start

        est_hours = (pod["num_chunks"] * 1_000_000 * 34.5) / pod["total_speed"] / 3600
        est_cost = est_hours * pod["gpu_count"] * 0.37  # ~$0.37/gpu/hr for 4090

        print(f"\n  {pod['label']} (pod {pod['pod_id']}):")
        print(f"    Chunks: {start}–{end-1} ({pod['num_chunks']} chunks)")
        print(f"    Speed:  {pod['total_speed']} sents/sec ({pod['gpu_count']} GPUs)")
        print(f"    ETA:    ~{est_hours:.1f}h")
        print(f"    SSH:    {pod['ssh_cmd']}")

    # Save fleet info
    fleet_data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": args.total_chunks,
        "total_speed": total_speed,
        "pods": fleet,
    }

    with open(FLEET_FILE, "w") as f:
        json.dump(fleet_data, f, indent=2)
    print(f"\nFleet info saved to: {FLEET_FILE}")

    # Print summary
    max_hours = max(
        (p["num_chunks"] * 1_000_000 * 34.5) / p["total_speed"] / 3600
        for p in fleet
    )
    print()
    print("=" * 60)
    print(f"FLEET READY: {len(fleet)} pods, {sum(p['gpu_count'] for p in fleet)} total GPUs")
    print(f"Estimated wall time: ~{max_hours:.1f}h (limited by slowest pod)")
    print(f"Next step: python dispatch_fleet.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
