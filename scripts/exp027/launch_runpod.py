#!/usr/bin/env python3
"""
Launch a cheap RunPod pod for exp027 data preparation.

Needs: fast network (>500 Mbps), big disk (200GB), cheap GPU (any).
GPU is barely used — this is a CPU + network + disk job.

Usage:
    python scripts/exp027/launch_runpod.py launch
    python scripts/exp027/launch_runpod.py status
    python scripts/exp027/launch_runpod.py destroy POD_ID
"""

import argparse
import os
import sys
import time

try:
    import runpod
except ImportError:
    print("pip install runpod")
    sys.exit(1)

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
if not RUNPOD_API_KEY:
    # Try from parameter-golf config
    import json
    for path in [
        os.path.expanduser("~/.runpod/config.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
                RUNPOD_API_KEY = cfg.get("api_key", "")
                break

runpod.api_key = RUNPOD_API_KEY

DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

# Cheap GPUs ordered by preference — we need network, not compute
CHEAP_GPUS = [
    ("NVIDIA GeForce RTX 3090", 1, 200),
    ("NVIDIA GeForce RTX 4090", 1, 200),
    ("NVIDIA RTX A6000", 1, 200),
    ("NVIDIA A40", 1, 200),
]


def cmd_launch(args):
    pod = None
    for gpu_id, gpu_count, disk_gb in CHEAP_GPUS:
        try:
            print(f"Trying {gpu_count}x {gpu_id}...")
            pod = runpod.create_pod(
                name="exp027-data-prep",
                image_name=DOCKER_IMAGE,
                gpu_type_id=gpu_id,
                gpu_count=gpu_count,
                volume_in_gb=0,
                container_disk_in_gb=disk_gb,
                ports="22/tcp",
                support_public_ip=True,
            )
            print(f"SUCCESS: {gpu_count}x {gpu_id} -> pod {pod['id']}")
            break
        except Exception as e:
            print(f"  unavailable: {str(e)[:80]}")

    if not pod:
        print("No GPUs available. Try again later.")
        sys.exit(1)

    pod_id = pod["id"]
    print(f"\nWaiting for pod {pod_id}...")

    for i in range(90):
        time.sleep(10)
        p = runpod.get_pod(pod_id)
        rt = p.get("runtime") or {}
        uptime = rt.get("uptimeInSeconds", 0)
        ports = rt.get("ports") or []
        if uptime > 0 and ports:
            ssh_info = None
            for port in ports:
                if port.get("privatePort") == 22:
                    ssh_info = f"ssh root@{port['ip']} -p {port['publicPort']}"
            print(f"\n{'='*60}")
            print(f"POD READY: {pod_id}")
            if ssh_info:
                print(f"SSH: {ssh_info}")
            print(f"{'='*60}")
            print(f"\nDeploy with:")
            print(f"  bash scripts/exp027/deploy_to_pod.sh {port['ip']} {port['publicPort']}")
            return
        if i % 6 == 0:
            print(f"  [{i*10}s] booting...")

    print(f"Pod {pod_id} still booting. Check RunPod console.")


def cmd_status(args):
    pods = runpod.get_pods()
    if not pods:
        print("No running pods.")
        return
    print(f"{'ID':20s} {'Name':25s} {'Status':10s} {'GPU':25s}")
    print("-" * 85)
    for p in pods:
        rt = p.get("runtime") or {}
        gpus = rt.get("gpus") or []
        gpu_name = gpus[0].get("id", "?") if gpus else "booting"
        ports = rt.get("ports") or []
        ssh_str = ""
        for port in ports:
            if port.get("privatePort") == 22:
                ssh_str = f"  ssh root@{port['ip']} -p {port['publicPort']}"
        print(f"{p['id']:20s} {p['name']:25s} {p.get('desiredStatus','?'):10s} {gpu_name:25s}{ssh_str}")


def cmd_destroy(args):
    pod_id = args.pod_id
    confirm = input(f"Destroy pod {pod_id}? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    runpod.terminate_pod(pod_id)
    print(f"Destroyed {pod_id}.")


def main():
    parser = argparse.ArgumentParser(description="RunPod launcher for exp027")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("launch", help="Launch a cheap pod with big disk")
    sub.add_parser("status", help="Show running pods")

    p_destroy = sub.add_parser("destroy", help="Destroy a pod")
    p_destroy.add_argument("pod_id", type=str)

    args = parser.parse_args()
    if args.cmd == "launch":
        cmd_launch(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "destroy":
        cmd_destroy(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
