#!/usr/bin/env python3
"""
RunPod pod manager for parameter-golf experiments.

Usage:
    python runpod_launch.py launch              # launch cheap GPU (tries 4090 > A6000 > A40 > L40S)
    python runpod_launch.py launch --gpu H100   # launch specific GPU
    python runpod_launch.py status              # show running pods
    python runpod_launch.py destroy POD_ID      # destroy specific pod
    python runpod_launch.py destroy-all         # destroy ALL pods (asks confirmation)
"""
import argparse
import os
import sys
import time
import json

try:
    import runpod
except ImportError:
    print("pip install runpod")
    sys.exit(1)

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
runpod.api_key = RUNPOD_API_KEY

DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

# Ordered by preference for cheap iteration
CHEAP_GPUS = [
    ("NVIDIA GeForce RTX 4090", 1, 80),
    ("NVIDIA RTX A6000", 1, 80),
    ("NVIDIA A40", 1, 80),
    ("NVIDIA L40S", 1, 80),
    ("NVIDIA GeForce RTX 3090", 1, 60),
]

SPECIFIC_GPUS = {
    "4090":  ("NVIDIA GeForce RTX 4090", 1, 80),
    "A6000": ("NVIDIA RTX A6000", 1, 80),
    "A40":   ("NVIDIA A40", 1, 80),
    "L40S":  ("NVIDIA L40S", 1, 80),
    "H100":  ("NVIDIA H100 80GB HBM3", 1, 100),
    "8xH100": ("NVIDIA H100 80GB HBM3", 8, 200),
    "A100":  ("NVIDIA A100 80GB PCIe", 1, 100),
}


def cmd_launch(args):
    if args.gpu:
        if args.gpu not in SPECIFIC_GPUS:
            print(f"Unknown GPU: {args.gpu}. Options: {list(SPECIFIC_GPUS.keys())}")
            sys.exit(1)
        candidates = [SPECIFIC_GPUS[args.gpu]]
    else:
        candidates = CHEAP_GPUS

    pod = None
    for gpu_id, gpu_count, disk_gb in candidates:
        try:
            print(f"Trying {gpu_count}x {gpu_id}...")
            pod = runpod.create_pod(
                name=f"pgolf-{args.name}",
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
            print(f"  unavailable: {str(e)[:60]}")

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
            print(f"\n{'='*50}")
            print(f"POD READY: {pod_id}")
            if ssh_info:
                print(f"SSH: {ssh_info}")
            print(f"{'='*50}")
            print(f"\nNext steps on the pod:")
            print(f"  bash /workspace/setup_pod.sh")
            print(f"  cp /workspace/train_gpt.py /workspace/parameter-golf/train_gpt.py")
            print(f"  bash /workspace/run_train.sh quick")
            return
        if i % 6 == 0:
            print(f"  [{i*10}s] booting...")

    print(f"Pod {pod_id} still booting. Check RunPod console.")


def cmd_status(args):
    pods = runpod.get_pods()
    if not pods:
        print("No running pods.")
        return
    print(f"{'ID':20s} {'Name':25s} {'Status':10s} {'Uptime':>8s} {'GPU':20s}")
    print("-" * 90)
    for p in pods:
        rt = p.get("runtime") or {}
        uptime = rt.get("uptimeInSeconds", 0)
        gpus = rt.get("gpus") or []
        gpu_name = gpus[0].get("id", "?") if gpus else "booting"
        uptime_str = f"{uptime//60}m" if uptime else "0"
        ports = rt.get("ports") or []
        ssh_str = ""
        for port in ports:
            if port.get("privatePort") == 22:
                ssh_str = f"  ssh root@{port['ip']} -p {port['publicPort']}"
        print(f"{p['id']:20s} {p['name']:25s} {p.get('desiredStatus','?'):10s} {uptime_str:>8s} {gpu_name:20s}{ssh_str}")


def cmd_destroy(args):
    pod_id = args.pod_id
    print(f"Destroying pod {pod_id}...")
    runpod.terminate_pod(pod_id)
    print("Done.")


def cmd_destroy_all(args):
    pods = runpod.get_pods()
    if not pods:
        print("No pods to destroy.")
        return
    print(f"Will destroy {len(pods)} pod(s):")
    for p in pods:
        print(f"  {p['id']}  {p['name']}")
    confirm = input("Confirm? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    for p in pods:
        runpod.terminate_pod(p["id"])
        print(f"  Destroyed {p['id']}")
    print("All pods destroyed.")


def main():
    parser = argparse.ArgumentParser(description="RunPod manager for parameter-golf")
    sub = parser.add_subparsers(dest="cmd")

    p_launch = sub.add_parser("launch", help="Launch a pod")
    p_launch.add_argument("--gpu", type=str, help=f"GPU type: {list(SPECIFIC_GPUS.keys())}")
    p_launch.add_argument("--name", default="exp", help="Pod name suffix")

    sub.add_parser("status", help="Show running pods")

    p_destroy = sub.add_parser("destroy", help="Destroy a pod")
    p_destroy.add_argument("pod_id", type=str)

    sub.add_parser("destroy-all", help="Destroy all pods")

    args = parser.parse_args()
    if args.cmd == "launch":
        cmd_launch(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "destroy":
        cmd_destroy(args)
    elif args.cmd == "destroy-all":
        cmd_destroy_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
