#!/usr/bin/env python3
"""exp032: Launch cheap RunPod pod for token counting.

Doesn't need a powerful GPU — just CPU + RAM + disk for dataset.
Uses cheapest available GPU since RunPod requires one.
"""
import json
import os
import sys
import time

try:
    import runpod
except ImportError:
    print("pip install runpod")
    sys.exit(1)

runpod.api_key = json.load(open(os.path.expanduser("~/.runpod/config.json")))["api_key"]

DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def launch():
    # 1×H100 — will reuse for both token counting and training
    configs = [
        ("SPOT 1xH100 SXM", "NVIDIA H100 80GB HBM3", 1, "COMMUNITY"),
        ("SPOT 1xH100 PCIe", "NVIDIA H100 PCIe", 1, "COMMUNITY"),
        ("ON-DEMAND 1xH100", "NVIDIA H100 80GB HBM3", 1, "SECURE"),
    ]

    pod = None
    for label, gpu_type, gpu_count, cloud_type in configs:
        print(f"Trying {label}...")
        try:
            pod = runpod.create_pod(
                name="exp032-kazakh-adapt",
                image_name=DOCKER_IMAGE,
                gpu_type_id=gpu_type,
                gpu_count=gpu_count,
                volume_in_gb=0,
                container_disk_in_gb=100,  # need space for dataset
                ports="22/tcp",
                support_public_ip=True,
                cloud_type=cloud_type,
            )
            print(f"SUCCESS: {label} -> pod {pod['id']}")
            break
        except Exception as e:
            print(f"  {label} unavailable: {e}")

    if pod is None:
        print("All GPU options exhausted!")
        sys.exit(1)

    pod_id = pod["id"]
    print(f"\nWaiting for pod {pod_id}...")

    for i in range(120):
        time.sleep(10)
        p = runpod.get_pod(pod_id)
        rt = p.get("runtime") or {}
        uptime = rt.get("uptimeInSeconds", 0)
        ports = rt.get("ports") or []
        if uptime > 0 and ports:
            ssh_ip = None
            ssh_port = None
            for port in ports:
                if port.get("privatePort") == 22:
                    ssh_ip = port["ip"]
                    ssh_port = port["publicPort"]
            if ssh_ip:
                print(f"\n{'='*60}")
                print(f"POD READY: {pod_id}")
                ssh_cmd = f"ssh root@{ssh_ip} -p {ssh_port} -o StrictHostKeyChecking=no"
                print(f"SSH: {ssh_cmd}")
                print(f"{'='*60}")

                info = {
                    "pod_id": pod_id,
                    "ssh_ip": ssh_ip,
                    "ssh_port": ssh_port,
                    "ssh_cmd": ssh_cmd,
                }
                with open("/tmp/exp032_pod.json", "w") as f:
                    json.dump(info, f, indent=2)
                print(f"Saved to /tmp/exp032_pod.json")
                return info
        if i % 6 == 0:
            print(f"  [{i*10}s] booting...")

    print(f"Pod {pod_id} still booting after 20min. Check RunPod console.")
    return None


if __name__ == "__main__":
    launch()
