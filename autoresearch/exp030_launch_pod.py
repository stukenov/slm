#!/usr/bin/env python3
"""exp030: Launch 4xH100 RunPod pod for GEC 1B experiments."""
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

DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def launch():
    configs = [
        ("SPOT 4xH100", 4, "COMMUNITY"),
        ("ON-DEMAND 4xH100", 4, "SECURE"),
    ]

    pod = None
    for label, gpu_count, cloud_type in configs:
        print(f"Trying {label}...")
        try:
            pod = runpod.create_pod(
                name="exp030-gec-1b",
                image_name=DOCKER_IMAGE,
                gpu_type_id="NVIDIA H100 80GB HBM3",
                gpu_count=gpu_count,
                volume_in_gb=0,
                container_disk_in_gb=200,
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
                print(f"SSH: ssh root@{ssh_ip} -p {ssh_port} -o StrictHostKeyChecking=no")
                print(f"{'='*60}")
                info = {
                    "pod_id": pod_id,
                    "ssh_ip": ssh_ip,
                    "ssh_port": ssh_port,
                    "ssh_cmd": f"ssh root@{ssh_ip} -p {ssh_port} -o StrictHostKeyChecking=no",
                    "api_key": RUNPOD_API_KEY,
                }
                with open("/tmp/exp030_pod.json", "w") as f:
                    json.dump(info, f, indent=2)
                print(f"Connection info saved to /tmp/exp030_pod.json")
                return info
        if i % 6 == 0:
            print(f"  [{i*10}s] booting...")

    print(f"Pod {pod_id} still booting after 20min. Check RunPod console.")
    return None


if __name__ == "__main__":
    launch()
