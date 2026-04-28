#!/usr/bin/env python3
"""Launch 2x4090 RunPod pod for GEC dataset generation with GPT-OSS-120B."""
import os, sys, time, json

try:
    import runpod
except ImportError:
    print("pip install runpod"); sys.exit(1)

runpod.api_key = os.environ.get(
    "RUNPOD_API_KEY",
    "REDACTED_RUNPOD_API_KEY",
)

DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def launch():
    configs = [
        ("SPOT 2x4090", "NVIDIA GeForce RTX 4090", 2, "COMMUNITY"),
        ("ON-DEMAND 2x4090", "NVIDIA GeForce RTX 4090", 2, "SECURE"),
    ]

    pod = None
    for label, gpu, count, cloud in configs:
        print(f"Trying {label}...")
        try:
            pod = runpod.create_pod(
                name="exp031-gec-gen",
                image_name=DOCKER_IMAGE,
                gpu_type_id=gpu,
                gpu_count=count,
                volume_in_gb=0,
                container_disk_in_gb=150,  # Need space for 33GB model
                ports="22/tcp,8000/http",
                support_public_ip=True,
                cloud_type=cloud,
            )
            print(f"SUCCESS: {label} -> pod {pod['id']}")
            break
        except Exception as e:
            print(f"  unavailable: {e}")

    if not pod:
        print("FAILED"); sys.exit(1)

    pod_id = pod["id"]
    print(f"Waiting for pod {pod_id}...")

    for i in range(120):
        time.sleep(10)
        p = runpod.get_pod(pod_id)
        rt = p.get("runtime") or {}
        ports = rt.get("ports") or []
        if rt.get("uptimeInSeconds", 0) > 0 and ports:
            for port in ports:
                if port.get("privatePort") == 22:
                    info = {
                        "pod_id": pod_id,
                        "ssh_ip": port["ip"],
                        "ssh_port": port["publicPort"],
                    }
                    with open("/tmp/exp031_pod.json", "w") as f:
                        json.dump(info, f, indent=2)
                    print(f"\nREADY: ssh root@{port['ip']} -p {port['publicPort']}")
                    return info
        if i % 6 == 0:
            print(f"  [{i*10}s] booting...")

    print("Timeout")
    return None


if __name__ == "__main__":
    launch()
