#!/usr/bin/env python3
"""Launch SPOT 8xH100 RunPod pod for exp028v2 1.08B training."""
import os, sys, time, json

try:
    import runpod
except ImportError:
    print("pip install runpod")
    sys.exit(1)

RUNPOD_API_KEY = os.environ.get(
    "RUNPOD_API_KEY",
    "REDACTED_RUNPOD_API_KEY",
)
runpod.api_key = RUNPOD_API_KEY

# RunPod native image — boots fast (pytorch/pytorch:2.5.1 got stuck last time)
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def launch():
    # Try spot 8xH100 first, then spot 4xH100, then on-demand as last resort
    configs = [
        ("SPOT 8xH100", 8, True),
        ("SPOT 4xH100", 4, True),
    ]

    pod = None
    for label, gpu_count, is_spot in configs:
        print(f"Trying {label}...")
        try:
            kwargs = dict(
                name="exp028v2-llama-1b",
                image_name=DOCKER_IMAGE,
                gpu_type_id="NVIDIA H100 80GB HBM3",
                gpu_count=gpu_count,
                volume_in_gb=0,
                container_disk_in_gb=200,
                ports="22/tcp",
                support_public_ip=True,
            )
            if is_spot:
                kwargs["cloud_type"] = "COMMUNITY"  # spot/community = cheaper
            pod = runpod.create_pod(**kwargs)
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
                # Save connection info
                info = {
                    "pod_id": pod_id,
                    "ssh_ip": ssh_ip,
                    "ssh_port": ssh_port,
                    "ssh_cmd": f"ssh root@{ssh_ip} -p {ssh_port} -o StrictHostKeyChecking=no",
                    "api_key": RUNPOD_API_KEY,
                }
                with open("/tmp/exp028_pod.json", "w") as f:
                    json.dump(info, f, indent=2)
                print(f"Connection info saved to /tmp/exp028_pod.json")
                return info
        if i % 6 == 0:
            print(f"  [{i*10}s] booting...")

    print(f"Pod {pod_id} still booting after 20min. Check RunPod console.")
    return None


if __name__ == "__main__":
    launch()
