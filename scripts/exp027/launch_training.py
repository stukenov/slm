#!/usr/bin/env python3
"""
Launch EkiTil 123M training on RunPod 1×H100 SXM 80GB.

Usage:
    python scripts/exp027/launch_training.py
"""
import json
import os
import subprocess
import sys
import time
import textwrap

try:
    import runpod
except ImportError:
    print("pip install runpod")
    sys.exit(1)

# RunPod API key
for path in [os.path.expanduser("~/.runpod/config.json")]:
    if os.path.exists(path):
        with open(path) as f:
            runpod.api_key = json.load(f).get("api_key", "")
        break

# HF token
HF_TOKEN = ""
for path in [os.path.expanduser("~/.cache/huggingface/token"), ".env"]:
    if os.path.exists(path):
        with open(path) as f:
            content = f.read().strip()
            if path.endswith(".env"):
                for line in content.split("\n"):
                    if line.startswith("HF_TOKEN="):
                        HF_TOKEN = line.split("=", 1)[1].strip().strip('"')
            else:
                HF_TOKEN = content
        if HF_TOKEN:
            break

DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# H100 GPU IDs to try (prefer SXM, fallback to PCIe)
H100_GPUS = [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100",
]


def wait_for_pod(pod_id, timeout=600):
    """Wait for pod to be ready with SSH."""
    print(f"Waiting for pod {pod_id}...")
    for i in range(timeout // 10):
        p = runpod.get_pod(pod_id)
        rt = p.get("runtime") or {}
        uptime = rt.get("uptimeInSeconds", 0)
        ports = rt.get("ports") or []
        if uptime > 0 and ports:
            for port in ports:
                if port.get("privatePort") == 22:
                    return port["ip"], port["publicPort"]
        if i % 6 == 0:
            print(f"  [{i*10}s] booting...")
        time.sleep(10)
    return None, None


def ssh_cmd(host, port):
    return ["ssh", "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=no",
            f"root@{host}", "-p", str(port)]


def scp_cmd(host, port, local, remote):
    return ["scp", "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=no",
            "-P", str(port), local, f"root@{host}:{remote}"]


def run(cmd, check=True):
    print(f"  $ {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        raise RuntimeError(f"Command failed: {result.returncode}")
    return result


def main():
    print("=" * 60)
    print("  EkiTil 123M — Launch Training on 1×H100")
    print("=" * 60)

    # 1. Create pod
    pod = None
    for gpu_id in H100_GPUS:
        try:
            print(f"\nTrying {gpu_id}...")
            pod = runpod.create_pod(
                name="ekitil-123m-train",
                image_name=DOCKER_IMAGE,
                gpu_type_id=gpu_id,
                gpu_count=1,
                volume_in_gb=0,
                container_disk_in_gb=100,
                ports="22/tcp",
                support_public_ip=True,
            )
            print(f"SUCCESS: {gpu_id} -> pod {pod['id']}")
            break
        except Exception as e:
            print(f"  unavailable: {str(e)[:100]}")

    if not pod:
        print("\nNo H100 available. Try again later.")
        sys.exit(1)

    pod_id = pod["id"]

    # 2. Wait for SSH
    host, port = wait_for_pod(pod_id)
    if not host:
        print(f"Pod {pod_id} timed out. Check RunPod console.")
        sys.exit(1)

    print(f"\n  Pod ready: ssh root@{host} -p {port}")
    print("  Waiting 15s for SSH daemon...")
    time.sleep(15)

    # 3. Upload training script
    print("\n--- Uploading training script ---")
    run(scp_cmd(host, port, "scripts/exp027/train_ekitil_123m.py", "/workspace/train_ekitil_123m.py"))

    # 4. Setup environment + install deps
    print("\n--- Installing dependencies ---")
    setup_script = textwrap.dedent(f"""\
        set -e
        echo "=== Setup ==="
        pip install -q datasets transformers huggingface_hub numpy accelerate
        export HF_TOKEN="{HF_TOKEN}"
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
        nvidia-smi | head -4
        python3 -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA {{torch.cuda.get_device_name(0)}}')"
        echo "=== Setup done ==="
    """)
    run(ssh_cmd(host, port) + ["bash", "-c", setup_script])

    # 5. Verify: quick smoke test (build model, 1 forward pass)
    print("\n--- Smoke test (build model + forward pass) ---")
    smoke_script = textwrap.dedent(f"""\
        set -e
        export HF_TOKEN="{HF_TOKEN}"
        cd /workspace
        python3 -c "
from train_ekitil_123m import build_model, download_data
import torch
# Build model
model = build_model('cuda')
print('Model OK')
# Quick forward
x = torch.randint(0, 64000, (2, 128), device='cuda')
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = model(input_ids=x)
    print(f'Forward OK, logits shape: {{out.logits.shape}}')
del model, x, out
torch.cuda.empty_cache()
print('Smoke test PASSED')
"
    """)
    result = run(ssh_cmd(host, port) + ["bash", "-c", smoke_script])
    print(result.stdout[-500:] if result.stdout else "")

    # 6. Verify HF auth
    print("\n--- Verify HF auth ---")
    hf_check = f'export HF_TOKEN="{HF_TOKEN}" && huggingface-cli whoami'
    result = run(ssh_cmd(host, port) + ["bash", "-c", hf_check])
    print(f"  HF user: {result.stdout.strip()[:100]}")

    # 7. Launch training in screen
    print("\n--- Launching training in screen ---")
    train_cmd = textwrap.dedent(f"""\
        export HF_TOKEN="{HF_TOKEN}"
        cd /workspace
        screen -dmS ekitil bash -c '
            export HF_TOKEN="{HF_TOKEN}"
            python3 train_ekitil_123m.py \\
                --batch-size 32 \\
                --grad-accum 4 \\
                --lr 6e-4 \\
                --warmup-steps 2000 \\
                2>&1 | tee /workspace/train.log
            echo "TRAINING DONE" >> /workspace/train.log
        '
        sleep 2
        screen -ls
    """)
    result = run(ssh_cmd(host, port) + ["bash", "-c", train_cmd])
    print(result.stdout[-300:] if result.stdout else "")

    print("\n" + "=" * 60)
    print(f"  TRAINING LAUNCHED!")
    print(f"  Pod: {pod_id}")
    print(f"  SSH: ssh root@{host} -p {port}")
    print(f"  Monitor: ssh root@{host} -p {port} 'tail -f /workspace/train.log'")
    print(f"  Screen: ssh root@{host} -p {port} 'screen -r ekitil'")
    print(f"  H100 1×80GB, batch=32, grad_accum=4, lr=6e-4")
    print(f"  ~262K tok/step, ~9,400 steps, ETA ~2-4h")
    print("=" * 60)

    # Save pod info
    info = {
        "pod_id": pod_id,
        "host": host,
        "port": port,
        "gpu": "H100 80GB",
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "batch_size": 32,
            "grad_accum": 4,
            "lr": "6e-4",
            "warmup_steps": 2000,
        }
    }
    with open("scripts/exp027/pod_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n  Pod info saved to scripts/exp027/pod_info.json")


if __name__ == "__main__":
    main()
