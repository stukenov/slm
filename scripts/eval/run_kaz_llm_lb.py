#!/usr/bin/env python3
"""Run kaz-llm-lb MC benchmarks on vast.ai using horde-common eval script.

Usage:
    PYTHONPATH=src .venv-cloud/bin/python scripts/run_kaz_llm_lb.py
"""

import os
import subprocess
import sys
import tempfile
import time

MODEL_ID = os.environ.get("EVAL_MODEL_ID", "stukenov/sozkz-core-llama-150m-kk-instruct-v2")

HF_TOKEN = ""
token_path = os.path.expanduser("~/.cache/huggingface/token")
if os.path.exists(token_path):
    HF_TOKEN = open(token_path).read().strip()


def ssh_cmd(host, port, cmd, timeout=7200):
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=60",
         "-p", str(port), f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def scp_to(host, port, local, remote, timeout=120):
    return subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port),
         local, f"root@{host}:{remote}"],
        check=True, timeout=timeout,
    )


def scp_from(host, port, remote, local, timeout=120):
    return subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port),
         f"root@{host}:{remote}", local],
        timeout=timeout,
    )


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from slm.cloud import vastai

    # 1. Create tarball of horde-common/scripts
    scripts_dir = "/tmp/horde-common/scripts"
    tarball = tempfile.mktemp(suffix=".tar.gz")
    print("Creating eval tarball...")
    subprocess.run(
        ["tar", "czf", tarball, "-C", "/tmp/horde-common", "scripts"],
        check=True,
    )

    # 2. Find GPU
    print("Searching for GPU...")
    min_gpu_ram = int(os.environ.get("EVAL_MIN_GPU_RAM", "20"))
    offers = vastai.search_offers(
        f"rentable=true num_gpus=1 compute_cap>=800 compute_cap<=890 gpu_ram>={min_gpu_ram} "
        "dph<=0.60 inet_down>=500 disk_space>=50"
    )
    if not offers:
        offers = vastai.search_offers(
            f"rentable=true num_gpus=1 compute_cap>=750 compute_cap<=890 gpu_ram>={min_gpu_ram} "
            "dph<=0.80 inet_down>=200 disk_space>=50"
        )
    if not offers:
        print("ERROR: No GPU offers")
        sys.exit(1)

    best = sorted(offers, key=lambda x: float(x.get("dph_total", 99)))[0]
    offer_id = int(best["id"])
    gpu = best.get("gpu_name", "?")
    price = float(best.get("dph_total", 0))
    print(f"Selected: {gpu} @ ${price:.3f}/hr (offer {offer_id})")

    # 3. Create instance
    print("Creating instance...")
    instance_id = vastai.create_instance(
        offer_id,
        image="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
        disk=30,
        label="kaz-llm-lb-eval",
    )
    print(f"Instance: {instance_id}")

    try:
        print("Waiting for instance...")
        vastai.wait_for_instance(instance_id, timeout=600)
        host, port = vastai.ssh_url(instance_id)
        print(f"SSH: {host}:{port}")

        # Wait for SSH
        for attempt in range(30):
            try:
                r = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-o", "BatchMode=yes", "-p", str(port), f"root@{host}", "echo ok"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    print(f"SSH ready (attempt {attempt+1})")
                    break
            except subprocess.TimeoutExpired:
                pass
            time.sleep(10)
        else:
            raise RuntimeError("SSH not reachable")

        # 4. Upload eval tarball
        print("Uploading eval code...")
        scp_to(host, port, tarball, "/tmp/eval.tar.gz")
        os.unlink(tarball)

        # 5. Setup and run
        setup_cmd = f"""
set -e
cd /tmp
tar xzf eval.tar.gz
cd scripts

# Install deps
pip install -q torch transformers datasets tqdm langchain langchain-core python-dotenv huggingface_hub bitsandbytes sentencepiece accelerate peft protobuf 2>&1 | tail -3

# Set HF token
export HUGGINGFACE_TOKEN="{HF_TOKEN}"
export HF_TOKEN="{HF_TOKEN}"
python -c "from huggingface_hub import login; login(token='{HF_TOKEN}')"

mkdir -p /tmp/results

echo "=== Running MC eval ==="
python mc-eval-simplified-inference.py \\
    --model_id "{MODEL_ID}" \\
    --output_path /tmp/results \\
    --dtype bfloat16

echo "=== DONE ==="
ls -la /tmp/results/ 2>/dev/null || true
ls -la *.json 2>/dev/null || true
"""
        print("Running MC eval (est. 10-30 min)...")
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=120",
             "-o", "TCPKeepAlive=yes",
             "-p", str(port), f"root@{host}", setup_cmd],
            capture_output=True, text=True, timeout=3600,
        )
        print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
        if result.returncode != 0 and result.stderr:
            print("STDERR:", result.stderr[-3000:])

        # 6. Download results
        os.makedirs("results", exist_ok=True)

        # The script saves JSON as {model_name_sanitized}.json in CWD (/tmp/scripts)
        model_sanitized = MODEL_ID.replace("/", "__")

        # Try downloading from scripts dir (CWD)
        scp_from(host, port,
                 f"/tmp/scripts/{model_sanitized}.json",
                 f"results/{model_sanitized}.json")
        print(f"Downloaded results/{model_sanitized}.json")

        # Also get CSV files from output_path
        find_result = ssh_cmd(host, port, "ls /tmp/results/", timeout=30)
        if find_result.stdout.strip():
            for f in find_result.stdout.strip().split("\n"):
                f = f.strip()
                if f:
                    scp_from(host, port, f"/tmp/results/{f}", f"results/{f}")
                    print(f"Downloaded results/{f}")

    finally:
        print(f"Destroying instance {instance_id}...")
        vastai.destroy_instance(instance_id)
        print("Done.")


if __name__ == "__main__":
    main()
