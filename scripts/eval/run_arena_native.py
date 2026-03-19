#!/usr/bin/env python3
"""Run Kaz-Offline-Arena natively on vast.ai and download judge results.

Usage:
    PYTHONPATH=src .venv-cloud/bin/python scripts/run_arena_native.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

MODEL_ID = "stukenov/sozkz-core-llama-150m-kk-instruct-v2"

# Read secrets from .env
OPENAI_API_KEY = ""
HF_TOKEN = ""

env_path = os.path.join(os.path.dirname(__file__), "..", "Kaz-Offline-Arena", ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if line.startswith("OPENAI_API_KEY="):
            OPENAI_API_KEY = line.split("=", 1)[1]
        elif line.startswith("HUGGINGFACE_TOKEN="):
            HF_TOKEN = line.split("=", 1)[1]

if not HF_TOKEN:
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

    # 1. Create tarball of Kaz-Offline-Arena
    arena_dir = os.path.join(os.path.dirname(__file__), "..", "Kaz-Offline-Arena")
    tarball = tempfile.mktemp(suffix=".tar.gz")
    print("Creating arena tarball...")
    subprocess.run(
        ["tar", "czf", tarball,
         "-C", os.path.dirname(arena_dir),
         "Kaz-Offline-Arena"],
        check=True,
    )

    # 2. Find GPU
    print("Searching for GPU...")
    offers = vastai.search_offers(
        "rentable=true num_gpus=1 compute_cap>=800 compute_cap<=890 gpu_ram>=20 "
        "dph<=0.60 inet_down>=500 disk_space>=30"
    )
    if not offers:
        offers = vastai.search_offers(
            "rentable=true num_gpus=1 compute_cap>=750 compute_cap<=890 gpu_ram>=16 "
            "dph<=0.50 inet_down>=200 disk_space>=30"
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
        label="arena-native",
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

        # 4. Upload arena tarball
        print("Uploading arena code...")
        scp_to(host, port, tarball, "/tmp/arena.tar.gz")
        os.unlink(tarball)

        # 5. Setup and run
        setup_cmd = f"""
set -e
cd /tmp
tar xzf arena.tar.gz
cd Kaz-Offline-Arena

# Write .env
cat > .env << 'ENVEOF'
OPENAI_API_KEY={OPENAI_API_KEY}
HUGGINGFACE_TOKEN={HF_TOKEN}
ENVEOF

# Install deps (simplified - skip git installs, use pip packages)
pip install -q torch transformers datasets tqdm openai pydantic tenacity python-dotenv fire pandas huggingface_hub bitsandbytes choix numpy 2>&1 | tail -3

echo "=== Running inference ==="
python main.py inference --model_id="{MODEL_ID}" --question_types="WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS" --batch_size=1

echo "=== Running judge ==="
python main.py judge

echo "=== DONE ==="
ls -la output/judge/
"""
        print("Running inference + judge (est. 30-60 min)...")
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=60",
             "-p", str(port), f"root@{host}", setup_cmd],
            capture_output=True, text=True, timeout=7200,
        )
        print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
        if result.returncode != 0 and result.stderr:
            print("STDERR:", result.stderr[-3000:])

        # 6. Download results
        os.makedirs("results", exist_ok=True)

        # Find judge result file
        find_result = ssh_cmd(host, port, "ls /tmp/Kaz-Offline-Arena/output/judge/", timeout=30)
        judge_files = [f.strip() for f in find_result.stdout.strip().split("\n") if f.strip()]
        print(f"Judge files: {judge_files}")

        for jf in judge_files:
            scp_from(host, port,
                     f"/tmp/Kaz-Offline-Arena/output/judge/{jf}",
                     f"results/{jf}")
            print(f"Downloaded results/{jf}")

        # Also download inference results
        find_inf = ssh_cmd(host, port, "ls /tmp/Kaz-Offline-Arena/output/inference/", timeout=30)
        inf_files = [f.strip() for f in find_inf.stdout.strip().split("\n") if f.strip()]
        for inf_f in inf_files:
            scp_from(host, port,
                     f"/tmp/Kaz-Offline-Arena/output/inference/{inf_f}",
                     f"results/{inf_f}")
            print(f"Downloaded results/{inf_f}")

    finally:
        print(f"Destroying instance {instance_id}...")
        vastai.destroy_instance(instance_id)
        print("Done.")


if __name__ == "__main__":
    main()
