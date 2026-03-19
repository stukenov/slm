#!/usr/bin/env python3
"""Launch autonomous MC benchmark jobs on vast.ai.
Each GPU group gets its own instance running in screen.
No persistent SSH needed - fire and forget.

Usage:
    PYTHONPATH=src .venv-cloud/bin/python scripts/launch_batch_eval.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

HF_TOKEN = ""
token_path = os.path.expanduser("~/.cache/huggingface/token")
if os.path.exists(token_path):
    HF_TOKEN = open(token_path).read().strip()

# GPU groups: (label, min_gpu_ram, max_price, disk, models)
GROUPS = [
    {
        "label": "bench-small",
        "min_gpu_ram": 24,
        "max_price": 0.20,
        "disk": 50,
        "models": [
            "ai-forever/mGPT-1.3B-kazakh",
            "AmanMussa/llama2-kazakh-7b",
        ],
    },
    {
        "label": "bench-8b",
        "min_gpu_ram": 24,
        "max_price": 0.30,
        "disk": 60,
        "models": [
            "TilQazyna/llama-kaz-instruct-8B-1",
            "issai/LLama-3.1-KazLLM-1.0-8B",
            "inceptionai/Llama-3.1-Sherkala-8B-Chat",
        ],
    },
    {
        "label": "bench-13b",
        "min_gpu_ram": 48,
        "max_price": 0.50,
        "disk": 60,
        "models": [
            "ai-forever/mGPT-13B",
        ],
    },
    {
        "label": "bench-70b",
        "min_gpu_ram": 80,
        "max_price": 1.50,
        "disk": 150,
        "models": [
            "issai/LLama-3.1-KazLLM-1.0-70B",
        ],
    },
]


def ssh_cmd(host, port, cmd, timeout=300):
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         "-p", str(port), f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def scp_to(host, port, local, remote, timeout=120):
    return subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port),
         local, f"root@{host}:{remote}"],
        check=True, timeout=timeout,
    )


def create_tarball():
    """Create tarball with eval scripts."""
    tarball = tempfile.mktemp(suffix=".tar.gz")
    # Include horde-common/scripts + our remote_eval.py
    subprocess.run(
        ["tar", "czf", tarball,
         "-C", "/tmp/horde-common", "scripts",
         "-C", os.path.dirname(os.path.abspath(__file__)), "remote_eval.py"],
        check=True,
    )
    return tarball


def launch_group(group, tarball):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from slm.cloud import vastai

    label = group["label"]
    min_ram = group["min_gpu_ram"]
    max_price = group["max_price"]
    disk = group["disk"]
    models = group["models"]

    print(f"\n{'='*60}")
    print(f"GROUP: {label} ({len(models)} models, GPU>={min_ram}GB)")
    print(f"{'='*60}")

    # Search GPU
    query = (f"rentable=true num_gpus=1 compute_cap>=800 compute_cap<=890 "
             f"gpu_ram>={min_ram} dph<={max_price} inet_down>=200 disk_space>={disk}")
    offers = vastai.search_offers(query)

    if not offers:
        # Relax constraints
        query2 = (f"rentable=true num_gpus=1 compute_cap>=750 compute_cap<=890 "
                  f"gpu_ram>={min_ram} dph<={max_price * 1.5} inet_down>=100 disk_space>={disk}")
        offers = vastai.search_offers(query2)

    if not offers:
        print(f"  ERROR: No GPU offers for {label}")
        return None

    best = sorted(offers, key=lambda x: float(x.get("dph_total", 99)))[0]
    offer_id = int(best["id"])
    gpu = best.get("gpu_name", "?")
    price = float(best.get("dph_total", 0))
    print(f"  GPU: {gpu} @ ${price:.3f}/hr (offer {offer_id})")

    # Create instance
    instance_id = vastai.create_instance(
        offer_id,
        image="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
        disk=disk,
        label=label,
    )
    print(f"  Instance: {instance_id}")

    # Wait for SSH
    vastai.wait_for_instance(instance_id, timeout=600)
    host, port = vastai.ssh_url(instance_id)
    print(f"  SSH: {host}:{port}")

    for attempt in range(30):
        try:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                 "-o", "BatchMode=yes", "-p", str(port), f"root@{host}", "echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"  SSH ready (attempt {attempt+1})")
                break
        except subprocess.TimeoutExpired:
            pass
        time.sleep(10)
    else:
        print(f"  ERROR: SSH not reachable for {label}")
        return None

    # Upload tarball
    scp_to(host, port, tarball, "/tmp/eval.tar.gz")
    print(f"  Uploaded eval code")

    # Setup + launch in screen
    models_str = " ".join(f'"{m}"' for m in models)

    setup_cmd = f"""
set -e
cd /tmp && tar xzf eval.tar.gz
pip install -q torch transformers datasets tqdm langchain langchain-core \
    python-dotenv huggingface_hub bitsandbytes sentencepiece accelerate peft protobuf 2>&1 | tail -3
export HF_TOKEN="{HF_TOKEN}"
python -c "from huggingface_hub import login; login(token='{HF_TOKEN}')"
mkdir -p /tmp/results

# Install vastai CLI for self-destruct
pip install -q vastai 2>&1 | tail -1
mkdir -p ~/.config/vastai

# Launch in screen (autonomous)
screen -dmS bench bash -c '
export HF_TOKEN="{HF_TOKEN}"
cd /tmp
python remote_eval.py \
    --models {models_str} \
    --dtype bfloat16 \
    --hf-token "{HF_TOKEN}" \
    --instance-id "{instance_id}" \
    2>&1 | tee /tmp/results/bench.log
'
echo "Screen session started"
screen -ls
"""

    r = ssh_cmd(host, port, setup_cmd, timeout=600)
    print(f"  Setup output: {r.stdout[-500:]}")
    if r.stderr:
        print(f"  Setup stderr: {r.stderr[-300:]}")

    print(f"  LAUNCHED: {label} on {gpu} (instance {instance_id})")
    print(f"  Monitor: ssh -p {port} root@{host} 'tail -f /tmp/results/bench.log'")
    return {
        "label": label,
        "instance_id": instance_id,
        "host": host,
        "port": port,
        "gpu": gpu,
        "models": models,
    }


def main():
    tarball = create_tarball()
    print(f"Tarball: {tarball}")

    launched = []
    for group in GROUPS:
        try:
            info = launch_group(group, tarball)
            if info:
                launched.append(info)
        except Exception as e:
            print(f"ERROR launching {group['label']}: {e}")
            import traceback
            traceback.print_exc()

    os.unlink(tarball)

    print(f"\n{'='*60}")
    print("ALL LAUNCHED")
    print(f"{'='*60}")
    for info in launched:
        print(f"  [{info['label']}] Instance {info['instance_id']} on {info['gpu']}")
        print(f"    Models: {', '.join(info['models'])}")
        print(f"    Monitor: ssh -p {info['port']} root@{info['host']} 'tail -f /tmp/results/bench.log'")
    print(f"\nResults will auto-upload to {HF_REPO}")
    print("Instances will self-destruct after completion.")

    # Save launch info
    with open("results/batch_launch.json", "w") as f:
        json.dump(launched, f, indent=2)


HF_REPO = "stukenov/s-openbench-eval"

if __name__ == "__main__":
    main()
