#!/usr/bin/env python3
"""
Launch instruct dataset collection + translation on vast.ai.

1. Find cheap 2-GPU instance
2. Upload project + translation model
3. Run prepare_instruct_chatml.py → translate_instruct.py → upload to HF

Usage:
    PYTHONPATH=src .venv-cloud/bin/python scripts/run_instruct_vastai.py --dry-run
    PYTHONPATH=src .venv-cloud/bin/python scripts/run_instruct_vastai.py --monitor
    PYTHONPATH=src .venv-cloud/bin/python scripts/run_instruct_vastai.py --num-gpus 1 --max-price 0.30
"""

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR / "src"))
from slm.cloud import vastai

DEFAULT_IMAGE = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
HF_REPO = "stukenov/sozkz-instruct-chatml-kk-v1"

# Include translation model in tarball (up to 500MB per file)
TAR_EXCLUDES = {
    ".git", ".venv", ".venv-cloud", "venv", "outputs", "logs", "__pycache__",
    ".cache", "wandb", "node_modules", ".ruff_cache", "Kaz-Offline-Arena",
    "omniaudio", "nanochat-kazakh", "600", "model_ct2", "model_cache",
}
# Translation model downloaded on instance from HF: HPLT/translate-en-kk-v2.0-hplt_opus


def resolve_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_path.exists():
        token = cache_path.read_text().strip()
        if token:
            return token
    raise RuntimeError("HF token not found")


def create_tarball() -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()

    def _filter(info: tarfile.TarInfo):
        parts = Path(info.name).parts
        for part in parts:
            if part in TAR_EXCLUDES:
                return None
        if info.size > 500 * 1024 * 1024:
            return None
        return info

    with tarfile.open(tmp.name, "w:gz") as tar:
        tar.add(str(PROJECT_DIR), arcname="slm", filter=_filter)

    size_mb = os.path.getsize(tmp.name) / 1024 / 1024
    print(f"Tarball: {size_mb:.1f} MB")
    return Path(tmp.name)


def ssh_run(host, port, cmd, timeout=600):
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
         "-p", str(port), f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SSH failed: {cmd}\n{result.stderr[:500]}")
    return result.stdout


def scp_to(local, host, port, remote):
    result = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port),
         str(local), f"root@{host}:{remote}"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"scp failed: {result.stderr[:500]}")


def wait_ssh(host, port, retries=60, delay=10):
    for i in range(1, retries + 1):
        try:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                 "-o", "BatchMode=yes", "-p", str(port), f"root@{host}", "echo ok"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                print(f"SSH ready (attempt {i})")
                return
        except subprocess.TimeoutExpired:
            pass
        print(f"SSH not ready ({i}/{retries})...")
        time.sleep(delay)
    raise RuntimeError("SSH timeout")


def generate_run_script(num_gpus, hf_token, instance_id, vast_api_key):
    return f"""\
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_TOKEN='{hf_token}'

LOGFILE="/root/slm/logs/instruct_pipeline.log"
mkdir -p /root/slm/logs
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== Instruct pipeline started: $(date -u) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\\n' ', ')"
df -h /

cd /root/slm

# ---- Install deps ----
echo ">>> Installing dependencies..."
pip install -q datasets sentencepiece ctranslate2 huggingface_hub 2>&1 | tail -5

# ---- Download translation model from HuggingFace ----
echo ">>> Downloading HPLT EN→KK translation model..."
python scripts/download_translation_model.py
echo ">>> Translation model downloaded and converted."

# ---- Phase 1: Download pre-collected EN dataset from HuggingFace ----
echo ">>> Phase 1: Downloading EN dataset from HuggingFace..."
python -c "
from datasets import load_dataset
ds = load_dataset('stukenov/sozkz-instruct-chatml-en-v1', split='train')
print(f'Downloaded {{len(ds)}} rows')
ds.to_parquet('data/instruct_chatml_en.parquet')
print('Saved to parquet')
"
echo ">>> Download complete."

# ---- Phase 2: Translate EN→KK ----
echo ">>> Phase 2: Translating EN→KK ({num_gpus} GPUs)..."
python scripts/translate_instruct.py \\
    --input data/instruct_chatml_en.parquet \\
    --output data/instruct_chatml_kk.parquet \\
    --num-gpus {num_gpus} \\
    --checkpoint-every 50000 \\
    --upload \\
    --repo {HF_REPO}
echo ">>> Translation and upload complete."

# ---- Self-destruct ----
echo "=== Pipeline finished: $(date -u) ==="
sleep 60
INSTANCE_ID="{instance_id}"
VAST_KEY="{vast_api_key}"
if [ -n "$INSTANCE_ID" ] && [ -n "$VAST_KEY" ]; then
    echo ">>> Self-destructing instance $INSTANCE_ID..."
    curl -s -X PUT -H "Authorization: Bearer $VAST_KEY" \\
        "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" \\
        -d '{{"state": "stopped"}}' || true
    curl -s -X DELETE -H "Authorization: Bearer $VAST_KEY" \\
        "https://console.vast.ai/api/v0/instances/$INSTANCE_ID/" || true
fi
"""


def main():
    parser = argparse.ArgumentParser(description="Run instruct pipeline on vast.ai")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-price", type=float, default=0.60)
    parser.add_argument("--disk", type=int, default=80)
    parser.add_argument("--gpu", default=None, help="Force GPU type")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()

    hf_token = resolve_hf_token()
    print(f"HF token OK (len={len(hf_token)})")

    # Find GPU offer
    print(f"Searching for {args.num_gpus}-GPU offers (max ${args.max_price}/hr)...")
    query_parts = [
        "rentable=true",
        f"num_gpus={args.num_gpus}",
        f"disk_space>={args.disk}",
        "reliability>0.9",
    ]
    if args.gpu:
        query_parts.append(f"gpu_name={args.gpu}")
    raw = vastai.search_offers(query=" ".join(query_parts))

    # Filter by price and sort
    offers = [o for o in raw if float(o.get("dph_total", 999)) <= args.max_price]
    if not offers:
        print(f"No offers found under ${args.max_price}/hr. Available:")
        for o in sorted(raw, key=lambda x: float(x.get("dph_total", 999)))[:5]:
            print(f"  {o.get('gpu_name', '?')} x{o.get('num_gpus', '?')}: ${float(o.get('dph_total', 0)):.3f}/hr")
        sys.exit(1)

    # Sort by value: prefer higher download speed and VRAM, penalize price
    def score(o):
        price = max(float(o.get("dph_total", 1)), 0.01)
        vram = float(o.get("gpu_ram", 0))
        dl = float(o.get("dlperf", o.get("dl_perf", 1)))
        return dl * (vram ** 0.5) / price
    offers.sort(key=score, reverse=True)
    best = offers[0]
    gpu_name = best.get("gpu_name", "?")
    dph = float(best.get("dph_total", 0))
    print(f"Best: {gpu_name} x{best.get('num_gpus', '?')} @ ${dph:.3f}/hr")

    if args.dry_run:
        print("\nTop 5 offers:")
        for o in offers[:5]:
            print(f"  {o.get('gpu_name')} x{o.get('num_gpus')}: ${float(o.get('dph_total', 0)):.3f}/hr, "
                  f"VRAM={o.get('gpu_ram', '?')}MB, disk={o.get('disk_space', '?')}GB")
        return

    offer_id = int(best["id"])

    # Create instance
    env_vars = {"HF_TOKEN": hf_token}
    onstart = f"mkdir -p ~/.cache/huggingface && echo '{hf_token}' > ~/.cache/huggingface/token"
    print(f"Creating instance (offer {offer_id})...")
    instance_id = vastai.create_instance(
        offer_id, image=DEFAULT_IMAGE, disk=args.disk,
        onstart_cmd=onstart, env_vars=env_vars, label="slm-instruct-pipeline",
    )
    print(f"Instance: {instance_id}")

    try:
        print("Waiting for instance...")
        vastai.wait_for_instance(instance_id, timeout=600)
        host, port = vastai.ssh_url(instance_id)
        print(f"SSH: root@{host} -p {port}")
        wait_ssh(host, port)

        # Upload project
        print("Packing project (incl. translation model)...")
        tarball = create_tarball()
        try:
            print("Uploading...")
            scp_to(tarball, host, port, "/root/project.tar.gz")
        finally:
            tarball.unlink(missing_ok=True)

        # Extract + setup
        print("Extracting on remote...")
        ssh_run(host, port, "cd /root && tar xzf project.tar.gz", timeout=120)

        # Upload run script
        vast_api_key = vastai._get_api_key()
        actual_gpus = int(best.get("num_gpus", args.num_gpus))
        script = generate_run_script(actual_gpus, hf_token, instance_id, vast_api_key)

        proc = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(port), f"root@{host}",
             "cat > /root/slm/run_pipeline.sh && chmod +x /root/slm/run_pipeline.sh"],
            input=script, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to upload script: {proc.stderr[:300]}")

        # Launch
        print("Launching pipeline...")
        ssh_run(host, port,
                "cd /root/slm && setsid bash run_pipeline.sh > /root/slm/logs/instruct_pipeline.log 2>&1 < /dev/null &",
                timeout=30)
        time.sleep(3)

        print(f"\nPipeline launched!")
        print(f"  Instance: {instance_id}")
        print(f"  GPU: {gpu_name} x{actual_gpus} @ ${dph:.3f}/hr")
        print(f"  HF repo: {HF_REPO}")
        print(f"\nMonitor: PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud monitor --instance-id {instance_id}")
        print(f"Destroy: PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud destroy --instance-id {instance_id}")

        if args.monitor:
            print("\nTailing log (Ctrl+C to detach)...\n")
            try:
                p = subprocess.Popen(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                     "-p", str(port), f"root@{host}",
                     "tail -f /root/slm/logs/instruct_pipeline.log"],
                )
                p.wait()
            except KeyboardInterrupt:
                print("\nDetached.")

    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Instance {instance_id} may still be running!")
        print(f"Destroy: PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud destroy --instance-id {instance_id}")
        raise


if __name__ == "__main__":
    main()
