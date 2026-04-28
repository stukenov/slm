"""
OmniAudio v2: Mass training across CloudRift GPU instances.

Rents one GPU per model size, deploys code, runs 2-stage training (CTC → E2E).
Uses CloudRift REST API directly (no CLI needed).

Usage:
  # List available GPUs
  python cloudrift_train.py list

  # Rent instances and launch training for all sizes
  python cloudrift_train.py launch --sizes 50m 150m 600m 1b

  # Launch only specific sizes
  python cloudrift_train.py launch --sizes 50m 150m

  # Check status of running instances
  python cloudrift_train.py status

  # Monitor training logs
  python cloudrift_train.py monitor

  # Terminate specific instances (ASKS CONFIRMATION)
  python cloudrift_train.py terminate

Environment:
  CLOUDRIFT_API_KEY  — API key (or pass --api-key)
  HF_TOKEN           — HuggingFace token for model upload
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CloudRift API client
# ---------------------------------------------------------------------------

API_BASE = "https://api.cloudrift.ai"
API_VERSION = "~upcoming"


def _api_call(endpoint: str, data: dict, api_key: str, retries: int = 8) -> dict:
    """POST to CloudRift API. Returns parsed JSON response. Retries on 500 TransactionSerializationError."""
    import urllib.request
    import urllib.error

    url = f"{API_BASE}{endpoint}"
    payload = json.dumps({"version": API_VERSION, "data": data}).encode()

    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            data=payload,
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            if e.code == 500 and "TransactionSerializationError" in body and attempt < retries - 1:
                wait = (attempt + 1) * 15
                print(f"  API 500 TransactionSerializationError, retrying in {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            print(f"API error {e.code} on {endpoint}: {body[:500]}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Model size configs — maps size to GPU requirements + training params
# ---------------------------------------------------------------------------

# GPU preference order per model size (name prefix → min VRAM in bytes)
# RTX 4090 = 24GB, L40S = 48GB, RTX PRO 6000 = 96GB
VRAM_24GB = 24 * 1024**3
VRAM_32GB = 32 * 1024**3
VRAM_48GB = 48 * 1024**3

MODEL_CONFIGS = {
    "50m": {
        "min_vram": VRAM_24GB,  # fits on RTX 4090 (24GB)
        "gpu_count": 1,
        "ctc_config": "omniaudio/configs/v2_llm50m_ctc_cloudrift.yaml",
        "e2e_config": "omniaudio/configs/v2_llm50m_e2e_cloudrift.yaml",
    },
    "150m": {
        "min_vram": VRAM_24GB,  # fits on RTX 4090 (24GB)
        "gpu_count": 1,
        "ctc_config": "omniaudio/configs/v2_llm150m_ctc_cloudrift.yaml",
        "e2e_config": "omniaudio/configs/v2_llm150m_e2e_cloudrift.yaml",
    },
    "600m": {
        "min_vram": VRAM_24GB,  # CTC fits on 24GB; E2E with 600M LLM needs 32GB+
        "gpu_count": 1,
        "ctc_config": "omniaudio/configs/v2_llm600m_ctc_cloudrift.yaml",
        "e2e_config": "omniaudio/configs/v2_llm600m_e2e_cloudrift.yaml",
    },
    "1b": {
        "min_vram": VRAM_32GB,  # E2E with frozen 1B LLM needs 32GB+
        "gpu_count": 1,
        "ctc_config": "omniaudio/configs/v2_llm1b_ctc_cloudrift.yaml",
        "e2e_config": "omniaudio/configs/v2_llm1b_e2e_cloudrift.yaml",
    },
}

# State file to track rented instances
STATE_FILE = Path(__file__).parent / "cloudrift_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"instances": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _find_ssh_key() -> Path | None:
    """Find the best available SSH key."""
    for name in ["cloudrift_omniaudio", "id_ed25519", "id_rsa"]:
        p = Path.home() / ".ssh" / name
        if p.exists():
            return p
    return None


def _run_ssh(ip: str, cmd: str, ssh_key_path: Path | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a command via SSH using subprocess (safe, no shell injection)."""
    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
    ]
    if ssh_key_path and ssh_key_path.exists():
        ssh_args += ["-i", str(ssh_key_path)]
    ssh_args += [f"root@{ip}", cmd]
    return subprocess.run(ssh_args, capture_output=True, text=True, timeout=timeout)


def _run_rsync(src: str, ip: str, dst: str, ssh_key_path: Path | None = None) -> subprocess.CompletedProcess:
    """Run rsync to deploy code (safe, no shell injection)."""
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30"
    if ssh_key_path and ssh_key_path.exists():
        ssh_cmd += f" -i {ssh_key_path}"
    rsync_args = [
        "rsync", "-avz", "--progress",
        "-e", ssh_cmd,
        "--exclude=.git", "--exclude=__pycache__", "--exclude=.venv*",
        "--exclude=outputs/", "--exclude=logs/", "--exclude=*.pt",
        "--exclude=nano/", "--exclude=kz-calm/",
        f"{src}/", f"root@{ip}:{dst}/",
    ]
    return subprocess.run(rsync_args, timeout=600)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args):
    """List available GPU instance types on CloudRift."""
    api_key = args.api_key
    result = _api_call("/api/v1/instance-types/list", {"selector": "All"}, api_key)

    print(f"\n{'Variant':<35} {'Brand':<18} {'$/hr':>8} {'GPUs':>5} {'VRAM':>8} {'Avail':>6}")
    print("-" * 90)

    for it in result["data"]["instance_types"]:
        brand = it.get("brand_short", "—")

        for v in it.get("variants", []):
            gpu_count = v.get("gpu_count", 0)
            if gpu_count == 0:
                continue  # skip CPU-only
            vram_gb = v.get("vram", 0) / (1024**3)
            v_cost = v.get("cost_per_hour", 0) / 100
            v_avail = v.get("available_nodes", 0)
            marker = " <<<" if v_avail > 0 else ""
            print(f"  {v['name']:<33} {brand:<18} ${v_cost:>6.2f} {gpu_count:>5} {vram_gb:>6.0f}GB {v_avail:>6}{marker}")

    print()


def _find_best_variant(instance_types: list, min_vram: int, gpu_count: int) -> tuple:
    """Find cheapest available variant matching VRAM and GPU requirements.
    Returns (variant_name, cost_per_hour_cents) or (None, None)."""
    candidates = []
    for it in instance_types:
        for v in it.get("variants", []):
            if v.get("gpu_count", 0) < gpu_count:
                continue
            if v.get("vram", 0) < min_vram:
                continue
            avail = v.get("available_nodes", 0)
            if avail <= 0:
                continue
            candidates.append((v["name"], v.get("cost_per_hour", 99999), it["name"]))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0], candidates[0][1]


def _get_ssh_key_id(api_key: str) -> str:
    """Get the first SSH key ID, or generate one."""
    result = _api_call("/api/v1/ssh-keys/list", {}, api_key)
    keys = result["data"]["keys"]
    if keys:
        return keys[0]["id"]
    print("No SSH key found, generating one...")
    result = _api_call("/api/v1/ssh-keys/add", {"name": "omniaudio-training"}, api_key)
    priv_key_bytes = result["data"]["private_key"]
    if isinstance(priv_key_bytes, list):
        priv_pem = bytes(priv_key_bytes).decode()
    else:
        priv_pem = priv_key_bytes
    key_path = Path.home() / ".ssh" / "cloudrift_omniaudio"
    key_path.write_text(priv_pem)
    key_path.chmod(0o600)
    print(f"  Private key saved to {key_path}")
    result = _api_call("/api/v1/ssh-keys/list", {}, api_key)
    return result["data"]["keys"][-1]["id"]


def cmd_dryrun(args):
    """Dry run: show what would be rented, estimated cost, speed estimates."""
    api_key = args.api_key
    sizes = args.sizes

    result = _api_call("/api/v1/instance-types/list", {"selector": "All"}, api_key)
    instance_types = result["data"]["instance_types"]

    # Dataset: ~1M samples, 2100h audio
    DATASET_SAMPLES = 1_000_000

    print("\n=== DRY RUN: OmniAudio CloudRift Training Plan ===\n")

    total_cost_hr = 0
    plan = []
    for size in sizes:
        cfg = MODEL_CONFIGS[size]
        variant_name, cost = _find_best_variant(
            instance_types, cfg["min_vram"], cfg["gpu_count"]
        )

        # Estimate training time based on batch size and dataset
        import yaml
        ctc_config_path = Path(__file__).parent.parent / cfg["ctc_config"].replace("omniaudio/", "")
        e2e_config_path = Path(__file__).parent.parent / cfg["e2e_config"].replace("omniaudio/", "")

        ctc_batch = 32  # default
        e2e_batch = 16
        ctc_epochs = 2
        e2e_epochs = 5
        if ctc_config_path.exists():
            with open(ctc_config_path) as f:
                cc = yaml.safe_load(f)
            ctc_batch = cc.get("per_device_train_batch_size", 32)
            ctc_epochs = cc.get("num_train_epochs", 2)
            grad_accum_ctc = cc.get("gradient_accumulation_steps", 1)
        if e2e_config_path.exists():
            with open(e2e_config_path) as f:
                ec = yaml.safe_load(f)
            e2e_batch = ec.get("per_device_train_batch_size", 16)
            e2e_epochs = ec.get("num_train_epochs", 5)
            grad_accum_e2e = ec.get("gradient_accumulation_steps", 1)

        ctc_steps = (DATASET_SAMPLES // ctc_batch) * ctc_epochs
        e2e_steps = (DATASET_SAMPLES // e2e_batch) * e2e_epochs

        # Rough step/sec estimates by model size (conservative)
        step_sec = {"50m": 0.3, "150m": 0.6, "600m": 1.2, "1b": 2.5}
        sec_per_step = step_sec.get(size, 1.0)

        ctc_hours = (ctc_steps * sec_per_step) / 3600
        e2e_hours = (e2e_steps * sec_per_step * 1.5) / 3600  # E2E ~1.5x slower
        total_hours = ctc_hours + e2e_hours
        cost_usd = (cost / 100) * total_hours if cost else 0

        status = f"AVAILABLE ({variant_name})" if variant_name else "NO GPU AVAILABLE"
        plan.append({
            "size": size, "variant": variant_name, "cost_hr": cost,
            "ctc_batch": ctc_batch, "e2e_batch": e2e_batch,
            "ctc_steps": ctc_steps, "e2e_steps": e2e_steps,
            "ctc_hours": ctc_hours, "e2e_hours": e2e_hours,
            "total_hours": total_hours, "total_cost": cost_usd,
            "status": status,
        })
        if cost:
            total_cost_hr += cost / 100

    for p in plan:
        print(f"  {p['size'].upper()}: {p['status']}")
        if p["variant"]:
            print(f"    GPU: {p['variant']} (${p['cost_hr']/100:.2f}/hr)")
            print(f"    CTC: batch={p['ctc_batch']}, ~{p['ctc_steps']:,} steps, ~{p['ctc_hours']:.1f}h")
            print(f"    E2E: batch={p['e2e_batch']}, ~{p['e2e_steps']:,} steps, ~{p['e2e_hours']:.1f}h")
            print(f"    Total: ~{p['total_hours']:.1f}h, ~${p['total_cost']:.2f}")
        print()

    total_cost = sum(p["total_cost"] for p in plan)
    max_hours = max((p["total_hours"] for p in plan), default=0)
    print(f"  === SUMMARY ===")
    print(f"  Parallel runtime: ~{max_hours:.1f}h (all sizes run simultaneously)")
    print(f"  Combined GPU-hours: ~{sum(p['total_hours'] for p in plan):.1f}h")
    print(f"  Estimated total cost: ~${total_cost:.2f}")
    print(f"  Running cost/hr (all instances): ${total_cost_hr:.2f}/hr")

    # Check prerequisites
    print(f"\n  === PREREQUISITES ===")
    ssh_key = _find_ssh_key()
    print(f"  SSH key: {'OK (' + str(ssh_key) + ')' if ssh_key else 'MISSING'}")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
        if hf_token_path.exists():
            hf_token = hf_token_path.read_text().strip()
    print(f"  HF token: {'OK' if hf_token else 'MISSING'}")
    acct = _api_call("/api/v1/account/info", {}, api_key)
    balance = acct["data"]["balance"] / 100
    print(f"  CloudRift balance: ${balance:.2f}")
    enough = balance >= total_cost
    print(f"  Budget check: {'OK' if enough else 'INSUFFICIENT'} (need ~${total_cost:.2f})")
    print()


def cmd_launch(args):
    """Rent instances and launch training for specified sizes."""
    api_key = args.api_key
    sizes = list(args.sizes)
    state = load_state()

    for size in list(sizes):
        if size in state["instances"]:
            inst = state["instances"][size]
            print(f"  {size}: already tracked (instance {inst['instance_id']}), skipping")
            sizes = [s for s in sizes if s != size]

    if not sizes:
        print("All requested sizes already have instances. Use 'status' or 'terminate' first.")
        return

    result = _api_call("/api/v1/instance-types/list", {"selector": "All"}, api_key)
    instance_types = result["data"]["instance_types"]

    ssh_key_id = _get_ssh_key_id(api_key)
    print(f"Using SSH key: {ssh_key_id}")

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
        if hf_token_path.exists():
            hf_token = hf_token_path.read_text().strip()
    if not hf_token:
        print("WARNING: No HF_TOKEN found — models won't be uploaded to HuggingFace")

    launched = []
    for size in sizes:
        cfg = MODEL_CONFIGS[size]
        variant_name, cost = _find_best_variant(
            instance_types, cfg["min_vram"], cfg["gpu_count"]
        )
        if not variant_name:
            print(f"  {size}: No available GPU with >={cfg['min_vram']/(1024**3):.0f}GB VRAM. Skipping.")
            continue

        print(f"  {size}: renting {variant_name} (${cost/100:.2f}/hr)...")

        setup_script = _build_setup_script(size, hf_token)

        rent_data = {
            "selector": {"ByInstanceTypeAndLocation": {"instance_type": variant_name}},
            "with_public_ip": True,
            "name": f"omniaudio-{size}",
            "recipe": "Ubuntu 24.04 Server (R580, CUDA 12.9)",
            "config": {
                "VirtualMachine": {
                    "image_url": "https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpu-580-129-20251015-183936.img",
                    "ssh_key": {"ByName": ["stukenov"]},
                }
            },
        }

        try:
            resp = _api_call("/api/v1/instances/rent", rent_data, api_key)
            instance_ids = resp["data"]["instance_ids"]
            instance_id = instance_ids[0]
            print(f"  {size}: rented instance {instance_id}")

            state["instances"][size] = {
                "instance_id": instance_id,
                "variant": variant_name,
                "cost_per_hour": cost,
                "stage": "setup",
                "rented_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_state(state)
            launched.append(size)
        except Exception as e:
            print(f"  {size}: rent failed: {e}")

    if launched:
        print(f"\nLaunched: {', '.join(launched)}")
        print("Instances are booting. Run 'status' to check, then 'deploy' when ready.")
    else:
        print("\nNo instances launched. Check GPU availability with 'list'.")


def _build_setup_script(size: str, hf_token: str) -> str:
    """Build a cloud-init script that sets up the training environment."""
    return textwrap.dedent(f"""\
        #!/bin/bash
        set -e
        apt-get update && apt-get install -y screen git python3-pip python3-venv
        pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        pip3 install transformers datasets huggingface_hub pyyaml soundfile librosa
        mkdir -p ~/.cache/huggingface
        echo '{hf_token}' > ~/.cache/huggingface/token
        huggingface-cli whoami || true
        echo "Setup complete for omniaudio-{size}"
    """)


def cmd_status(args):
    """Check status of all tracked instances."""
    api_key = args.api_key
    state = load_state()

    if not state["instances"]:
        print("No tracked instances. Run 'launch' first.")
        return

    instance_ids = [v["instance_id"] for v in state["instances"].values()]
    result = _api_call("/api/v1/instances/list",
                       {"selector": {"ById": instance_ids}}, api_key)

    instances_by_id = {}
    for inst in result["data"]["instances"]:
        instances_by_id[inst["id"]] = inst

    print(f"\n{'Size':<8} {'Instance ID':<40} {'Status':<12} {'Stage':<10} {'$/hr':>8}")
    print("-" * 85)
    for size, info in state["instances"].items():
        iid = info["instance_id"]
        live = instances_by_id.get(iid, {})
        status = live.get("status", "unknown")
        stage = info.get("stage", "?")
        cost = info.get("cost_per_hour", 0) / 100
        ip = live.get("public_ip", live.get("ip", ""))

        print(f"  {size:<6} {iid:<40} {status:<12} {stage:<10} ${cost:>6.2f}")
        if ip:
            print(f"         IP: {ip}")
            # Update IP in state
            state["instances"][size]["ip"] = ip

    save_state(state)
    print()


def cmd_deploy(args):
    """Deploy code to running instances and start training."""
    api_key = args.api_key
    state = load_state()

    if not state["instances"]:
        print("No tracked instances.")
        return

    instance_ids = [v["instance_id"] for v in state["instances"].values()]
    result = _api_call("/api/v1/instances/list",
                       {"selector": {"ById": instance_ids}}, api_key)

    instances_by_id = {}
    for inst in result["data"]["instances"]:
        instances_by_id[inst["id"]] = inst

    project_root = Path(__file__).parent.parent.parent  # slm/
    ssh_key = _find_ssh_key()

    for size, info in state["instances"].items():
        iid = info["instance_id"]
        live = instances_by_id.get(iid, {})
        status = live.get("status", "unknown")
        ip = live.get("public_ip", live.get("ip", ""))

        if status != "running":
            print(f"  {size}: status={status}, not ready. Skipping.")
            continue
        if not ip:
            print(f"  {size}: no IP address yet. Skipping.")
            continue

        print(f"  {size}: deploying to {ip}...")
        cfg = MODEL_CONFIGS[size]

        # rsync project
        print(f"    rsync...")
        ret = _run_rsync(str(project_root), ip, "/root/slm", ssh_key)
        if ret.returncode != 0:
            print(f"    rsync failed (exit {ret.returncode}). Skipping {size}.")
            continue

        # Install omniaudio package
        r = _run_ssh(ip, "cd /root/slm && pip3 install -e '.[audio]' 2>&1 | tail -3", ssh_key, timeout=300)
        print(f"    pip install: {r.stdout.strip()}")

        # Create the training launch script
        train_script = _build_train_script(size, cfg)
        _run_ssh(ip, f"cat > /root/train_{size}.sh << 'TRAINEOF'\n{train_script}\nTRAINEOF\nchmod +x /root/train_{size}.sh", ssh_key)

        # Launch in screen
        _run_ssh(ip, f"mkdir -p /root/slm/logs && screen -dmS train_{size} bash /root/train_{size}.sh", ssh_key)

        print(f"    {size}: training launched in screen 'train_{size}'")
        state["instances"][size]["stage"] = "ctc_training"
        state["instances"][size]["ip"] = ip
        save_state(state)

    print("\nDone. Use 'monitor' to watch progress.")


def _build_train_script(size: str, cfg: dict) -> str:
    """Build the training launch script for a specific size."""
    return textwrap.dedent(f"""\
        #!/bin/bash
        set -e
        cd /root/slm
        export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
        mkdir -p logs

        echo "=========================================="
        echo "STAGE 1: CTC Pretrain — {size}"
        echo "=========================================="
        python -m omniaudio.train_v2 \\
            --config {cfg['ctc_config']} \\
            2>&1 | tee logs/cloudrift_{size}_ctc.log

        echo "=========================================="
        echo "STAGE 2: E2E — {size}"
        echo "=========================================="
        python -m omniaudio.train_v2 \\
            --config {cfg['e2e_config']} \\
            2>&1 | tee logs/cloudrift_{size}_e2e.log

        echo "=========================================="
        echo "DONE: {size}"
        echo "=========================================="
    """)


def cmd_monitor(args):
    """Tail training logs from all running instances."""
    state = load_state()

    if not state["instances"]:
        print("No tracked instances.")
        return

    ssh_key = _find_ssh_key()

    for size, info in state["instances"].items():
        ip = info.get("ip", "")
        if not ip:
            print(f"  {size}: no IP. Run 'status' first.")
            continue

        stage = info.get("stage", "ctc_training")
        log_suffix = "ctc" if "ctc" in stage else "e2e"

        print(f"\n{'='*50}")
        print(f"  {size} ({ip}) — {stage}")
        print(f"{'='*50}")

        r = _run_ssh(ip, f"tail -20 /root/slm/logs/cloudrift_{size}_{log_suffix}.log 2>/dev/null || echo 'No log yet'", ssh_key, timeout=15)
        print(r.stdout)

        # Check if screen is still alive
        r2 = _run_ssh(ip, f"screen -ls | grep train_{size} || true", ssh_key, timeout=10)
        if f"train_{size}" not in r2.stdout:
            print(f"  WARNING: Screen 'train_{size}' not found — training may have finished or crashed")


def cmd_terminate(args):
    """Terminate tracked instances (with confirmation)."""
    api_key = args.api_key
    state = load_state()

    if not state["instances"]:
        print("No tracked instances.")
        return

    print("\nInstances to terminate:")
    for size, info in state["instances"].items():
        print(f"  {size}: {info['instance_id']} (rented {info.get('rented_at', '?')})")

    confirm = input("\nType 'yes' to terminate ALL, or comma-separated sizes (e.g. '50m,150m'): ").strip()

    if confirm == "yes":
        to_terminate = list(state["instances"].keys())
    elif confirm:
        to_terminate = [s.strip() for s in confirm.split(",") if s.strip() in state["instances"]]
    else:
        print("Cancelled.")
        return

    for size in to_terminate:
        info = state["instances"][size]
        iid = info["instance_id"]
        print(f"  Terminating {size} ({iid})...")
        try:
            _api_call("/api/v1/instances/terminate",
                      {"selector": {"ById": [iid]}}, api_key)
            del state["instances"][size]
            save_state(state)
            print(f"  {size}: terminated")
        except Exception as e:
            print(f"  {size}: failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OmniAudio v2: Mass training on CloudRift GPUs"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CLOUDRIFT_API_KEY", ""),
        help="CloudRift API key (or set CLOUDRIFT_API_KEY env var)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available GPU instances")

    dryrun_p = sub.add_parser("dryrun", help="Dry run — show what would be rented without renting")
    dryrun_p.add_argument(
        "--sizes", nargs="+", choices=["50m", "150m", "600m", "1b"],
        default=["50m", "150m", "600m", "1b"],
    )

    launch_p = sub.add_parser("launch", help="Rent instances and prepare for training")
    launch_p.add_argument(
        "--sizes", nargs="+", choices=["50m", "150m", "600m", "1b"],
        default=["50m", "150m", "600m", "1b"],
        help="Model sizes to train (default: all)",
    )

    sub.add_parser("status", help="Check instance status")
    sub.add_parser("deploy", help="Deploy code and start training")
    sub.add_parser("monitor", help="Monitor training logs")
    sub.add_parser("terminate", help="Terminate instances")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set CLOUDRIFT_API_KEY")
        sys.exit(1)

    cmd_map = {
        "list": cmd_list,
        "dryrun": cmd_dryrun,
        "launch": cmd_launch,
        "status": cmd_status,
        "deploy": cmd_deploy,
        "monitor": cmd_monitor,
        "terminate": cmd_terminate,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
