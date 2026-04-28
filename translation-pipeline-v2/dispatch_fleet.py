#!/usr/bin/env python3
"""
Dispatch translation work to a fleet of RunPod instances.

Reads fleet_100bt.json (created by launch_fleet.py), deploys the pipeline
to each pod via SCP, and starts translation in screen sessions.

Usage:
    python dispatch_fleet.py                   # Deploy & start all pods
    python dispatch_fleet.py --status          # Check status of all pods
    python dispatch_fleet.py --rebalance       # Reassign chunks based on progress
    python dispatch_fleet.py --destroy         # Destroy all pods (asks confirmation)
"""

import argparse
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
FLEET_FILE = os.path.join(BASE, "fleet_100bt.json")

# Files to deploy to each pod
DEPLOY_FILES = [
    "config_100bt.py",
    "config.py",
    "filters.py",
    "sentence_splitter.py",
    "postprocessor.py",
    "translator.py",
    "pipeline_100bt.py",
    "setup_model.sh",
    "run_100bt.sh",
]


def load_fleet() -> dict:
    if not os.path.exists(FLEET_FILE):
        print(f"Fleet file not found: {FLEET_FILE}")
        print("Run launch_fleet.py first.")
        sys.exit(1)
    with open(FLEET_FILE) as f:
        return json.load(f)


def ssh_cmd(pod: dict) -> list[str]:
    return [
        "ssh", f"root@{pod['ssh_ip']}", "-p", str(pod['ssh_port']),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
    ]


def scp_cmd(pod: dict, local: str, remote: str) -> list[str]:
    return [
        "scp", "-P", str(pod['ssh_port']),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
        local, f"root@{pod['ssh_ip']}:{remote}",
    ]


def run_ssh(pod: dict, command: str, timeout: int = 120) -> tuple[int, str]:
    """Run command on pod via SSH. Returns (returncode, output)."""
    cmd = ssh_cmd(pod) + [command]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, str(e)


def deploy_to_pod(pod: dict) -> bool:
    """SCP pipeline files and setup model on a pod."""
    label = pod["label"]
    print(f"\n[{label}] Deploying pipeline...")

    # Create work directory
    rc, out = run_ssh(pod, "mkdir -p /workspace/translate && echo OK")
    if "OK" not in out:
        print(f"  [{label}] Failed to create dir: {out}")
        return False

    # SCP files
    for fname in DEPLOY_FILES:
        local_path = os.path.join(BASE, fname)
        if not os.path.exists(local_path):
            print(f"  [{label}] WARN: {fname} not found locally, skipping")
            continue
        cmd = scp_cmd(pod, local_path, f"/workspace/translate/{fname}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"  [{label}] SCP failed for {fname}: {r.stderr.strip()}")
            return False

    print(f"  [{label}] Files deployed.")

    # Install deps
    print(f"  [{label}] Installing dependencies...")
    rc, out = run_ssh(pod,
        "cd /workspace/translate && "
        "pip install -q ctranslate2 sentencepiece datasets huggingface_hub xxhash pyarrow tqdm 2>&1 | tail -3",
        timeout=300)
    if rc != 0:
        print(f"  [{label}] pip install failed: {out}")
        return False
    print(f"  [{label}] Dependencies installed.")

    # Setup model
    print(f"  [{label}] Setting up translation model...")
    rc, out = run_ssh(pod,
        "cd /workspace/translate && bash setup_model.sh 2>&1 | tail -5",
        timeout=300)
    if rc != 0:
        print(f"  [{label}] Model setup failed: {out}")
        return False
    print(f"  [{label}] Model ready.")

    # Verify HF auth
    print(f"  [{label}] Checking HF auth...")
    rc, out = run_ssh(pod,
        'python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()[\'name\'])"',
        timeout=30)
    if rc != 0:
        print(f"  [{label}] HF auth FAILED: {out}")
        print(f"  [{label}] You need to set HF_TOKEN on this pod!")
        return False
    print(f"  [{label}] HF auth OK: {out}")

    # Smoke test
    print(f"  [{label}] Running smoke test...")
    rc, out = run_ssh(pod,
        "cd /workspace/translate && python3 pipeline_100bt.py --smoke-test 2>&1 | tail -10",
        timeout=300)
    if "ALL SMOKE TESTS PASSED" in out:
        print(f"  [{label}] Smoke test PASSED")
    else:
        print(f"  [{label}] Smoke test FAILED:")
        print(out)
        return False

    return True


def start_translation(pod: dict) -> bool:
    """Start translation in a screen session on the pod."""
    label = pod["label"]
    start = pod["start_chunk"]
    end = pod["end_chunk"]
    gpus = pod["gpu_count"]

    screen_name = f"translate_{start}_{end}"
    cmd = (
        f"cd /workspace/translate && "
        f"screen -dmS {screen_name} bash -c '"
        f"python3 pipeline_100bt.py "
        f"--num-gpus {gpus} --start-chunk {start} --end-chunk {end} "
        f"2>&1 | tee translation_{start}_{end}.log"
        f"'"
    )

    rc, out = run_ssh(pod, cmd)
    if rc != 0:
        print(f"  [{label}] Failed to start: {out}")
        return False

    # Verify screen is running
    time.sleep(2)
    rc, out = run_ssh(pod, f"screen -ls | grep {screen_name}")
    if screen_name in out:
        print(f"  [{label}] Translation started in screen '{screen_name}'")
        print(f"    Chunks: {start}–{end-1} ({end-start} chunks)")
        print(f"    Monitor: {' '.join(ssh_cmd(pod))} 'tail -f /workspace/translate/translation_{start}_{end}.log'")
        return True
    else:
        print(f"  [{label}] Screen not found, may have crashed immediately")
        return False


def check_status(fleet: dict):
    """Check progress on all pods."""
    print("=" * 70)
    print("FLEET STATUS")
    print("=" * 70)

    for pod in fleet["pods"]:
        label = pod["label"]
        print(f"\n[{label}] pod={pod['pod_id']}, chunks {pod['start_chunk']}–{pod['end_chunk']-1}")

        # Check if pod is alive
        rc, out = run_ssh(pod, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1", timeout=15)
        if rc != 0:
            print(f"  STATUS: UNREACHABLE")
            continue

        # Check screen
        rc, out = run_ssh(pod, "screen -ls 2>/dev/null | grep translate || echo 'NO_SCREEN'", timeout=15)
        if "NO_SCREEN" in out:
            print(f"  STATUS: No translation screen running")
        else:
            print(f"  STATUS: Running — {out.strip()}")

        # Check progress
        rc, out = run_ssh(pod,
            "cd /workspace/translate && "
            "python3 -c \""
            "import json,os; "
            "f='progress_100bt.json'; "
            "p=json.load(open(f)) if os.path.exists(f) else {}; "
            "done=len(p.get('chunks_completed',[])); "
            "verified=len(p.get('chunks_verified',[])); "
            "rows=p.get('total_rows_translated',0); "
            "print(f'Chunks done: {done}, verified: {verified}, rows: {rows:,}')\" 2>/dev/null",
            timeout=15)
        if rc == 0 and out:
            print(f"  PROGRESS: {out}")

        # Last log line
        rc, out = run_ssh(pod,
            f"tail -1 /workspace/translate/translation_{pod['start_chunk']}_{pod['end_chunk']}.log 2>/dev/null || echo 'no log'",
            timeout=15)
        if rc == 0:
            print(f"  LAST LOG: {out[:120]}")


def destroy_fleet(fleet: dict):
    """Destroy all pods after confirmation."""
    import runpod as rp
    rp.api_key = json.load(open(os.path.expanduser("~/.runpod/config.json")))["api_key"]

    print("\nPods to destroy:")
    for pod in fleet["pods"]:
        print(f"  {pod['label']}: pod {pod['pod_id']} (chunks {pod['start_chunk']}–{pod['end_chunk']-1})")

    answer = input("\nType 'DESTROY' to confirm: ")
    if answer != "DESTROY":
        print("Aborted.")
        return

    for pod in fleet["pods"]:
        try:
            rp.terminate_pod(pod["pod_id"])
            print(f"  Terminated: {pod['label']} ({pod['pod_id']})")
        except Exception as e:
            print(f"  Failed to terminate {pod['pod_id']}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Dispatch translation to RunPod fleet")
    parser.add_argument("--status", action="store_true", help="Check fleet status")
    parser.add_argument("--destroy", action="store_true", help="Destroy all pods")
    parser.add_argument("--deploy-only", action="store_true", help="Deploy without starting")
    args = parser.parse_args()

    fleet = load_fleet()
    pods = fleet["pods"]

    if args.status:
        check_status(fleet)
        return

    if args.destroy:
        destroy_fleet(fleet)
        return

    # Deploy and start
    print("=" * 60)
    print(f"DISPATCHING TO {len(pods)} PODS")
    print("=" * 60)

    ready = []
    failed = []

    for pod in pods:
        ok = deploy_to_pod(pod)
        if ok:
            ready.append(pod)
        else:
            failed.append(pod)

    if failed:
        print(f"\n[WARN] {len(failed)} pods failed deployment: {[p['label'] for p in failed]}")
        if not ready:
            print("FATAL: No pods ready!")
            sys.exit(1)

        # Rebalance chunks across ready pods
        print("Rebalancing chunks across ready pods...")
        total_chunks = fleet["total_chunks"]
        total_speed = sum(p["total_speed"] for p in ready)
        cursor = 0
        for i, pod in enumerate(ready):
            fraction = pod["total_speed"] / total_speed
            num = max(1, round(fraction * total_chunks))
            if i == len(ready) - 1:
                num = total_chunks - cursor
            pod["start_chunk"] = cursor
            pod["end_chunk"] = min(cursor + num, total_chunks)
            cursor = pod["end_chunk"]
            print(f"  {pod['label']}: chunks {pod['start_chunk']}–{pod['end_chunk']-1}")

        # Update fleet file
        fleet["pods"] = ready
        with open(FLEET_FILE, "w") as f:
            json.dump(fleet, f, indent=2)

    if args.deploy_only:
        print("\n--deploy-only: skipping start. Run without flag to start translation.")
        return

    # Start translation on all ready pods
    print()
    print("=" * 60)
    print("STARTING TRANSLATION")
    print("=" * 60)

    for pod in ready:
        start_translation(pod)

    # Summary
    print()
    print("=" * 60)
    print("ALL PODS DISPATCHED")
    print("=" * 60)
    print()
    print("Monitor commands:")
    for pod in ready:
        s, e = pod["start_chunk"], pod["end_chunk"]
        print(f"  {pod['label']}: {' '.join(ssh_cmd(pod))} 'tail -f /workspace/translate/translation_{s}_{e}.log'")
    print()
    print(f"Fleet status:  python dispatch_fleet.py --status")
    print(f"Destroy fleet: python dispatch_fleet.py --destroy")


if __name__ == "__main__":
    main()
