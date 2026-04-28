#!/usr/bin/env python3
"""Launch a RunPod pod for exp038 YouTube speech collection via runpodctl CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=capture)


def launch() -> None:
    gpu_types = [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A6000",
        "NVIDIA L40S",
        "NVIDIA A100 SXM",
    ]
    for gpu in gpu_types:
        print(f"Trying {gpu}...")
        try:
            result = run(
                [
                    "runpodctl", "create", "pod",
                    "--name", "exp038-youtube-kk-audio",
                    "--imageName", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                    "--gpuType", gpu,
                    "--gpuCount", "1",
                    "--containerDiskSize", "120",
                    "--volumeSize", "0",
                    "--ports", "22/tcp,8501/http",
                ],
                capture=True,
            )
            pod_id = ""
            for line in result.stdout.splitlines():
                print(line)
                if "pod" in line.lower():
                    # runpodctl prints something like "pod <id> created"
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.lower() == "pod" and i + 1 < len(parts):
                            pod_id = parts[i + 1]
            print(f"\nCreated on {gpu}")
            if pod_id:
                print(f"\nReview UI (after deploy):  https://{pod_id}-8501.proxy.runpod.net")
            print("\nNext steps:")
            print("  runpodctl get pod                          # check status + SSH port")
            print("  runpodctl ssh info <pod-id>                # get SSH connection details")
            print("  bash scripts/exp038/deploy_to_pod.sh <host> <port>")
            return
        except subprocess.CalledProcessError as exc:
            print(f"  unavailable: {exc.stderr[:120] if exc.stderr else ''}")
    raise SystemExit("No compatible GPU was available")


def status() -> None:
    run(["runpodctl", "get", "pod"])


def destroy(pod_id: str) -> None:
    run(["runpodctl", "remove", "pod", pod_id])
    print(f"Destroyed {pod_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod launcher for exp038")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("launch")
    sub.add_parser("status")
    p_destroy = sub.add_parser("destroy")
    p_destroy.add_argument("pod_id")
    args = parser.parse_args()

    if args.cmd == "launch":
        launch()
    elif args.cmd == "status":
        status()
    else:
        destroy(args.pod_id)


if __name__ == "__main__":
    main()
