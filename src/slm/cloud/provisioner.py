"""Orchestrate the full lifecycle: launch → provision → train → cleanup."""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import yaml

from . import gpu_selector, remote_script, vastai

log = logging.getLogger(__name__)


def _load_config(config_path: str | Path) -> dict:
    """Load YAML config with inheritance (lightweight, no torch dependency)."""
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    inherits = config.pop("inherits", None)
    if inherits:
        search_dir = config_path.parent
        while search_dir != search_dir.parent:
            candidate = search_dir / f"{inherits}.yaml"
            if candidate.exists():
                base_config = _load_config(candidate)
                base_config.update(config)
                return base_config
            search_dir = search_dir.parent
        raise FileNotFoundError(f"Could not find base config '{inherits}.yaml'")
    return config

# Docker image with CUDA 12.4 + PyTorch pre-installed
DEFAULT_IMAGE = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"

# Files/dirs excluded from the project tarball
TAR_EXCLUDES = {
    ".git", ".venv", ".venv-cloud", "venv", "outputs", "logs", "__pycache__",
    ".cache", "wandb", "node_modules", ".ruff_cache",
}

_active_instance: int | None = None


def _resolve_hf_token() -> str:
    """Find HuggingFace token from env or local cache."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_path.exists():
        token = cache_path.read_text().strip()
        if token:
            return token

    raise RuntimeError(
        "HuggingFace token not found. Set HF_TOKEN env var "
        "or log in with: huggingface-cli login"
    )


def _atexit_handler():
    """Warn about orphan instances on unclean exit."""
    if _active_instance is not None:
        print(
            f"\n⚠ WARNING: Instance {_active_instance} may still be running!\n"
            f"  Check:   python -m slm.cloud status\n"
            f"  Destroy: python -m slm.cloud destroy --instance-id {_active_instance}\n",
            file=sys.stderr,
        )


atexit.register(_atexit_handler)


def _create_project_tarball(project_dir: Path) -> Path:
    """Create a tarball of the project, excluding heavy dirs."""
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        for part in parts:
            if part in TAR_EXCLUDES:
                return None
        # Skip large binary files
        if info.size > 100 * 1024 * 1024:  # >100MB
            return None
        return info

    with tarfile.open(tmp.name, "w:gz") as tar:
        tar.add(str(project_dir), arcname="slm", filter=_filter)

    size_mb = os.path.getsize(tmp.name) / 1024 / 1024
    log.info("Created tarball: %.1f MB", size_mb)
    return Path(tmp.name)


def _wait_for_ssh(host: str, port: int, *, retries: int = 60, delay: int = 10) -> None:
    """Poll SSH until it's reachable."""
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    "-o", "ConnectTimeout=5",
                    "-o", "BatchMode=yes",
                    "-p", str(port),
                    f"root@{host}",
                    "echo ok",
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                log.info("SSH ready (attempt %d/%d)", attempt, retries)
                return
        except subprocess.TimeoutExpired:
            pass
        log.info("SSH not ready, retry %d/%d in %ds...", attempt, retries, delay)
        time.sleep(delay)

    raise RuntimeError(
        f"SSH not reachable at {host}:{port} after {retries * delay}s"
    )


def _ssh_run(host: str, port: int, cmd: str, *, timeout: int = 600) -> str:
    """Run a command over SSH and return stdout."""
    result = subprocess.run(
        [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(port),
            f"root@{host}",
            cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (rc={result.returncode}): {cmd}\n"
            f"stderr: {result.stderr[:500]}"
        )
    return result.stdout


def _scp_to(local_path: Path, host: str, port: int, remote_path: str) -> None:
    """Copy a local file to the remote instance."""
    result = subprocess.run(
        [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-P", str(port),
            str(local_path),
            f"root@{host}:{remote_path}",
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"scp failed: {result.stderr[:500]}")


def launch(
    config_path: str,
    hf_repo: str,
    *,
    gpu: str | None = None,
    max_price: float = 0.50,
    num_gpus: int = 1,
    disk: int = 60,
    do_monitor: bool = False,
    dry_run: bool = False,
    extra_train_args: str = "",
    train_module: str = "slm.train",
    pre_train_cmd: str = "",
) -> int | None:
    """Full launch pipeline: select GPU → create instance → upload code → start training.

    Returns the instance ID, or None if dry_run.
    """
    global _active_instance

    # 1. Load config
    config = _load_config(config_path)
    experiment_name = config.get("experiment_name", Path(config_path).stem)
    log.info("Experiment: %s", experiment_name)

    # 2. Resolve HF token
    hf_token = _resolve_hf_token()
    log.info("HF token resolved (length=%d)", len(hf_token))

    # 3. Select best GPU offer
    best = gpu_selector.select_best_offer(
        config,
        max_price=max_price,
        num_gpus=num_gpus,
        min_disk=disk,
        gpu_override=gpu,
        dry_run=dry_run,
    )

    if dry_run:
        print(f"Config: {config_path}")
        print(f"Experiment: {experiment_name}")
        print(f"HF repo: {hf_repo}")
        print(f"Best GPU: {best['gpu_name']} @ ${best['dph']:.3f}/hr")
        print(f"Estimated: {best['estimated_hours']:.1f}h, ${best['estimated_cost']:.2f}")
        return None

    offer_id = int(best["offer"]["id"])
    actual_num_gpus = int(best["offer"].get("num_gpus", num_gpus))

    # 4. Generate onstart script
    onstart = remote_script.generate_onstart_script(hf_token)

    # 5. Create instance
    print(f"Creating instance: {best['gpu_name']} @ ${best['dph']:.3f}/hr...")
    env_vars = {"HF_TOKEN": hf_token}
    instance_id = vastai.create_instance(
        offer_id,
        image=DEFAULT_IMAGE,
        disk=disk,
        onstart_cmd=onstart,
        env_vars=env_vars,
        label=f"slm-{experiment_name}",
    )
    _active_instance = instance_id
    print(f"Instance created: {instance_id}")

    # 6. Wait for instance to be running
    try:
        print("Waiting for instance to start...")
        vastai.wait_for_instance(instance_id, timeout=600)
    except (vastai.VastError, RuntimeError):
        print(f"Instance {instance_id} failed to start. Destroying...")
        vastai.destroy_instance(instance_id)
        _active_instance = None
        raise

    # 7. Wait for SSH access
    try:
        host, port = vastai.ssh_url(instance_id)
        print(f"SSH: root@{host} -p {port}")
        _wait_for_ssh(host, port)
    except RuntimeError:
        print(f"SSH unreachable. Destroying instance {instance_id}...")
        vastai.destroy_instance(instance_id)
        _active_instance = None
        raise

    # 8. Create and upload project tarball
    project_dir = Path(config_path).resolve().parent
    # Walk up to find project root (where pyproject.toml is)
    while project_dir != project_dir.parent:
        if (project_dir / "pyproject.toml").exists():
            break
        project_dir = project_dir.parent
    else:
        project_dir = Path.cwd()

    print("Packing project...")
    tarball = _create_project_tarball(project_dir)
    try:
        print("Uploading project to instance...")
        _scp_to(tarball, host, port, "/root/project.tar.gz")
    finally:
        tarball.unlink(missing_ok=True)

    # 9. Extract and install on remote
    print("Setting up remote environment...")
    _ssh_run(host, port, (
        "cd /root && tar xzf project.tar.gz && "
        "cd /root/slm && "
        "pip install -q -e '.' 2>&1 | tail -5"
    ), timeout=300)

    # 10. Generate and upload run script
    vast_api_key = vastai._get_api_key()
    run_script = remote_script.generate_run_script(
        config_path=f"/root/slm/{os.path.relpath(config_path, project_dir)}",
        hf_repo=hf_repo,
        experiment_name=experiment_name,
        num_gpus=actual_num_gpus,
        extra_train_args=extra_train_args,
        instance_id=instance_id,
        vast_api_key=vast_api_key,
        train_module=train_module,
        pre_train_cmd=pre_train_cmd,
    )
    # Upload via stdin to avoid quoting issues
    proc = subprocess.run(
        [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(port),
            f"root@{host}",
            "cat > /root/slm/run_cloud.sh && chmod +x /root/slm/run_cloud.sh",
        ],
        input=run_script, capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to upload run script: {proc.stderr[:500]}")

    # 11. Start training (detached via nohup)
    print("Starting training...")
    cmd = (
        f"mkdir -p /root/slm/logs && cd /root/slm && "
        f"nohup bash run_cloud.sh "
        f"> /root/slm/logs/cloud_{experiment_name}.log 2>&1 &"
    )
    # Use ssh -f to background immediately
    subprocess.run(
        [
            "ssh", "-f",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(port),
            f"root@{host}",
            cmd,
        ],
        capture_output=True, text=True, timeout=30,
    )
    time.sleep(3)

    print(f"\nTraining launched on instance {instance_id}!")
    print(f"  GPU:    {best['gpu_name']}")
    print(f"  Cost:   ${best['dph']:.3f}/hr (est. ${best['estimated_cost']:.2f} total)")
    print(f"  ETA:    ~{best['estimated_hours']:.1f}h")
    print(f"  HF:     {hf_repo}")
    print(f"\nMonitor: python -m slm.cloud monitor --instance-id {instance_id}")
    print(f"Destroy: python -m slm.cloud destroy --instance-id {instance_id}")

    # 12. Optionally tail the log
    if do_monitor:
        _active_instance = None  # don't warn on clean exit
        monitor(instance_id)
    else:
        _active_instance = None

    return instance_id


def monitor(instance_id: int) -> None:
    """Tail the training log on a running instance."""
    host, port = vastai.ssh_url(instance_id)
    print(f"Connecting to instance {instance_id} ({host}:{port})...")
    print("Press Ctrl+C to detach (instance keeps running)\n")

    try:
        proc = subprocess.Popen(
            [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-p", str(port),
                f"root@{host}",
                "tail -f /root/slm/logs/cloud_*.log",
            ],
        )
        proc.wait()
    except KeyboardInterrupt:
        print("\nDetached. Instance still running.")


def status() -> None:
    """Show all active vast.ai instances."""
    instances = vastai.show_instances()
    if not instances:
        print("No active instances.")
        return

    print(f"{'ID':>8}  {'Status':>10}  {'GPU':>15}  {'$/hr':>7}  {'Label'}")
    print("-" * 65)
    for inst in instances:
        print(
            f"{inst.get('id', '?'):>8}  "
            f"{inst.get('actual_status', '?'):>10}  "
            f"{inst.get('gpu_name', '?'):>15}  "
            f"${float(inst.get('dph_total', 0)):>6.3f}  "
            f"{inst.get('label', '')}"
        )


def destroy(instance_id: int) -> None:
    """Destroy an instance by ID."""
    vastai.destroy_instance(instance_id)
    print(f"Instance {instance_id} destroyed.")
