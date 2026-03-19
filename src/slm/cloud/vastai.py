"""Wrapper around the vast.ai CLI (subprocess + JSON parsing)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

def _find_vastai_cli() -> str:
    """Find vastai CLI binary, checking venv and PATH."""
    # Check project-local cloud venv first
    local_venv = Path(__file__).resolve().parents[3] / ".venv-cloud" / "bin" / "vastai"
    if local_venv.exists():
        return str(local_venv)
    if shutil.which("vastai"):
        return "vastai"
    raise VastError("vastai CLI not found. Install with: pip install vastai")


class VastError(Exception):
    """Raised when a vast.ai CLI call fails."""


def _get_api_key() -> str:
    """Resolve vast.ai API key: env → config dir → ~/.vast_api_key → error."""
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key.strip()

    # vastai CLI stores key here by default
    config_key = Path.home() / ".config" / "vastai" / "vast_api_key"
    if config_key.exists():
        return config_key.read_text().strip()

    keyfile = Path.home() / ".vast_api_key"
    if keyfile.exists():
        return keyfile.read_text().strip()

    raise VastError(
        "vast.ai API key not found. Set VAST_API_KEY env var "
        "or run: vastai set api-key <KEY>"
    )


def _run(args: list[str], *, timeout: int = 60) -> str:
    """Run a vastai CLI command and return stdout."""
    cli = _find_vastai_cli()
    cmd = [cli] + args
    log.debug("Running: %s", " ".join(cmd))

    env = os.environ.copy()
    api_key = _get_api_key()
    env["VAST_API_KEY"] = api_key

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise VastError(f"vastai command timed out after {timeout}s: {cmd}") from exc

    if result.returncode != 0:
        raise VastError(
            f"vastai command failed (rc={result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result.stdout


def _parse_json(raw: str) -> list[dict] | dict:
    """Parse JSON from vastai output, tolerating leading non-JSON text."""
    text = raw.strip()
    if not text:
        return []
    # vastai sometimes prints status messages before JSON
    for i, ch in enumerate(text):
        if ch in ("[", "{"):
            text = text[i:]
            break
    else:
        return []
    return json.loads(text)


def search_offers(
    query: str = "rentable=true",
    order: str = "dph_total",
    offer_type: str = "bid",
) -> list[dict]:
    """Search available GPU offers."""
    raw = _run([
        "search", "offers", query,
        "-o", order,
        "--type", offer_type,
        "--raw",
    ], timeout=120)
    return _parse_json(raw)


def create_instance(
    offer_id: int,
    *,
    image: str = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
    disk: int = 60,
    onstart_cmd: str = "",
    env_vars: dict[str, str] | None = None,
    label: str = "",
) -> int:
    """Create an instance from an offer. Returns instance ID."""
    args = [
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk),
        "--raw",
    ]
    if onstart_cmd:
        args += ["--onstart-cmd", onstart_cmd]
    if label:
        args += ["--label", label]
    if env_vars:
        for k, v in env_vars.items():
            args += ["--env", f"{k}={v}"]

    raw = _run(args, timeout=120)
    data = _parse_json(raw)
    if isinstance(data, dict) and "new_contract" in data:
        return int(data["new_contract"])
    raise VastError(f"Unexpected create response: {raw[:500]}")


def show_instances() -> list[dict]:
    """List all current instances."""
    raw = _run(["show", "instances", "--raw"], timeout=120)
    return _parse_json(raw)


def show_instance(instance_id: int) -> dict:
    """Get details for a single instance."""
    raw = _run(["show", "instance", str(instance_id), "--raw"], timeout=120)
    data = _parse_json(raw)
    if isinstance(data, list):
        return data[0] if data else {}
    return data


def ssh_url(instance_id: int) -> tuple[str, int]:
    """Return (host, port) for SSH access to an instance."""
    info = show_instance(instance_id)
    host = info.get("ssh_host", "")
    port = int(info.get("ssh_port", 0))
    if not host or not port:
        raise VastError(f"SSH not available for instance {instance_id}")
    return host, port


def wait_for_instance(instance_id: int, *, timeout: int = 300) -> dict:
    """Poll until instance is running. Returns instance info."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = show_instance(instance_id)
        status = info.get("actual_status") or info.get("status_msg") or ""
        log.info("Instance %d status: %s", instance_id, status)
        if status == "running":
            return info
        if status and ("error" in status.lower() or "failed" in status.lower()):
            raise VastError(f"Instance {instance_id} failed: {status}")
        time.sleep(10)
    raise VastError(f"Instance {instance_id} did not start within {timeout}s")


def destroy_instance(instance_id: int) -> None:
    """Destroy (delete) an instance."""
    _run(["destroy", "instance", str(instance_id)], timeout=120)
    log.info("Destroyed instance %d", instance_id)
