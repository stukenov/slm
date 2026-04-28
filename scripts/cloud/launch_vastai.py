#!/usr/bin/env python3
"""Launch OmniAudio v2 training on vast.ai (2× RTX 5090).

Usage:
    python scripts/cloud/launch_vastai.py [--dry-run] [--gpu RTX_5090] [--num-gpus 2]

Requires: vastai CLI installed and configured with API key.
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(cmd, check=True, capture=False):
    """Run shell command."""
    print(f"  $ {cmd}")
    if capture:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return r.stdout.strip()
    subprocess.run(cmd, shell=True, check=check)


def find_offer(gpu_name, num_gpus):
    """Find cheapest matching vast.ai offer."""
    query = f"gpu_name={gpu_name} num_gpus={num_gpus} reliability>0.95 inet_down>200 disk_space>=100"
    out = run(f"vastai search offers '{query}' -o 'dph+' --limit 5 --raw", capture=True)
    offers = json.loads(out)
    if not offers:
        print(f"No offers found for {num_gpus}× {gpu_name}")
        sys.exit(1)
    best = offers[0]
    print(f"\nBest offer: {num_gpus}× {gpu_name}")
    print(f"  ID: {best['id']}")
    print(f"  Price: ${best['dph_total']:.4f}/hr")
    print(f"  Location: {best.get('geolocation', 'unknown')}")
    print(f"  VRAM: {best.get('gpu_ram', '?')}MB per GPU")
    print(f"  Disk: {best.get('disk_space', '?')}GB")
    print(f"  Reliability: {best.get('reliability', '?')}%")
    return best


def create_instance(offer_id, disk_gb=100):
    """Create vast.ai instance."""
    image = "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
    hf_token = os.environ.get("HF_TOKEN", "")
    out = run(
        f"vastai create instance {offer_id} "
        f"--image '{image}' "
        f"--disk {disk_gb} "
        f"--env 'HF_TOKEN={hf_token}' "
        f"--onstart-cmd 'mkdir -p /workspace' "
        f"--raw",
        capture=True,
    )
    data = json.loads(out)
    instance_id = data.get("new_contract")
    if not instance_id:
        print(f"Failed to create instance: {data}")
        sys.exit(1)
    print(f"\nInstance created: {instance_id}")
    return instance_id


def wait_for_ssh(instance_id, timeout=600):
    """Wait for instance to be ready and return SSH command."""
    print(f"\nWaiting for instance {instance_id} to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        out = run(f"vastai show instance {instance_id} --raw", capture=True)
        info = json.loads(out)
        status = info.get("actual_status", "unknown")
        ssh_host = info.get("ssh_host")
        ssh_port = info.get("ssh_port")
        print(f"  Status: {status} (elapsed: {int(time.time() - start)}s)")
        if status == "running" and ssh_host and ssh_port:
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no -p {ssh_port} root@{ssh_host}"
            print(f"\nSSH ready: {ssh_cmd}")
            return ssh_cmd, ssh_host, ssh_port
        time.sleep(15)
    print("Timeout waiting for SSH")
    sys.exit(1)


def deploy(ssh_cmd, ssh_host, ssh_port):
    """Upload project code and checkpoint to instance."""
    rsync_ssh = f"ssh -o StrictHostKeyChecking=no -p {ssh_port}"

    print("\n=== Deploying code ===")
    # Create remote directory structure first
    run(f'{ssh_cmd} "mkdir -p /workspace/slm/omniaudio /workspace/slm/scripts/cloud '
        f'/workspace/slm/checkpoints /workspace/slm/tokenizers /workspace/slm/outputs /workspace/slm/logs"')

    # Sync omniaudio package, configs, scripts, tokenizer
    for path in [
        "omniaudio/",
        "scripts/cloud/",
        "checkpoints/",
        "tokenizers/kazakh-gpt2-50k/",
    ]:
        src = os.path.join(PROJECT_DIR, path)
        if os.path.exists(src):
            dest = f"root@{ssh_host}:/workspace/slm/{path}"
            run(f'rsync -avz --progress -e "{rsync_ssh}" {src} {dest}')

    # Create needed directories
    run(f'{ssh_cmd} "mkdir -p /workspace/slm/outputs /workspace/slm/logs"')

    # Create checkpoint dir for init_from (train_v2 expects dir/model.pt)
    run(
        f'{ssh_cmd} "mkdir -p /workspace/slm/checkpoints/ctc_pretrain_best && '
        f'mv /workspace/slm/checkpoints/ctc_pretrain_best.pt '
        f'/workspace/slm/checkpoints/ctc_pretrain_best/model.pt 2>/dev/null; true"'
    )


def setup_env(ssh_cmd):
    """Run setup script on remote and login to HF."""
    print("\n=== Setting up environment ===")
    run(f'{ssh_cmd} "cd /workspace/slm && bash scripts/cloud/setup_remote.sh"')

    # Login to HF via python (env vars don't persist on vast.ai)
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        run(f"""{ssh_cmd} 'source /workspace/.venv/bin/activate && python3 -c "from huggingface_hub import login; login(token=\\"{hf_token}\\")"'""")
        print("HF login OK")


def start_training(ssh_cmd):
    """Start training in detached screen."""
    print("\n=== Starting training ===")
    run(
        f'{ssh_cmd} "cd /workspace/slm && screen -dmS train bash -c '
        f"'source /workspace/.venv/bin/activate && bash scripts/cloud/train_remote.sh; exec bash'\""
    )
    print("\nTraining started in screen 'train'")
    print(f"Monitor: {ssh_cmd} 'tail -f /workspace/slm/logs/ctc_cloud.log'")


def main():
    parser = argparse.ArgumentParser(description="Launch OmniAudio training on vast.ai")
    parser.add_argument("--dry-run", action="store_true", help="Find offer but don't create")
    parser.add_argument("--gpu", default="RTX_5090", help="GPU model (default: RTX_5090)")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs (default: 2)")
    parser.add_argument("--disk", type=int, default=100, help="Disk space in GB")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(hf_token_path):
            os.environ["HF_TOKEN"] = open(hf_token_path).read().strip()
        else:
            print("WARNING: HF_TOKEN not set. Gated dataset download will fail.")

    print(f"=== OmniAudio v2 Training on vast.ai ===")
    print(f"Config: {args.num_gpus}× {args.gpu}")

    # Find offer
    offer = find_offer(args.gpu, args.num_gpus)

    if args.dry_run:
        est_hours = 7.2  # 2×5090 estimate
        print(f"\nEstimated cost: ${offer['dph_total'] * est_hours:.2f} "
              f"({est_hours}h × ${offer['dph_total']:.2f}/hr)")
        print("(dry run, not creating instance)")
        return

    # Create instance
    instance_id = create_instance(offer["id"], args.disk)

    # Wait for SSH
    ssh_cmd, ssh_host, ssh_port = wait_for_ssh(instance_id)

    # Deploy
    time.sleep(10)  # let container fully init
    deploy(ssh_cmd, ssh_host, ssh_port)

    # Setup
    setup_env(ssh_cmd)

    # Start training
    start_training(ssh_cmd)

    print(f"\n{'='*60}")
    print(f"Instance ID: {instance_id}")
    print(f"SSH: {ssh_cmd}")
    print(f"Monitor CTC:  {ssh_cmd} 'tail -f /workspace/slm/logs/ctc_cloud.log'")
    print(f"Monitor E2E:  {ssh_cmd} 'tail -f /workspace/slm/logs/e2e_cloud.log'")
    print(f"Destroy when done: vastai destroy instance {instance_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
