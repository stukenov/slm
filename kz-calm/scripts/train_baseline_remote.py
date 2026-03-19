#!/usr/bin/env python3
"""Self-contained remote training script for vast.ai / cloud GPU.

Usage:
    python train_baseline_remote.py [--max_steps 50000] [--push_to_hub]
"""

import argparse
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def install_deps():
    """Install required packages."""
    packages = [
        "torch",
        "transformers",
        "datasets",
        "sentencepiece",
        "huggingface_hub",
        "tensorboard",
        "pyyaml",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)


def download_tokenizer(hf_repo: str, dest: str = "/tmp/kzcalm_tok") -> str:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=hf_repo, filename="tokenizer.model", cache_dir=dest)
    logger.info(f"Tokenizer downloaded: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo", default="stukenov/kzcalm-baseline-v1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Installing dependencies...")
    install_deps()

    import torch
    import yaml

    # Write config inline (no file dependency)
    config = {
        "sample_rate": 24000,
        "audio_channels": 1,
        "codec": {"name": "kyutai/mimi", "frozen": True, "latent_dim": 512, "frame_rate": 12.5},
        "tokenizer": {"type": "sentencepiece", "vocab_size": 4096, "hf_repo": "stukenov/kzcalm-sp-tokenizer-4k-kk-v1"},
        "model": {
            "num_layers": 10, "d_model": 512, "num_heads": 8, "d_ff": 2048,
            "dropout": 0.1, "max_text_len": 512, "max_audio_frames": 1500,
        },
        "flow": {"num_sampling_steps": 8, "sigma_min": 0.001, "loss_type": "huber"},
        "training": {
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "learning_rate": 3e-4,
            "warmup_steps": 2000,
            "max_steps": args.max_steps,
            "weight_decay": 0.01,
            "bf16": True,
            "compile": True,
            "grad_clip": 1.0,
            "num_workers": 4,
            "save_steps": 10000,
            "logging_steps": 100,
        },
        "data": {"hf_codes_dataset": "stukenov/kzcalm-mimi-codes-kk-v1", "latent_dir": "/workspace/latents_512", "split": "train"},
        "output_dir": "/workspace/outputs",
        "experiment_name": "exp001_baseline_v2",
    }

    # Ensure kzcalm is importable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from kzcalm.train import train
    logger.info(f"Training with config: max_steps={args.max_steps}, batch_size={args.batch_size}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train(config)

    if args.push_to_hub:
        logger.info(f"Pushing to HF Hub: {args.hub_repo}")
        from huggingface_hub import HfApi
        api = HfApi()
        output_dir = f"/workspace/outputs/{config['experiment_name']}"
        api.create_repo(args.hub_repo, exist_ok=True)
        api.upload_folder(folder_path=output_dir, repo_id=args.hub_repo)
        logger.info("Upload complete.")


if __name__ == "__main__":
    main()
