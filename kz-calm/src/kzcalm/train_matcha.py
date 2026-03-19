"""DDP training loop for Matcha-TTS."""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from kzcalm.codec.mel import MelExtractor
from kzcalm.data.dataset import MelDataset, PreExtractedMelDataset, mel_collate_fn
from kzcalm.model.matcha import MatchaTTS
from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "inherits" in cfg:
        parent_name = cfg.pop("inherits")
        parent_path = Path(path).parent
        while parent_path != parent_path.parent:
            candidate = parent_path / f"{parent_name}.yaml"
            if candidate.exists():
                with open(candidate) as f:
                    base = yaml.safe_load(f)
                _deep_merge(base, cfg)
                return base
            parent_path = parent_path.parent
        raise FileNotFoundError(f"Base config '{parent_name}.yaml' not found")
    return cfg


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _get_lr(step: int, warmup: int, max_steps: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _download_tokenizer(hf_repo: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=hf_repo, filename="tokenizer.model")


def train(config: dict):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_cfg = config["training"]
    model_cfg = config["model"]
    codec_cfg = config["codec"]
    tok_cfg = config["tokenizer"]
    data_cfg = config["data"]

    # Dataset — pre-extracted shards or streaming
    mel_dir = data_cfg.get("mel_dir")
    if mel_dir:
        if is_main:
            logger.info(f"Loading pre-extracted mels from {mel_dir}")
        dataset = PreExtractedMelDataset(
            mel_dir=mel_dir,
            max_mel_frames=model_cfg.get("max_audio_frames", 3000),
        )
        sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            collate_fn=mel_collate_fn,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
        )
        if is_main:
            logger.info(f"Dataset: {len(dataset)} samples")
    else:
        # Tokenizer needed for streaming
        if tok_cfg.get("model_path"):
            tok_path = tok_cfg["model_path"]
        elif tok_cfg.get("hf_repo"):
            tok_path = _download_tokenizer(tok_cfg["hf_repo"])
        else:
            raise ValueError("tokenizer config missing")
        tokenizer = KazakhTokenizer(tok_path)

        mel_extractor = MelExtractor(
            sample_rate=config.get("sample_rate", 24000),
            n_mels=codec_cfg.get("n_mels", 100),
            n_fft=codec_cfg.get("n_fft", 1024),
            hop_length=codec_cfg.get("hop_length", 256),
        )
        streaming = data_cfg.get("streaming", True)
        dataset = MelDataset(
            hf_dataset=data_cfg["hf_audio_dataset"],
            tokenizer=tokenizer,
            mel_extractor=mel_extractor,
            split=data_cfg.get("split", "train"),
            max_text_len=model_cfg.get("max_text_len", 512),
            max_mel_frames=model_cfg.get("max_audio_frames", 3000),
            max_samples=data_cfg.get("max_samples", 0),
            streaming=streaming,
        )
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            collate_fn=mel_collate_fn,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=True,
            shuffle=not streaming,
        )

    # Model — get vocab_size
    if tok_cfg.get("type") == "char":
        from kzcalm.tokenizer.char_tokenizer import KazakhCharTokenizer
        vocab_size = KazakhCharTokenizer().vocab_size
    elif not mel_dir:
        vocab_size = tokenizer.vocab_size
    else:
        if tok_cfg.get("model_path"):
            tok_path = tok_cfg["model_path"]
        elif tok_cfg.get("hf_repo"):
            tok_path = _download_tokenizer(tok_cfg["hf_repo"])
        else:
            raise ValueError("tokenizer config missing")
        from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer as KT
        vocab_size = KT(tok_path).vocab_size

    model = MatchaTTS(
        vocab_size=vocab_size,
        mel_dim=codec_cfg.get("latent_dim", 100),
        encoder_dim=model_cfg.get("encoder_dim", 256),
        encoder_layers=model_cfg.get("encoder_layers", 4),
        encoder_heads=model_cfg.get("encoder_heads", 4),
        encoder_ff=model_cfg.get("encoder_ff", 1024),
        unet_channels=model_cfg.get("unet_channels", [256, 256, 512]),
        dropout=model_cfg.get("dropout", 0.1),
        max_text_len=model_cfg.get("max_text_len", 512),
    ).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"MatchaTTS: {n_params / 1e6:.1f}M params")

    # torch.compile
    if train_cfg.get("compile", False):
        if is_main:
            logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_ddp else model
    # unwrap compiled model if needed
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    # Optimizer
    max_lr = float(train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.98),
    )

    # Resume from checkpoint
    global_step = 0
    resume_path = config.get("resume_checkpoint")
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["step"]
        if is_main:
            logger.info(f"Resumed from {resume_path} at step {global_step}")

    # Output
    output_dir = Path(config.get("output_dir", "outputs")) / config["experiment_name"]
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(output_dir / "tb"))
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    # Training loop
    max_steps = train_cfg["max_steps"]
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    save_steps = train_cfg.get("save_steps", 5000)
    log_steps = train_cfg.get("logging_steps", 100)
    dur_weight = float(model_cfg.get("duration_loss_weight", 1.0))
    use_bf16 = train_cfg.get("bf16", True)

    model.train()
    optimizer.zero_grad()
    accum_flow = 0.0
    accum_dur = 0.0
    accum_prior = 0.0
    t0 = time.time()
    epoch = 0

    if is_main:
        logger.info(f"Training for {max_steps} steps, DDP={is_ddp}, world_size={world_size}")

    while global_step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1

        for batch in dataloader:
            if global_step >= max_steps:
                break

            lr = _get_lr(global_step, train_cfg.get("warmup_steps", 4000), max_steps, max_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            text = batch["text_ids"].to(device)
            text_mask_bool = batch["text_mask"].to(device)
            mel = batch["mel"].to(device)
            mel_mask = batch["mel_mask"].to(device)

            # text_mask in dataset is bool (True=pad), model expects float (1=valid)
            text_mask = (~text_mask_bool).float()

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                flow_loss, dur_loss, prior_loss = model(text, text_mask, mel, mel_mask)
                loss = (flow_loss + dur_weight * dur_loss + prior_loss) / accum_steps

            loss.backward()
            accum_flow += flow_loss.item() / accum_steps
            accum_dur += dur_loss.item() / accum_steps
            accum_prior += prior_loss.item() / accum_steps

            if (global_step + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if is_main and global_step % log_steps == 0:
                elapsed = time.time() - t0
                sps = (global_step + 1) / max(elapsed, 1)
                logger.info(
                    f"step={global_step} flow={accum_flow:.4f} dur={accum_dur:.4f} "
                    f"prior={accum_prior:.4f} lr={lr:.2e} speed={sps:.1f} steps/s"
                )
                writer.add_scalar("train/flow_loss", accum_flow, global_step)
                writer.add_scalar("train/dur_loss", accum_dur, global_step)
                writer.add_scalar("train/prior_loss", accum_prior, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                accum_flow = 0.0
                accum_dur = 0.0
                accum_prior = 0.0

            if is_main and global_step > 0 and global_step % save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "config": config,
                }, ckpt_dir / "model.pt")
                logger.info(f"Saved checkpoint: {ckpt_dir}")

            global_step += 1

    if is_main:
        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": global_step,
            "config": config,
        }, output_dir / "model_final.pt")
        writer.close()
        logger.info(f"Training complete. Output: {output_dir}")

    if is_ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
