"""Training loop for KZ-CALM TTS (flow-matching on continuous Mimi latents)."""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kzcalm.data.dataset import (
    LatentDataset, latent_collate_fn,
    CodesDataset, collate_fn,
    MelDataset, mel_collate_fn,
)
from kzcalm.codec.mel import MelExtractor
from kzcalm.model.backbone import TTSBackbone
from kzcalm.model.code_embedding import MimiLatentExtractor
from kzcalm.model.flow_head import FlowMatchingLoss
from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load YAML config with inheritance."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "inherits" in cfg:
        parent_name = cfg.pop("inherits")
        parent_path = Path(path).parent
        while parent_path != parent_path.parent:
            candidate = parent_path / f"{parent_name}.yaml"
            if candidate.exists():
                base = load_config(str(candidate))
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


def _get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _download_sp_tokenizer(hf_repo: str, cache_dir: str = "/tmp/kzcalm_tok") -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=hf_repo, filename="tokenizer.model", cache_dir=cache_dir)


def train(config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    tok_cfg = config["tokenizer"]

    # --- Tokenizer ---
    if tok_cfg.get("model_path"):
        tok_path = tok_cfg["model_path"]
    elif tok_cfg.get("hf_repo"):
        tok_path = _download_sp_tokenizer(tok_cfg["hf_repo"])
    else:
        raise ValueError("tokenizer.model_path or tokenizer.hf_repo required")
    tokenizer = KazakhTokenizer(tok_path)
    logger.info(f"Tokenizer vocab_size={tokenizer.vocab_size}")

    # --- Data ---
    codec_cfg = config["codec"]
    codec_type = codec_cfg.get("type", "mimi")
    latent_dim = codec_cfg["latent_dim"]
    latent_dir = data_cfg.get("latent_dir")

    if codec_type == "mel":
        # Mel spectrogram mode: stream audio, extract mel on the fly
        mel_extractor = MelExtractor(
            sample_rate=config.get("sample_rate", 24000),
            n_mels=codec_cfg.get("n_mels", 80),
            n_fft=codec_cfg.get("n_fft", 1024),
            hop_length=codec_cfg.get("hop_length", 256),
        )
        hf_audio_dataset = data_cfg.get("hf_audio_dataset")
        if not hf_audio_dataset:
            raise ValueError("data.hf_audio_dataset required for codec.type=mel")
        dataset = MelDataset(
            hf_dataset=hf_audio_dataset,
            tokenizer=tokenizer,
            mel_extractor=mel_extractor,
            split=data_cfg.get("split", "train"),
            max_text_len=model_cfg.get("max_text_len", 512),
            max_mel_frames=model_cfg.get("max_audio_frames", 3000),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            collate_fn=mel_collate_fn,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True,
        )
        use_preextracted = True  # mel is computed on the fly but acts like preextracted
        logger.info(f"Mel mode: streaming from {hf_audio_dataset}, latent_dim={latent_dim}")
    elif latent_dir:
        logger.info(f"Using pre-extracted latents from {latent_dir}")
        dataset = LatentDataset(
            latent_dir=latent_dir,
            tokenizer=tokenizer,
            max_text_len=model_cfg.get("max_text_len", 512),
            max_latent_frames=model_cfg.get("max_audio_frames", 1500) * 2,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            collate_fn=latent_collate_fn,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        use_preextracted = True
        logger.info(f"Loaded {len(dataset)} samples from latent shards")
    else:
        logger.info("No latent_dir — using online Mimi extraction (slow)")
        latent_extractor = MimiLatentExtractor(
            mimi_model_name=codec_cfg["name"], device=device
        )
        dataset = CodesDataset(
            hf_dataset=data_cfg["hf_codes_dataset"],
            tokenizer=tokenizer,
            split=data_cfg.get("split", "train"),
            max_text_len=model_cfg.get("max_text_len", 512),
            max_audio_frames=model_cfg.get("max_audio_frames", 1500),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            collate_fn=collate_fn,
            num_workers=train_cfg.get("num_workers", 2),
            pin_memory=True,
        )
        use_preextracted = False

    # --- Backbone model ---
    max_latent_frames = model_cfg.get("max_audio_frames", 1500) * 2
    model = TTSBackbone(
        vocab_size=tokenizer.vocab_size,
        latent_dim=latent_dim,
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg.get("dropout", 0.1),
        max_text_len=model_cfg.get("max_text_len", 512),
        max_audio_frames=max_latent_frames,
    ).to(device)

    # torch.compile for speed
    if train_cfg.get("compile", True) and device == "cuda":
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.1f}M")

    # --- Loss & Optimizer ---
    flow_cfg = config["flow"]
    criterion = FlowMatchingLoss(
        sigma_min=flow_cfg["sigma_min"],
        loss_type=flow_cfg["loss_type"],
    )

    max_lr = float(train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr,
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.98),
    )

    # --- Output ---
    output_dir = Path(config["output_dir"]) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # --- Training loop ---
    model.train()
    global_step = 0
    max_steps = train_cfg["max_steps"]
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    use_bf16 = train_cfg.get("bf16", True)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    save_steps = train_cfg.get("save_steps", 5000)
    log_steps = train_cfg.get("logging_steps", 100)

    optimizer.zero_grad()
    accum_loss = 0.0
    t0 = time.time()

    logger.info(f"Starting training for {max_steps} steps (accum={accum_steps}, preextracted={use_preextracted})")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            lr = _get_lr(global_step, train_cfg["warmup_steps"], max_steps, max_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)

            if codec_type == "mel":
                x1 = batch["mel"].to(device)            # (B, T, 80)
                latent_mask = batch["mel_mask"].to(device)  # (B, T)
            elif use_preextracted:
                x1 = batch["latents"].to(device)       # (B, 2T, 512)
                latent_mask = batch["latent_mask"].to(device)  # (B, 2T)
            else:
                codes = batch["codes"].to(device)
                codes_mask = batch["codes_mask"].to(device)
                with torch.no_grad():
                    x1 = latent_extractor(codes)
                latent_mask = codes_mask.repeat_interleave(2, dim=1)

            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=device)
            t_expand = t[:, None, None]
            x_t = (1 - t_expand) * x0 + t_expand * x1

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                velocity = model(
                    x_t, text_ids, t,
                    text_padding_mask=text_mask,
                    latent_padding_mask=(latent_mask == 0),
                )
                loss = criterion(velocity, x0, x1, mask=latent_mask)
                loss = loss / accum_steps

            loss.backward()
            accum_loss += loss.item()

            if (global_step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if global_step % log_steps == 0:
                elapsed = time.time() - t0
                steps_per_sec = (global_step + 1) / max(elapsed, 1)
                real_loss = accum_loss * accum_steps
                logger.info(
                    f"step={global_step} loss={real_loss:.4f} lr={lr:.2e} "
                    f"speed={steps_per_sec:.1f} steps/s"
                )
                writer.add_scalar("train/loss", real_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                accum_loss = 0.0

            if global_step > 0 and global_step % save_steps == 0:
                _save_checkpoint(model, optimizer, config, global_step, output_dir)

            global_step += 1

    _save_checkpoint(model, optimizer, config, global_step, output_dir, final=True)
    writer.close()
    logger.info(f"Training complete. Output: {output_dir}")


def _save_checkpoint(model, optimizer, config, step, output_dir, final=False):
    # Handle compiled model
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if final:
        ckpt_dir = output_dir
        name = "model_final.pt"
    else:
        ckpt_dir = output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(exist_ok=True)
        name = "model.pt"

    torch.save(
        {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
         "step": step, "config": config},
        ckpt_dir / name,
    )
    logger.info(f"Saved checkpoint: {ckpt_dir / name}")


def main():
    parser = argparse.ArgumentParser(description="Train KZ-CALM TTS")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
