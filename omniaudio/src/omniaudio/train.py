"""Training script for OmniAudio model."""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from omniaudio.data import AudioCollator, load_commonvoice_kk
from omniaudio.model import OmniAudioModel

logger = logging.getLogger(__name__)


def load_config(path):
    """Load YAML config with inheritance."""
    with open(path) as f:
        config = yaml.safe_load(f)
    inherits = config.pop("inherits", None)
    if inherits:
        base_path = Path(path).parent / f"{inherits}.yaml"
        base = load_config(base_path)
        base.update(config)
        return base
    return config


def train(config):
    experiment_name = config["experiment_name"]
    output_dir = Path(config["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build model
    model = OmniAudioModel(
        n_mels=config["n_mels"],
        audio_d_model=config["audio_d_model"],
        audio_n_heads=config["audio_n_heads"],
        audio_n_layers=config["audio_n_layers"],
        llm_vocab_size=config["llm_vocab_size"],
        llm_d_model=config["llm_d_model"],
        llm_n_heads=config["llm_n_heads"],
        llm_n_layers=config["llm_n_layers"],
        llm_intermediate_size=config["llm_intermediate_size"],
    )

    # Load from previous stage checkpoint
    if config.get("init_from_stage1"):
        ckpt = Path(config["init_from_stage1"]) / "model.pt"
        logger.info("Loading from stage1: %s", ckpt)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    # Freeze components based on stage
    if config.get("freeze_audio_encoder"):
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
    if config.get("freeze_llm"):
        for name, p in model.named_parameters():
            if any(k in name for k in ("text_embed", "layers.", "norm.", "lm_head")):
                p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %.2fM total, %.2fM trainable", total / 1e6, trainable / 1e6)

    model = model.to(device)
    use_bf16 = config.get("bf16", True) and device.type == "cuda"

    # Data
    collator = AudioCollator(
        tokenizer_path=config["tokenizer_path"],
        n_mels=config["n_mels"],
        sample_rate=config["sample_rate"],
        max_audio_len=config["max_audio_len"],
        max_text_len=config["max_text_len"],
    )

    max_train = config.get("max_train_samples")
    max_eval = config.get("max_eval_samples")
    train_ds = load_commonvoice_kk("train", max_samples=max_train)
    val_ds = load_commonvoice_kk("validation", max_samples=max_eval)
    logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))

    batch_size = config["per_device_train_batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=config.get("dataloader_num_workers", 4),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, collate_fn=collator,
        num_workers=config.get("dataloader_num_workers", 4),
    )

    # Optimizer + scheduler
    grad_accum = config.get("gradient_accumulation_steps", 1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    num_epochs = config["num_train_epochs"]
    max_steps = config.get("max_steps", -1)
    total_steps = len(train_loader) * num_epochs // grad_accum
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.05)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            mel = batch["mel"].to(device)
            text_ids = batch["text_ids"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(mel, text_ids)
                loss = loss / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
            num_batches += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.get("max_grad_norm", 1.0))
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.get("logging_steps", 50) == 0:
                    avg = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    logger.info("Step %d | Loss: %.4f | LR: %.2e", global_step, avg, lr)

                if config.get("save_steps") and global_step % config["save_steps"] == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), ckpt_dir / "model.pt")
                    logger.info("Saved checkpoint-%d", global_step)

                if max_steps > 0 and global_step >= max_steps:
                    break

            if max_steps > 0 and global_step >= max_steps:
                break

        avg_train = epoch_loss / max(num_batches, 1)
        logger.info("Epoch %d | Train loss: %.4f", epoch + 1, avg_train)

        # Validation
        if config.get("eval_steps"):  # run eval at end of epoch too
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    mel = batch["mel"].to(device)
                    text_ids = batch["text_ids"].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        loss = model(mel, text_ids)
                    val_loss += loss.item()
                    val_batches += 1
            avg_val = val_loss / max(val_batches, 1)
            logger.info("Epoch %d | Val loss: %.4f", epoch + 1, avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), best_dir / "model.pt")
                logger.info("New best model (val_loss=%.4f)", avg_val)

        if max_steps > 0 and global_step >= max_steps:
            break

    # Save final
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pt")
    logger.info("Training complete: %s", experiment_name)


def main():
    parser = argparse.ArgumentParser(description="OmniAudio Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
        if args.max_steps <= 50:
            config["eval_steps"] = None
            config["save_steps"] = None

    train(config)


if __name__ == "__main__":
    main()
