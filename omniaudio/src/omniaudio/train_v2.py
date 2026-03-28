"""Training script for OmniAudio v2: 3-stage (CTC pretrain, alignment, E2E)."""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from omniaudio.data_v2 import AudioCollatorV2, load_speech_dataset
from omniaudio.model_v2 import OmniAudioV2Model

logger = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    inherits = config.pop("inherits", None)
    if inherits:
        base_path = Path(path).parent / f"{inherits}.yaml"
        base = load_config(base_path)
        base.update(config)
        return base
    return config


def get_trainable_params(model, config):
    stage = config["stage"]
    for p in model.parameters():
        p.requires_grad = False

    if stage == "ctc_pretrain":
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.ctc_head.parameters():
            p.requires_grad = True
    elif stage == "alignment":
        for p in model.projector.parameters():
            p.requires_grad = True
    elif stage == "e2e":
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.projector.parameters():
            p.requires_grad = True
        for p in model.ctc_head.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Stage: %s | Params: %.2fM total, %.2fM trainable", stage, total / 1e6, trainable / 1e6)


def train(config):
    experiment_name = config["experiment_name"]
    output_dir = Path(config["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = config["stage"]
    logger.info("Device: %s | Stage: %s", device, stage)

    encoder_config = {
        "n_mels": config["n_mels"],
        "d_model": config["audio_d_model"],
        "n_heads": config["audio_n_heads"],
        "n_layers": config["audio_n_layers"],
        "n_conv": config["audio_n_conv"],
    }

    llm_name = config.get("llm_name") if stage != "ctc_pretrain" else None
    model = OmniAudioV2Model(
        encoder_config=encoder_config, llm_name=llm_name,
        vocab_size=config["vocab_size"], llm_dim=config.get("llm_dim", 768),
    )

    init_from = config.get("init_from")
    if init_from:
        ckpt = Path(init_from) / "model.pt"
        logger.info("Loading checkpoint: %s", ckpt)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("Loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))

    get_trainable_params(model, config)
    model = model.to(device)
    use_bf16 = config.get("bf16", True) and device.type == "cuda"

    augment = config.get("augment", stage != "ctc_pretrain")
    collator = AudioCollatorV2(
        tokenizer_path=config["tokenizer_path"], n_mels=config["n_mels"],
        sample_rate=config["sample_rate"], max_audio_len=config["max_audio_len"],
        max_text_len=config["max_text_len"], augment=augment,
    )
    dataset_name = config.get("dataset_name", "fleurs")
    train_ds = load_speech_dataset(dataset_name, "train", max_samples=config.get("max_train_samples"))
    val_ds = load_speech_dataset(dataset_name, "validation", max_samples=config.get("max_eval_samples"))
    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    batch_size = config["per_device_train_batch_size"]
    num_workers = config.get("dataloader_num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collator,
                            num_workers=num_workers)

    grad_accum = config.get("gradient_accumulation_steps", 1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    num_epochs = config["num_train_epochs"]
    total_steps = len(train_loader) * num_epochs // grad_accum
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.05)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ctc_weight = float(config.get("ctc_weight", 0.0))

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            mel = batch["mel"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if stage == "ctc_pretrain":
                    loss = model.forward_ctc(mel, batch["ctc_targets"].to(device),
                                             batch["ctc_target_lengths"].to(device))
                else:
                    text_ids = batch["text_ids"].to(device)
                    ctc_t = batch["ctc_targets"].to(device) if ctc_weight > 0 else None
                    ctc_l = batch["ctc_target_lengths"].to(device) if ctc_weight > 0 else None
                    loss = model.forward_e2e(mel, text_ids, ctc_weight=ctc_weight,
                                            ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
                loss = loss / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
            num_batches += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.get("logging_steps", 50) == 0:
                    avg = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    logger.info("Step %d | Loss: %.4f | LR: %.2e", global_step, avg, lr)

                save_steps = config.get("save_steps")
                if save_steps and global_step % save_steps == 0:
                    _save_checkpoint(model, output_dir, global_step)

        avg_train = epoch_loss / max(num_batches, 1)
        logger.info("Epoch %d/%d | Train loss: %.4f", epoch + 1, num_epochs, avg_train)

        val_loss = _run_validation(model, val_loader, device, stage, use_bf16, ctc_weight)
        logger.info("Epoch %d/%d | Val loss: %.4f", epoch + 1, num_epochs, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, output_dir, "best")
            logger.info("New best (val_loss=%.4f)", val_loss)

    _save_checkpoint(model, output_dir, "final")
    logger.info("Done: %s", experiment_name)


def _run_validation(model, val_loader, device, stage, use_bf16, ctc_weight):
    model.train(False)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if stage == "ctc_pretrain":
                    loss = model.forward_ctc(mel, batch["ctc_targets"].to(device),
                                             batch["ctc_target_lengths"].to(device))
                else:
                    text_ids = batch["text_ids"].to(device)
                    ctc_t = batch["ctc_targets"].to(device) if ctc_weight > 0 else None
                    ctc_l = batch["ctc_target_lengths"].to(device) if ctc_weight > 0 else None
                    loss = model.forward_e2e(mel, text_ids, ctc_weight=ctc_weight,
                                            ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def _save_checkpoint(model, output_dir, step):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(exist_ok=True)
    state = {name: param.data for name, param in model.named_parameters() if not name.startswith("llm.")}
    torch.save(state, ckpt_dir / "model.pt")
    logger.info("Saved checkpoint-%s", step)


def main():
    parser = argparse.ArgumentParser(description="OmniAudio v2 Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
