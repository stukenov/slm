"""Training script for OmniAudio v2: 3-stage (CTC pretrain, alignment, E2E)."""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

from omniaudio.data_v2 import AudioCollatorV2, PrecomputedMelCollator, load_speech_dataset
from omniaudio.model_v2 import OmniAudioV2Model, OmniAudioScratchModel

logger = logging.getLogger(__name__)


def _is_ddp():
    return dist.is_initialized()


def _rank():
    return dist.get_rank() if _is_ddp() else 0


def _world_size():
    return dist.get_world_size() if _is_ddp() else 1


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
    model_type = config.get("model_type", "pretrained")

    if model_type == "scratch":
        freeze_decoder = config.get("freeze_decoder", False)
        if stage == "ctc_pretrain":
            for p in model.parameters():
                p.requires_grad = False
            for p in model.encoder.parameters():
                p.requires_grad = True
            for p in model.ctc_head.parameters():
                p.requires_grad = True
        elif freeze_decoder:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.encoder.parameters():
                p.requires_grad = True
            for p in model.ctc_head.parameters():
                p.requires_grad = True
            # projector between encoder and decoder
            if hasattr(model, "projector"):
                for p in model.projector.parameters():
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Stage: %s | Params: %.2fM total, %.2fM trainable", stage, total / 1e6, trainable / 1e6)
        return

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
        # Unfreeze last N layers of LLM decoder
        unfreeze_layers = config.get("unfreeze_llm_layers", 0)
        if unfreeze_layers > 0 and model.llm is not None:
            total_layers = len(model.llm.model.layers)
            for i in range(total_layers - unfreeze_layers, total_layers):
                for p in model.llm.model.layers[i].parameters():
                    p.requires_grad = True
            # Also unfreeze final norm and lm_head
            for p in model.llm.model.norm.parameters():
                p.requires_grad = True
            for p in model.llm.lm_head.parameters():
                p.requires_grad = True
            logger.info("Unfroze last %d/%d LLM layers + norm + lm_head", unfreeze_layers, total_layers)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Stage: %s | Params: %.2fM total, %.2fM trainable", stage, total / 1e6, trainable / 1e6)


def train(config):
    experiment_name = config["experiment_name"]
    output_dir = Path(config["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    stage = config["stage"]
    if _rank() == 0:
        logger.info("Device: %s | Stage: %s | World: %d GPUs", device, stage, _world_size())

    encoder_config = {
        "n_mels": config["n_mels"],
        "d_model": config["audio_d_model"],
        "n_heads": config["audio_n_heads"],
        "n_layers": config["audio_n_layers"],
        "n_conv": config["audio_n_conv"],
    }

    model_type = config.get("model_type", "pretrained")
    if model_type == "scratch":
        decoder_config = {
            "d_model": config["decoder_d_model"],
            "n_heads": config["decoder_n_heads"],
            "n_layers": config["decoder_n_layers"],
        }
        model = OmniAudioScratchModel(
            encoder_config=encoder_config, decoder_config=decoder_config,
            vocab_size=config["vocab_size"],
        )
    else:
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
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in state.items():
            target = model_state.get(key)
            if target is None:
                filtered_state[key] = value
                continue
            if target.shape != value.shape:
                skipped.append((key, tuple(value.shape), tuple(target.shape)))
                continue
            filtered_state[key] = value
        if skipped:
            logger.info("Skipping %d init tensors with mismatched shapes", len(skipped))
            for key, src_shape, dst_shape in skipped[:10]:
                logger.info("  skip %s: %s -> %s", key, src_shape, dst_shape)
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        logger.info("Loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))

    get_trainable_params(model, config)
    model = model.to(device)
    if config.get("torch_compile", False) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)
    if _is_ddp():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if _is_ddp() else model
    use_bf16 = config.get("bf16", True) and device.type == "cuda"

    augment = config.get("augment", stage != "ctc_pretrain")
    dataset_name = config.get("dataset_name", "fleurs")
    if dataset_name == "sozkz_mels":
        collator = PrecomputedMelCollator(
            tokenizer_path=config["tokenizer_path"], max_audio_len=config["max_audio_len"],
            max_text_len=config["max_text_len"], augment=augment,
            text_lowercase=config.get("text_lowercase", False),
            text_strip_punctuation=config.get("text_strip_punctuation", False),
            text_collapse_whitespace=config.get("text_collapse_whitespace", True),
        )
    else:
        collator = AudioCollatorV2(
            tokenizer_path=config["tokenizer_path"], n_mels=config["n_mels"],
            sample_rate=config["sample_rate"], max_audio_len=config["max_audio_len"],
            max_text_len=config["max_text_len"], augment=augment,
            text_lowercase=config.get("text_lowercase", False),
            text_strip_punctuation=config.get("text_strip_punctuation", False),
            text_collapse_whitespace=config.get("text_collapse_whitespace", True),
        )
    train_ds = load_speech_dataset(dataset_name, "train", max_samples=config.get("max_train_samples"))
    val_ds = load_speech_dataset(dataset_name, "validation", max_samples=config.get("max_eval_samples"))
    if _rank() == 0:
        logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    batch_size = config["per_device_train_batch_size"]
    num_workers = config.get("dataloader_num_workers", 4)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if _is_ddp() else None
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
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
    hf_repo_id = config.get("hf_repo_id")

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            mel = batch["mel"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if stage == "ctc_pretrain":
                    ctc_t = batch["ctc_targets"].to(device)
                    ctc_l = batch["ctc_target_lengths"].to(device)
                    if model_type == "scratch":
                        loss = model(mel, ctc_only=True, ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
                    else:
                        loss = raw_model.forward_ctc(mel, ctc_t, ctc_l)
                elif model_type == "scratch":
                    text_ids = batch["text_ids"].to(device)
                    ctc_t = batch["ctc_targets"].to(device) if ctc_weight > 0 else None
                    ctc_l = batch["ctc_target_lengths"].to(device) if ctc_weight > 0 else None
                    loss = model(mel, text_ids, ctc_weight=ctc_weight,
                                 ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
                else:
                    text_ids = batch["text_ids"].to(device)
                    ctc_t = batch["ctc_targets"].to(device) if ctc_weight > 0 else None
                    ctc_l = batch["ctc_target_lengths"].to(device) if ctc_weight > 0 else None
                    loss = raw_model.forward_e2e(
                        mel,
                        text_ids,
                        ctc_weight=ctc_weight,
                        ctc_targets=ctc_t,
                        ctc_target_lengths=ctc_l,
                    )
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

                if _rank() == 0 and global_step % config.get("logging_steps", 50) == 0:
                    avg = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    logger.info("Step %d | Loss: %.4f | LR: %.2e", global_step, avg, lr)

                save_steps = config.get("save_steps")
                if _rank() == 0 and save_steps and global_step % save_steps == 0:
                    _save_checkpoint(raw_model, output_dir, global_step, hf_repo_id)

        avg_train = epoch_loss / max(num_batches, 1)
        if _rank() == 0:
            logger.info("Epoch %d/%d | Train loss: %.4f", epoch + 1, num_epochs, avg_train)

        val_loss = _run_validation(raw_model, val_loader, device, stage, use_bf16, ctc_weight, model_type)
        if _rank() == 0:
            logger.info("Epoch %d/%d | Val loss: %.4f", epoch + 1, num_epochs, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(raw_model, output_dir, "best", hf_repo_id)

    if _rank() == 0:
        _save_checkpoint(raw_model, output_dir, "final", hf_repo_id)
        logger.info("Done: %s", experiment_name)
    if _is_ddp():
        dist.destroy_process_group()


def _run_validation(model, val_loader, device, stage, use_bf16, ctc_weight, model_type="pretrained"):
    model.train(False)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if stage == "ctc_pretrain":
                    ctc_t = batch["ctc_targets"].to(device)
                    ctc_l = batch["ctc_target_lengths"].to(device)
                    if model_type == "scratch":
                        loss = model(mel, ctc_only=True, ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
                    else:
                        loss = model.forward_ctc(mel, ctc_t, ctc_l)
                elif model_type == "scratch":
                    text_ids = batch["text_ids"].to(device)
                    ctc_t = batch["ctc_targets"].to(device) if ctc_weight > 0 else None
                    ctc_l = batch["ctc_target_lengths"].to(device) if ctc_weight > 0 else None
                    loss = model(mel, text_ids, ctc_weight=ctc_weight,
                                 ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
                else:
                    text_ids = batch["text_ids"].to(device)
                    ctc_t = batch["ctc_targets"].to(device) if ctc_weight > 0 else None
                    ctc_l = batch["ctc_target_lengths"].to(device) if ctc_weight > 0 else None
                    loss = model.forward_e2e(mel, text_ids, ctc_weight=ctc_weight,
                                            ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def _save_checkpoint(model, output_dir, step, hf_repo_id=None):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(exist_ok=True)
    # Save all trainable params (encoder+projector+ctc + any unfrozen LLM layers)
    state = {name: param.data for name, param in model.named_parameters() if param.requires_grad or not name.startswith("llm.")}
    torch.save(state, ckpt_dir / "model.pt")
    logger.info("Saved checkpoint-%s", step)

    if hf_repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=hf_repo_id, exist_ok=True, repo_type="model")
            api.upload_file(
                path_or_fileobj=str(ckpt_dir / "model.pt"),
                path_in_repo=f"checkpoint-{step}/model.pt",
                repo_id=hf_repo_id,
            )
            # Also upload as "latest" for easy resume
            api.upload_file(
                path_or_fileobj=str(ckpt_dir / "model.pt"),
                path_in_repo="model.pt",
                repo_id=hf_repo_id,
            )
            logger.info("Pushed checkpoint-%s to %s", step, hf_repo_id)
        except Exception as e:
            logger.warning("HF push failed (training continues): %s", e)


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
