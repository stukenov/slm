#!/usr/bin/env python3
"""
Unified EkiTil training script for 123M / 300M / 600M models.
Uploads checkpoints to HuggingFace for remote resume.

Usage:
    # Single GPU:
    python3 train_ekitil.py --size 300m
    # Multi-GPU:
    torchrun --nproc_per_node=4 train_ekitil.py --size 600m
    # Resume from HF checkpoint:
    python3 train_ekitil.py --size 300m --resume hf
    # Resume from local checkpoint:
    python3 train_ekitil.py --size 300m --resume /root/checkpoints/step_5000
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, math, time, json, argparse, urllib.parse, urllib.request, shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---- Shared Config ----
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET = "stukenov/ekitil-corpus-tokenized-kkru-v1"
HF_TOKENIZER = "stukenov/ekitil-vocab-bpe-64k-kkru-v1"

VOCAB_SIZE = 64_000
SEQ_LEN = 2048
BLOCKS = 1_205_750
TOKENS_PER_EPOCH = BLOCKS * SEQ_LEN  # 2,469,376,000

TG_BOT_TOKEN = "8620178354:AAHRfG7FX4bIqK-Hq_W7XoGCaB7FI6MKFb8"
TG_CHAT_ID = "47474471"

CACHE_DIR = "/root/cache"
CKPT_DIR = "/root/checkpoints"

# ---- Model Configs ----
MODEL_CONFIGS = {
    "123m": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "intermediate_size": 2048,
        "epochs": 1,
        "lr": 6e-4,
        "batch_size": 16,
        "grad_accum": 8,
        "hf_output": "stukenov/ekitil-core-qwen3-123m-kkru-base-v1",
        "hf_ckpt_repo": "stukenov/ekitil-core-qwen3-123m-kkru-checkpoints",
    },
    "300m": {
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2816,
        "epochs": 2,
        "lr": 3e-4,
        "batch_size": 8,
        "grad_accum": 8,
        "hf_output": "stukenov/ekitil-core-qwen3-300m-kkru-base-v1",
        "hf_ckpt_repo": "stukenov/ekitil-core-qwen3-300m-kkru-checkpoints",
    },
    "600m": {
        "hidden_size": 1280,
        "num_hidden_layers": 28,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "intermediate_size": 4480,
        "epochs": 5,
        "lr": 2e-4,
        "batch_size": 4,
        "grad_accum": 16,
        "hf_output": "stukenov/ekitil-core-qwen3-600m-kkru-base-v1",
        "hf_ckpt_repo": "stukenov/ekitil-core-qwen3-600m-kkru-checkpoints",
    },
}


def tg_send(text: str):
    try:
        url = (f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage?"
               f"chat_id={TG_CHAT_ID}&text={urllib.parse.quote(text[:4000])}")
        urllib.request.urlopen(url, timeout=10)
    except Exception:
        pass


def download_data():
    """Download dataset and convert to memory-mapped numpy array."""
    data_bin = os.path.join(CACHE_DIR, "train.bin")
    if os.path.exists(data_bin):
        n_tokens = os.path.getsize(data_bin) // 2
        print(f"  Data cached: {data_bin} ({n_tokens:,} tokens)")
        return data_bin

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("  Downloading dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split="train", cache_dir=CACHE_DIR)
    print(f"  {len(ds):,} blocks loaded")

    print(f"  Writing {data_bin}...")
    with open(data_bin, "wb") as f:
        for i in range(0, len(ds), 10_000):
            batch = ds[i:i+10_000]["input_ids"]
            arr = np.array(batch, dtype=np.uint16)
            f.write(arr.tobytes())
            if i > 0 and i % 100_000 == 0:
                print(f"    {i:,}/{len(ds):,} blocks written")
    del ds
    gc.collect()
    n_tokens = os.path.getsize(data_bin) // 2
    print(f"  Done: {n_tokens:,} tokens")
    return data_bin


class DataLoader:
    """Random-access memmap dataloader. Each rank gets different samples."""
    def __init__(self, data_bin, batch_size, seq_len, rank=0, seed=42):
        n_tokens = os.path.getsize(data_bin) // 2
        self.n_blocks = n_tokens // seq_len
        self.data = np.memmap(data_bin, dtype=np.uint16, mode="r", shape=(self.n_blocks, seq_len))
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed + rank)

    def get_batch(self, device):
        idx = self.rng.integers(0, self.n_blocks, size=self.batch_size)
        block = torch.from_numpy(self.data[idx].astype(np.int64)).to(device)
        return block[:, :-1], block[:, 1:]


def build_model(cfg, device):
    """Build Qwen3ForCausalLM from config dict."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=64,
        intermediate_size=cfg["intermediate_size"],
        hidden_act="silu",
        max_position_embeddings=SEQ_LEN,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
    )

    model = Qwen3ForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: Qwen3 {n_params:.1f}M params")
    return model.to(device=device, dtype=torch.bfloat16)


def save_checkpoint(model, optimizer, step, tokens, loss, lr, is_ddp=False):
    ckpt_dir = os.path.join(CKPT_DIR, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    sd = model.module.state_dict() if is_ddp else model.state_dict()
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    torch.save(sd, os.path.join(ckpt_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(dict(step=step, tokens=tokens, loss=loss, lr=lr), f)
    print(f"  >>> Checkpoint saved: {ckpt_dir}")


def upload_checkpoint_to_hf(model, optimizer, step, tokens, loss, lr, cfg, is_ddp=False):
    """Upload full checkpoint (model + optimizer + meta) to HF for resume."""
    from huggingface_hub import HfApi

    ckpt_repo = cfg["hf_ckpt_repo"]
    print(f"  >>> Uploading full checkpoint step_{step} to {ckpt_repo}...", flush=True)

    try:
        out_dir = f"/root/hf_ckpt_step_{step}"
        os.makedirs(out_dir, exist_ok=True)

        # Save model weights via save_pretrained (safetensors)
        raw_model = model.module if is_ddp else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        raw_model.save_pretrained(out_dir)

        # Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))

        # Save full training meta for resume
        with open(os.path.join(out_dir, "training_meta.json"), "w") as f:
            json.dump(dict(
                step=step, tokens=tokens, loss=round(loss, 4),
                bpb=round(loss / math.log(2), 4), lr=lr,
                model_size=cfg.get("hf_output", "").split("-")[-4] if "hf_output" in cfg else "",
            ), f)

        api = HfApi(token=HF_TOKEN)
        api.create_repo(ckpt_repo, exist_ok=True, token=HF_TOKEN)
        api.upload_folder(
            folder_path=out_dir,
            repo_id=ckpt_repo,
            path_in_repo=f"step_{step}",
            token=HF_TOKEN,
        )
        print(f"  >>> Checkpoint uploaded to {ckpt_repo}/step_{step}", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception as e:
        print(f"  !!! Checkpoint upload failed: {e}", flush=True)


def load_checkpoint(model, optimizer, ckpt_dir, device, is_ddp=False):
    print(f"  Resuming from {ckpt_dir}...")
    sd = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device, weights_only=True)
    target = model.module if is_ddp else model
    raw = target._orig_mod if hasattr(target, "_orig_mod") else target
    raw.load_state_dict(sd)
    opt_sd = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location=device, weights_only=True)
    optimizer.load_state_dict(opt_sd)
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)
    print(f"  Resumed at step {meta['step']}, tokens {meta['tokens']:,}")
    return meta["step"], meta["tokens"]


def resume_from_hf(model, cfg, device):
    """Download latest checkpoint from HF and return step number."""
    from huggingface_hub import HfApi, hf_hub_download
    ckpt_repo = cfg["hf_ckpt_repo"]
    print(f"  Checking HF for checkpoints: {ckpt_repo}...")

    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(ckpt_repo, token=HF_TOKEN)
        # Find latest step
        steps = set()
        for f in files:
            if f.startswith("step_") and "/" in f:
                step_str = f.split("/")[0].replace("step_", "")
                if step_str.isdigit():
                    steps.add(int(step_str))
        if not steps:
            print("  No checkpoints found on HF")
            return 0

        latest = max(steps)
        print(f"  Found checkpoint step_{latest} on HF, downloading...")

        # Download model weights
        from transformers import Qwen3ForCausalLM
        raw = model.module if hasattr(model, "module") else model
        raw_inner = raw._orig_mod if hasattr(raw, "_orig_mod") else raw

        # Download safetensors
        local_path = hf_hub_download(
            ckpt_repo, f"step_{latest}/model.safetensors",
            token=HF_TOKEN, cache_dir=CACHE_DIR
        )
        from safetensors.torch import load_file
        sd = load_file(local_path)
        raw_inner.load_state_dict(sd)
        print(f"  Resumed model weights from HF step_{latest}")

        # Download meta
        try:
            meta_path = hf_hub_download(
                ckpt_repo, f"step_{latest}/training_meta.json",
                token=HF_TOKEN, cache_dir=CACHE_DIR
            )
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  Meta: {meta}")
        except Exception:
            pass

        return latest
    except Exception as e:
        print(f"  HF resume failed: {e}")
        return 0


def upload_final(model, cfg, is_ddp=False):
    """Upload final model to HuggingFace."""
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import PreTrainedTokenizerFast

    hf_output = cfg["hf_output"]
    print(f"  Uploading final model to {hf_output}...")

    out_dir = "/root/hf_model"
    raw_model = model.module if is_ddp else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    raw_model.save_pretrained(out_dir)

    tok_path = hf_hub_download(HF_TOKENIZER, "tokenizer.json", token=HF_TOKEN, cache_dir=CACHE_DIR)
    shutil.copy(tok_path, os.path.join(out_dir, "tokenizer.json"))

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|padding|>"
    tokenizer.bos_token = "<|startoftext|>"
    tokenizer.save_pretrained(out_dir)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(hf_output, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(folder_path=out_dir, repo_id=hf_output, token=HF_TOKEN)
    print(f"  Uploaded: {hf_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, required=True, choices=["123m", "300m", "600m"])
    parser.add_argument("--resume", type=str, default=None, help="'hf' or local checkpoint path")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-GPU micro batch")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--upload-every", type=int, default=2000, help="Upload checkpoint to HF every N steps")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.size]
    batch_size = args.batch_size or cfg["batch_size"]
    grad_accum = args.grad_accum or cfg["grad_accum"]
    lr = args.lr or cfg["lr"]

    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
    else:
        rank, world_size, device = 0, 1, "cuda"

    is_main = (rank == 0)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    total_tokens_target = TOKENS_PER_EPOCH * cfg["epochs"]
    size_label = args.size.upper()

    if is_main:
        print("=" * 60)
        print(f"  EkiTil {size_label} — Qwen3 bilingual kk-ru")
        print(f"  {world_size} GPU(s), DDP={ddp}, {cfg['epochs']} epoch(s)")
        print(f"  Target: {total_tokens_target/1e9:.2f}B tokens")
        print("=" * 60)

    # Data
    if ddp:
        if is_main:
            data_bin = download_data()
        dist.barrier()
        if not is_main:
            data_bin = os.path.join(CACHE_DIR, "train.bin")
    else:
        data_bin = download_data()

    loader = DataLoader(data_bin, batch_size, SEQ_LEN, rank=rank)

    # Model
    model = build_model(cfg, device)
    if not args.no_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[rank])

    # Training params
    tokens_per_step = batch_size * grad_accum * (SEQ_LEN - 1) * world_size
    total_steps = total_tokens_target // tokens_per_step
    save_steps = max(total_steps // 20, 500)
    log_steps = 25

    if is_main:
        print(f"  BS={batch_size} x {grad_accum}accum x {world_size}GPU = {tokens_per_step:,} tok/step")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Save every: {save_steps} steps")
        print(f"  Upload to HF every: {args.upload_every} steps")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.95), weight_decay=0.1, fused=True,
    )

    # Resume
    start_step = 0
    total_tokens = 0
    if args.resume == "hf":
        if is_main:
            start_step = resume_from_hf(model, cfg, device)
            total_tokens = start_step * tokens_per_step
        if ddp:
            dist.barrier()
    elif args.resume:
        start_step, total_tokens = load_checkpoint(model, optimizer, args.resume, device, is_ddp=ddp)

    # LR schedule
    def get_lr(step):
        if step < args.warmup_steps:
            return lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * min(progress, 1.0)))

    # Train
    if is_main:
        tg_send(f"🚀 EkiTil {size_label} training started\n"
                f"{world_size} GPU(s), {cfg['epochs']} epochs\n"
                f"Steps: {total_steps:,}, from step {start_step}\n"
                f"BS: {batch_size}x{grad_accum}x{world_size}, LR: {lr}")

    model.train()
    if is_main:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    running_loss = 0.0
    eta_h = 0.0

    for step in range(start_step, total_steps):
        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            x, y = loader.get_batch(device)
            if ddp and micro_step < grad_accum - 1:
                with model.no_sync():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids=x).logits
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                        loss = loss / grad_accum
                    loss.backward()
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=x).logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss = loss / grad_accum
                loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_tokens += tokens_per_step
        running_loss = 0.95 * running_loss + 0.05 * accum_loss if running_loss > 0 else accum_loss

        # Log
        if is_main and (step + 1) % log_steps == 0:
            elapsed = time.time() - t0
            tps = (total_tokens - start_step * tokens_per_step) / elapsed
            eta_h = (total_steps - step - 1) / max(1, (step - start_step + 1)) * elapsed / 3600
            bpb = running_loss / math.log(2)
            epoch_num = total_tokens / TOKENS_PER_EPOCH
            print(f"  step {step+1:>6}/{total_steps} | loss {running_loss:.4f} | bpb {bpb:.4f} | "
                  f"lr {current_lr:.2e} | {tps:.0f} tok/s | epoch {epoch_num:.2f} | ETA {eta_h:.1f}h",
                  flush=True)

        # Local checkpoint + upload to HF every save_steps
        if is_main and (step + 1) % save_steps == 0:
            save_checkpoint(model, optimizer, step + 1, total_tokens, running_loss, current_lr, is_ddp=ddp)
            upload_checkpoint_to_hf(model, optimizer, step + 1, total_tokens, running_loss, current_lr, cfg, is_ddp=ddp)
            tg_send(f"💾 {size_label} step {step+1}/{total_steps} | loss {running_loss:.4f} | "
                    f"{total_tokens/1e9:.2f}B tok | ETA {eta_h:.1f}h")

    # Done
    if is_main:
        train_time = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n  Training done: {train_time/3600:.1f}h, {total_tokens/1e9:.1f}B tokens")
        print(f"  Peak VRAM: {peak_vram:.1f}GB")
        print(f"  Final loss: {running_loss:.4f}")

        save_checkpoint(model, optimizer, total_steps, total_tokens, running_loss, get_lr(total_steps - 1), is_ddp=ddp)
        upload_final(model, cfg, is_ddp=ddp)

        results = dict(
            model_size=args.size,
            final_loss=round(running_loss, 4),
            final_bpb=round(running_loss / math.log(2), 4),
            total_steps=total_steps,
            total_tokens=total_tokens,
            epochs=cfg["epochs"],
            train_hours=round(train_time / 3600, 2),
            peak_vram_gb=round(peak_vram, 1),
            params_M=round(sum(p.numel() for p in (model.module if ddp else model).parameters()) / 1e6, 1),
            world_size=world_size,
            hf_model=cfg["hf_output"],
        )
        with open(os.path.join(CKPT_DIR, f"results_{args.size}.json"), "w") as f:
            json.dump(results, f, indent=2)

        tg_send(f"✅ EkiTil {size_label} DONE!\n"
                f"Loss: {running_loss:.4f} | BPB: {running_loss/math.log(2):.4f}\n"
                f"Time: {train_time/3600:.1f}h | VRAM: {peak_vram:.1f}GB\n"
                f"Tokens: {total_tokens/1e9:.1f}B ({cfg['epochs']} epochs)\n"
                f"Model: {cfg['hf_output']}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
