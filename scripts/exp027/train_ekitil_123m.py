#!/usr/bin/env python3
"""
Train EkiTil 123M (Qwen3 architecture) on 2.47B bilingual kk-ru tokens.
Multi-GPU DDP, spot-instance friendly with auto-resume from checkpoint.

Usage:
    # Single GPU:
    python3 train_ekitil_123m.py
    # Multi-GPU:
    torchrun --nproc_per_node=4 train_ekitil_123m.py
    # Resume:
    torchrun --nproc_per_node=4 train_ekitil_123m.py --resume /root/checkpoints/step_5000
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc, math, time, json, argparse, urllib.parse, urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---- Config ----
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET = "stukenov/ekitil-corpus-tokenized-kkru-v1"
HF_TOKENIZER = "stukenov/ekitil-vocab-bpe-64k-kkru-v1"
HF_OUTPUT = "stukenov/ekitil-core-qwen3-123m-kkru-base-v1"

VOCAB_SIZE = 64_000
SEQ_LEN = 2048
TOTAL_TOKENS = 2_469_376_000  # 1,205,750 blocks * 2048

TG_BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = "47474471"

CACHE_DIR = "/root/cache"
CKPT_DIR = "/root/checkpoints"


def tg_send(text: str):
    try:
        url = (f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage?"
               f"chat_id={TG_CHAT_ID}&text={urllib.parse.quote(text[:4000])}")
        urllib.request.urlopen(url, timeout=10)
    except Exception:
        pass


def download_data():
    """Download dataset parquet and convert to memory-mapped numpy array."""
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


def build_model(device):
    """Build Qwen3ForCausalLM 123M from transformers."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=2048,
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
    # Strip _orig_mod. prefix from torch.compile
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    torch.save(sd, os.path.join(ckpt_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(dict(step=step, tokens=tokens, loss=loss, lr=lr), f)
    print(f"  >>> Checkpoint saved: {ckpt_dir}")


def load_checkpoint(model, optimizer, ckpt_dir, device, is_ddp=False):
    print(f"  Resuming from {ckpt_dir}...")
    sd = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device, weights_only=True)
    target = model.module if is_ddp else model
    # Handle compiled model
    raw = target._orig_mod if hasattr(target, "_orig_mod") else target
    raw.load_state_dict(sd)
    opt_sd = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location=device, weights_only=True)
    optimizer.load_state_dict(opt_sd)
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)
    print(f"  Resumed at step {meta['step']}, tokens {meta['tokens']:,}")
    return meta["step"], meta["tokens"]


def upload_to_hf(model, tokenizer_repo):
    """Convert and upload to HuggingFace Hub."""
    print("  Uploading to HuggingFace...")
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import PreTrainedTokenizerFast
    import shutil

    out_dir = "/root/hf_model"
    model.save_pretrained(out_dir)

    tok_path = hf_hub_download(tokenizer_repo, "tokenizer.json", token=HF_TOKEN, cache_dir=CACHE_DIR)
    shutil.copy(tok_path, os.path.join(out_dir, "tokenizer.json"))

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|padding|>"
    tokenizer.bos_token = "<|startoftext|>"
    tokenizer.save_pretrained(out_dir)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_OUTPUT, exist_ok=True, token=HF_TOKEN)
    api.upload_folder(folder_path=out_dir, repo_id=HF_OUTPUT, token=HF_TOKEN)
    print(f"  Uploaded: {HF_OUTPUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU micro batch")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

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

    # Data — only rank 0 downloads, others wait
    if is_main:
        print("=" * 60)
        print("  EkiTil 123M — Qwen3 bilingual kk-ru")
        print(f"  {world_size} GPU(s), DDP={ddp}")
        print("=" * 60)
    if ddp:
        if is_main:
            data_bin = download_data()
        dist.barrier()
        if not is_main:
            data_bin = os.path.join(CACHE_DIR, "train.bin")
    else:
        data_bin = download_data()

    loader = DataLoader(data_bin, args.batch_size, SEQ_LEN, rank=rank)

    # Model
    model = build_model(device)
    if not args.no_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[rank])

    # Training params
    tokens_per_step = args.batch_size * args.grad_accum * (SEQ_LEN - 1) * world_size
    total_steps = TOTAL_TOKENS // tokens_per_step
    save_steps = max(total_steps // 20, 500)
    log_steps = 25

    if is_main:
        print(f"  BS={args.batch_size} x {args.grad_accum}accum x {world_size}GPU = {tokens_per_step:,} tok/step")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Save every: {save_steps} steps")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1, fused=True,
    )

    # Resume
    start_step = 0
    total_tokens = 0
    if args.resume:
        start_step, total_tokens = load_checkpoint(model, optimizer, args.resume, device, is_ddp=ddp)

    # LR schedule: cosine with warmup, min_lr = 0.1 * lr
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.1 + 0.5 * args.lr * 0.9 * (1 + math.cos(math.pi * min(progress, 1.0)))

    # Train
    if is_main:
        tg_send(f"🚀 EkiTil 123M training started\n"
                f"{world_size} GPU(s), steps: {total_steps:,}\n"
                f"BS: {args.batch_size}x{args.grad_accum}x{world_size}, LR: {args.lr}")

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

        for micro_step in range(args.grad_accum):
            x, y = loader.get_batch(device)
            # Only sync gradients on last micro step
            ctx = model.no_sync() if (ddp and micro_step < args.grad_accum - 1) else torch.utils.data.dataloader._utils.worker._worker_loop.__class__.__mro__[0]  # dummy
            if ddp and micro_step < args.grad_accum - 1:
                with model.no_sync():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids=x).logits
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                        loss = loss / args.grad_accum
                    loss.backward()
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=x).logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss = loss / args.grad_accum
                loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_tokens += tokens_per_step
        running_loss = 0.95 * running_loss + 0.05 * accum_loss if running_loss > 0 else accum_loss

        # Log
        if is_main and (step + 1) % log_steps == 0:
            elapsed = time.time() - t0
            tps = total_tokens / elapsed
            eta_h = (total_steps - step - 1) * elapsed / (step - start_step + 1) / 3600
            bpb = running_loss / math.log(2)
            print(f"  step {step+1:>6}/{total_steps} | loss {running_loss:.4f} | bpb {bpb:.4f} | "
                  f"lr {current_lr:.2e} | {tps:.0f} tok/s | ETA {eta_h:.1f}h")

        # Checkpoint
        if is_main and (step + 1) % save_steps == 0:
            save_checkpoint(model, optimizer, step + 1, total_tokens, running_loss, current_lr, is_ddp=ddp)
            tg_send(f"💾 Step {step+1}/{total_steps} | loss {running_loss:.4f} | "
                    f"{total_tokens/1e9:.2f}B tok | ETA {eta_h:.1f}h")

    # Done
    if is_main:
        train_time = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n  Training done: {train_time/3600:.1f}h, {total_tokens/1e9:.1f}B tokens")
        print(f"  Peak VRAM: {peak_vram:.1f}GB")
        print(f"  Final loss: {running_loss:.4f}")

        save_checkpoint(model, optimizer, total_steps, total_tokens, running_loss, get_lr(total_steps - 1), is_ddp=ddp)

        # Upload — unwrap DDP + compile
        raw_model = model.module if ddp else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        upload_to_hf(raw_model, HF_TOKENIZER)

        results = dict(
            final_loss=round(running_loss, 4),
            final_bpb=round(running_loss / math.log(2), 4),
            total_steps=total_steps,
            total_tokens=total_tokens,
            train_hours=round(train_time / 3600, 2),
            peak_vram_gb=round(peak_vram, 1),
            params_M=round(sum(p.numel() for p in raw_model.parameters()) / 1e6, 1),
            world_size=world_size,
        )
        with open(os.path.join(CKPT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        tg_send(f"✅ EkiTil 123M DONE!\n"
                f"Loss: {running_loss:.4f} | BPB: {running_loss/math.log(2):.4f}\n"
                f"Time: {train_time/3600:.1f}h | VRAM: {peak_vram:.1f}GB\n"
                f"Model: {HF_OUTPUT}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
