"""
exp028v2: 1.08B Llama with GQA + Z-loss on ~16.2B Kazakh tokens, 8xH100 DDP.
Architecture: 2048h, 22L, 16 heads, 4 KV heads (GQA), 5504 inter, tied embeddings.
NO QK-Norm, NO embedding scaling — both are NOT HF-compatible (exp028 lesson).
Supports checkpoint resume for spot/interruptible instances.
Usage: torchrun --nproc_per_node=8 train_1b_ddp.py [--resume /path/to/checkpoint]
"""
import os, sys, subprocess
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import math, time, json, signal
from contextlib import nullcontext
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from prepare_1b import MAX_SEQ_LEN, VOCAB_SIZE, make_dataloader, compute_val_bpb

Z_LOSS_COEFF = 1e-4  # logit regularization (PaLM/Gemma)
# NOTE: QK-Norm and EMB_SCALE removed — NOT compatible with HF LlamaForCausalLM

# Graceful shutdown flag for spot preemption
_SHUTDOWN_REQUESTED = False
def _sigterm_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print("\n  !!! SIGTERM received — saving checkpoint and exiting gracefully !!!")
signal.signal(signal.SIGTERM, _sigterm_handler)

@dataclass
class LlamaConfig:
    vocab_size: int = VOCAB_SIZE; seq_len: int = MAX_SEQ_LEN
    n_layer: int = 22; n_head: int = 16; n_kv_head: int = 4  # GQA: 4 KV heads
    n_embd: int = 2048; intermediate_size: int = 5504; tie_embeddings: bool = True

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :].to(torch.bfloat16))
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :].to(torch.bfloat16))
    def forward(self, T):
        return self.cos_cached[:,:,:T], self.sin_cached[:,:,:T]

def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_head, c.n_kv_head, c.n_embd // c.n_head
        self.q = nn.Linear(c.n_embd, c.n_head * self.hd, bias=False)
        self.k = nn.Linear(c.n_embd, c.n_kv_head * self.hd, bias=False)
        self.v = nn.Linear(c.n_embd, c.n_kv_head * self.hd, bias=False)
        self.o = nn.Linear(c.n_head * self.hd, c.n_embd, bias=False)
        # NO QK-Norm — removed, not HF-compatible (exp028 lesson)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        # RoPE (no QK-Norm before it)
        q = apply_rotary(q, cos, sin); k = apply_rotary(k, cos, sin)
        # GQA: expand KV heads
        if self.nkv < self.nh:
            r = self.nh // self.nkv
            k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Linear(c.n_embd, c.intermediate_size, bias=False)
        self.u = nn.Linear(c.n_embd, c.intermediate_size, bias=False)
        self.d = nn.Linear(c.intermediate_size, c.n_embd, bias=False)
    def forward(self, x): return self.d(F.silu(self.g(x)) * self.u(x))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = RMSNorm(c.n_embd); self.attn = Attention(c)
        self.ln2 = RMSNorm(c.n_embd); self.mlp = SwiGLU(c)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))

class Llama(nn.Module):
    def __init__(self, c):
        super().__init__(); self.config = c
        self.emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.norm = RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.rot = RotaryEmbedding(c.n_embd // c.n_head, c.seq_len)
        if c.tie_embeddings: self.head.weight = self.emb.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx):
        x = self.emb(idx)  # NO embedding scaling — not HF-compatible (exp028 lesson)
        cos, sin = self.rot(idx.shape[1])
        for layer in self.layers: x = layer(x, cos, sin)
        return self.head(self.norm(x))

def clean_state_dict(model_ddp):
    sd = model_ddp.module.state_dict()
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total steps (for smoke test)")
    parser.add_argument("--publish-every", type=int, default=0, help="Publish checkpoint to HF every N steps (0=disabled)")
    parser.add_argument("--first-publish", type=int, default=50, help="Publish first checkpoint early at this step (for verification)")
    args, _ = parser.parse_known_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    is_main = (rank == 0)

    cfg = LlamaConfig()
    batch_size = 16
    grad_accum = 4
    lr = 2e-4
    warmup_steps = 1000
    total_tokens_target = 16_200_000_000
    tokens_per_step = batch_size * grad_accum * cfg.seq_len * world_size
    total_steps = total_tokens_target // tokens_per_step
    if args.max_steps:
        total_steps = args.max_steps
    save_steps = min(500, max(total_steps // 20, 100))  # Save every 500 steps (was ~770)
    log_steps = 25

    if is_main:
        print(f"{'='*70}")
        print(f"  exp028v2: Llama 1.08B (GQA+Zloss) on ~16.2B tokens")
        print(f"  NO QK-Norm, NO embedding scaling (HF-compatible)")
        print(f"  {world_size}x GPU DDP")
        print(f"  GQA: {cfg.n_head} heads, {cfg.n_kv_head} KV heads")
        print(f"  BS={batch_size}x{world_size}GPUx{grad_accum}accum = {tokens_per_step//1024} blocks/step = {tokens_per_step:,} tok/step")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Save every: {save_steps} steps")
        print(f"{'='*70}")

    model = Llama(cfg).to(device=device, dtype=torch.bfloat16)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main: print(f"  Params: {num_params:.1f}M")
    # Resume from checkpoint BEFORE compile+DDP
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        meta_path = os.path.join(args.resume, "meta.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            start_step = meta["step"]
            if is_main: print(f"  Resuming from step {start_step}")
        model_path = os.path.join(args.resume, "model.pt")
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(sd)
            if is_main: print(f"  Loaded model weights")

    model = torch.compile(model)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    if args.resume and os.path.exists(args.resume):
        opt_path = os.path.join(args.resume, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))
            if is_main: print(f"  Loaded optimizer state")

    torch.manual_seed(42 + rank + start_step)
    loader = make_dataloader("train", batch_size, cfg.seq_len)

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * min(progress, 1.0)))

    model.train()
    total_tokens = start_step * tokens_per_step
    if is_main: torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    running_loss = 0.0

    # Skip steps if resuming
    if start_step > 0:
        for _ in range(start_step):
            next(loader)  # advance dataloader
        if is_main: print(f"  Skipped {start_step} dataloader batches")

    for step in range(start_step, total_steps):
        if _SHUTDOWN_REQUESTED:
            if is_main:
                ckpt_dir = f"/root/checkpoints/exp028_1b/step_{step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(clean_state_dict(model), f"{ckpt_dir}/model.pt")
                torch.save(optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
                with open(f"{ckpt_dir}/meta.json", "w") as f:
                    json.dump(dict(step=step, tokens=total_tokens, loss=running_loss), f)
                print(f"  >>> Emergency checkpoint saved: {ckpt_dir}")
            break
        current_lr = get_lr(step)
        for pg in optimizer.param_groups: pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(grad_accum):
            x, y = next(loader)
            x, y = x.to(device), y.to(device)
            ctx = model.no_sync() if micro < grad_accum - 1 else nullcontext()
            with ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    # Z-loss: regularize logit magnitudes (PaLM/Gemma)
                    z_loss = Z_LOSS_COEFF * logits.float().logsumexp(-1).pow(2).mean()
                    loss = ce_loss + z_loss
                (loss / grad_accum).backward()
                accum_loss += ce_loss.item() / grad_accum  # track CE only
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_tokens += tokens_per_step

        running_loss = 0.9 * running_loss + 0.1 * accum_loss if running_loss > 0 else accum_loss

        if is_main and (step + 1) % log_steps == 0:
            elapsed = time.time() - t0
            tps = total_tokens / elapsed
            eta_h = (total_steps - step - 1) * elapsed / (step + 1) / 3600
            print(f"  step {step+1:>6}/{total_steps} | loss {running_loss:.4f} | lr {current_lr:.2e} | {tps:.0f} tok/s | {elapsed/60:.0f}m | ETA {eta_h:.1f}h")

        # Determine if we should save checkpoint at this step
        should_save = is_main and (step + 1) % save_steps == 0
        # Also save at first_publish step for early verification
        if is_main and args.first_publish and (step + 1) == args.first_publish:
            should_save = True

        if should_save:
            ckpt_dir = f"/root/checkpoints/exp028_1b/step_{step+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(clean_state_dict(model), f"{ckpt_dir}/model.pt")
            torch.save(optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
            with open(f"{ckpt_dir}/meta.json", "w") as f:
                json.dump(dict(step=step+1, tokens=total_tokens, loss=running_loss, lr=current_lr), f)
            print(f"  >>> Saved checkpoint: {ckpt_dir}")

            # Cleanup: keep only last 3 local checkpoints (rest are on HF)
            import glob, shutil as _shutil
            _all = sorted(glob.glob("/root/checkpoints/exp028_1b/step_*"), key=lambda x: int(x.split("_")[-1]))
            for _old in _all[:-3]:
                _shutil.rmtree(_old)
                print(f"  >>> Cleaned old checkpoint: {_old}")

            # Publish to HF as background process (non-blocking)
            should_publish = args.publish_every and (step + 1) % args.publish_every == 0
            if args.first_publish and (step + 1) == args.first_publish:
                should_publish = True
            if should_publish:
                print(f"  >>> Publishing step {step+1} to HF (background)...")
                subprocess.Popen(
                    [sys.executable, "publish_checkpoint.py", ckpt_dir,
                     "--step", str(step+1), "--skip-inference"],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=open(f"/tmp/publish_{step+1}.log", "w"),
                    stderr=subprocess.STDOUT,
                )

    train_time = time.time() - t0

    if is_main:
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n  Training done: {train_time/3600:.1f}h, {total_tokens/1e9:.1f}B tokens")
        print(f"  Peak VRAM: {peak_vram:.1f}GB per GPU")
        print("  Running validation...")
        model_eval = model.module
        model_eval.eval()
        val_bpb = compute_val_bpb(model_eval, batch_size, device, cfg.seq_len)
        print(f"  val_bpb: {val_bpb:.6f}")

        final_dir = "/root/checkpoints/exp028_1b/final"
        os.makedirs(final_dir, exist_ok=True)
        torch.save(clean_state_dict(model), f"{final_dir}/model.pt")
        results_data = dict(
            val_bpb=val_bpb, train_loss=running_loss, total_steps=total_steps,
            total_tokens=total_tokens, train_hours=round(train_time/3600, 2),
            peak_vram_gb=round(peak_vram, 1), world_size=world_size,
            params_M=round(num_params, 1),
            arch="GQA-4kv+Zloss",
        )
        with open(f"{final_dir}/results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        with open(f"{final_dir}/meta.json", "w") as f:
            json.dump(dict(step=total_steps, tokens=total_tokens, loss=running_loss), f)
        print(f"  Final model saved: {final_dir}")

        # Publish final to HF (foreground — wait for it)
        if args.publish_every:
            print(f"  >>> Publishing FINAL model to HF...")
            subprocess.run(
                [sys.executable, "publish_checkpoint.py", final_dir,
                 "--step", "final"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
