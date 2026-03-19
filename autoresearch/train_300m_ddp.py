"""
Full training: 300M Llama on 9B Kazakh tokens, 16×RTX 4090 DDP.
Usage: torchrun --nproc_per_node=16 train_300m_ddp.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc, math, time, json
from contextlib import nullcontext
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from prepare import MAX_SEQ_LEN, VOCAB_SIZE, make_dataloader, compute_val_bpb

@dataclass
class LlamaConfig:
    vocab_size: int = VOCAB_SIZE; seq_len: int = MAX_SEQ_LEN
    n_layer: int = 18; n_head: int = 16; n_kv_head: int = 16
    n_embd: int = 1024; intermediate_size: int = 3584; tie_embeddings: bool = True

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
        self.q = nn.Linear(c.n_embd, c.n_head*self.hd, bias=False)
        self.k = nn.Linear(c.n_embd, c.n_kv_head*self.hd, bias=False)
        self.v = nn.Linear(c.n_embd, c.n_kv_head*self.hd, bias=False)
        self.o = nn.Linear(c.n_head*self.hd, c.n_embd, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B,T,self.nh,self.hd).transpose(1,2)
        k = self.k(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        v = self.v(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        q = apply_rotary(q, cos, sin); k = apply_rotary(k, cos, sin)
        if self.nkv < self.nh:
            r = self.nh // self.nkv
            k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(B,T,-1))

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
        self.ln1=RMSNorm(c.n_embd); self.attn=Attention(c)
        self.ln2=RMSNorm(c.n_embd); self.mlp=SwiGLU(c)
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
        x = self.emb(idx); cos, sin = self.rot(idx.shape[1])
        for layer in self.layers: x = layer(x, cos, sin)
        return self.head(self.norm(x))

def clean_state_dict(model_ddp):
    """Get state dict without _orig_mod. prefix from torch.compile."""
    sd = model_ddp.module.state_dict()
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

def main():
    # DDP setup
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    is_main = (rank == 0)

    # Config
    cfg = LlamaConfig()
    batch_size = 32  # per GPU (H100 80GB, 300M model uses ~15GB, 32 fits easily)
    grad_accum = 1   # no accumulation needed — NVLink fast sync, max throughput
    lr = 6e-4
    warmup_steps = 500
    total_tokens_target = 9_000_000_000  # 9B
    tokens_per_step = batch_size * grad_accum * cfg.seq_len * world_size
    total_steps = total_tokens_target // tokens_per_step
    save_steps = max(total_steps // 10, 500)
    log_steps = 25

    if is_main:
        print(f"{'='*70}")
        print(f"  exp020: Llama 300M on 9B Kazakh tokens")
        print(f"  {world_size}× RTX 4090 DDP")
        print(f"  BS={batch_size}×{world_size}GPU×{grad_accum}accum = {tokens_per_step//1024} blocks/step = {tokens_per_step:,} tok/step")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Save every: {save_steps} steps")
        print(f"{'='*70}")

    # Model
    model = Llama(cfg).to(device=device, dtype=torch.bfloat16)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main: print(f"  Params: {num_params:.1f}M")
    model = torch.compile(model)
    model = DDP(model, device_ids=[rank])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    # Dataloader (each rank gets different random samples)
    torch.manual_seed(42 + rank)
    loader = make_dataloader("train", batch_size, cfg.seq_len)

    # LR schedule: cosine with warmup
    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * min(progress, 1.0)))

    # Training
    model.train()
    total_tokens = 0
    if is_main: torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    running_loss = 0.0

    for step in range(total_steps):
        current_lr = get_lr(step)
        for pg in optimizer.param_groups: pg["lr"] = current_lr

        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        x, y = next(loader)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_tokens += tokens_per_step

        running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss > 0 else loss.item()

        if is_main and (step + 1) % log_steps == 0:
            elapsed = time.time() - t0
            tps = total_tokens / elapsed
            eta_h = (total_steps - step - 1) * elapsed / (step + 1) / 3600
            print(f"  step {step+1:>6}/{total_steps} | loss {running_loss:.4f} | lr {current_lr:.2e} | {tps:.0f} tok/s | {elapsed/60:.0f}m | ETA {eta_h:.1f}h")

        # Save checkpoint
        if is_main and (step + 1) % save_steps == 0:
            ckpt_dir = f"/root/checkpoints/step_{step+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(clean_state_dict(model), f"{ckpt_dir}/model.pt")
            torch.save(optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
            with open(f"{ckpt_dir}/meta.json", "w") as f:
                json.dump(dict(step=step+1, tokens=total_tokens, loss=running_loss, lr=current_lr), f)
            print(f"  >>> Saved checkpoint: {ckpt_dir}")

    train_time = time.time() - t0

    # Final validation (main process only)
    if is_main:
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n  Training done: {train_time/3600:.1f}h, {total_tokens/1e9:.1f}B tokens")
        print(f"  Peak VRAM: {peak_vram:.1f}GB per GPU")
        print("  Running validation...")
        model.eval()
        val_bpb = compute_val_bpb(model.module, batch_size, device, cfg.seq_len)
        print(f"  val_bpb: {val_bpb:.6f}")

        # Save final model
        final_dir = "/root/checkpoints/final"
        os.makedirs(final_dir, exist_ok=True)
        torch.save(clean_state_dict(model), f"{final_dir}/model.pt")
        with open(f"{final_dir}/results.json", "w") as f:
            json.dump(dict(
                val_bpb=val_bpb, train_loss=running_loss, total_steps=total_steps,
                total_tokens=total_tokens, train_hours=round(train_time/3600, 2),
                peak_vram_gb=round(peak_vram, 1), world_size=world_size,
                params_M=round(num_params, 1),
            ), f, indent=2)
        print(f"  Final model saved: {final_dir}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
