"""
Phase 1: Model Size Sweep (5 experiments x 5 min each = 25 min total)
Fills the gap between 150M (known best) and 900M (exp019 target).

Usage: uv run run_phase1.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, make_dataloader, compute_val_bpb


# ---------------------------------------------------------------------------
# Llama Model
# ---------------------------------------------------------------------------

@dataclass
class LlamaConfig:
    name: str = ""
    vocab_size: int = VOCAB_SIZE
    seq_len: int = MAX_SEQ_LEN
    n_layer: int = 24
    n_head: int = 24
    n_kv_head: int = 24
    n_embd: int = 1536
    intermediate_size: int = 5376
    tie_embeddings: bool = True
    batch_size: int = 4
    learning_rate: float = 3e-4


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # shape: [1, 1, seq, dim/2] — broadcasts with [batch, heads, seq, dim/2]
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :].to(torch.bfloat16))
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :].to(torch.bfloat16))

    def forward(self, seq_len):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def apply_rotary(x, cos, sin):
    # x: [batch, heads, seq, head_dim], cos/sin: [1, 1, seq, head_dim/2]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(config.n_embd // config.n_head, config.seq_len)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = self.rotary(T)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------------------
# Phase 1 Configs: 150M -> 250M -> 400M -> 600M -> 900M
# ---------------------------------------------------------------------------

PHASE1 = [
    LlamaConfig(
        name="150M_baseline",
        n_embd=768, n_layer=16, n_head=12, n_kv_head=12,
        intermediate_size=2048,
        batch_size=16, learning_rate=6e-4,
    ),
    LlamaConfig(
        name="250M",
        n_embd=896, n_layer=18, n_head=14, n_kv_head=14,
        intermediate_size=3136,
        batch_size=12, learning_rate=5e-4,
    ),
    LlamaConfig(
        name="400M",
        n_embd=1152, n_layer=20, n_head=18, n_kv_head=18,
        intermediate_size=4032,
        batch_size=6, learning_rate=4e-4,
    ),
    LlamaConfig(
        name="600M",
        n_embd=1280, n_layer=22, n_head=20, n_kv_head=20,
        intermediate_size=4480,
        batch_size=4, learning_rate=3e-4,
    ),
    LlamaConfig(
        name="900M_exp019",
        n_embd=1536, n_layer=24, n_head=24, n_kv_head=24,
        intermediate_size=5376,
        batch_size=4, learning_rate=3e-4,
    ),
]


# ---------------------------------------------------------------------------
# Training loop for one experiment
# ---------------------------------------------------------------------------

def run_experiment(config, device="cuda"):
    print(f"\n{'='*70}")
    print(f"  {config.name}: {config.n_embd}d/{config.n_layer}L/{config.n_head}h  BS={config.batch_size}  LR={config.learning_rate}")
    print(f"{'='*70}")

    model = Llama(config).to(device=device, dtype=torch.bfloat16)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_M = num_params / 1e6
    print(f"  Params: {num_params_M:.1f}M")

    # model = torch.compile(model)  # disabled: triton not available on all images

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        betas=(0.9, 0.95), weight_decay=0.1, fused=True,
    )

    loader = make_dataloader("train", config.batch_size, config.seq_len)
    warmup = 100

    def lr_schedule(step):
        if step < warmup:
            return config.learning_rate * (step + 1) / warmup
        p = (step - warmup) / max(1, 5000 - warmup)
        return config.learning_rate * 0.1 + 0.5 * config.learning_rate * 0.9 * (1 + math.cos(math.pi * min(p, 1.0)))

    model.train()
    step = 0
    total_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    while True:
        elapsed = time.time() - t0
        if elapsed >= TIME_BUDGET:
            break

        lr = lr_schedule(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = next(loader)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += x.numel()
        step += 1

        if step % 100 == 0:
            tps = total_tokens / (time.time() - t0)
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {tps:.0f} tok/s | {elapsed:.0f}s")

    train_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6

    print("  Validating...")
    model.eval()
    val_bpb = compute_val_bpb(model, config.batch_size, device, config.seq_len)

    mfu = (6 * num_params * total_tokens) / (train_time * 312e12) * 100

    result = dict(
        name=config.name,
        params_M=round(num_params_M, 1),
        val_bpb=round(val_bpb, 6),
        train_loss=round(loss.item(), 4),
        vram_gb=round(peak_vram / 1024, 1),
        tokens_M=round(total_tokens / 1e6, 1),
        steps=step,
        mfu=round(mfu, 1),
        time_s=round(train_time, 0),
        arch=f"{config.n_embd}d/{config.n_layer}L/{config.n_head}h",
    )

    print(f"\n  RESULT: val_bpb={val_bpb:.6f}  params={num_params_M:.0f}M  "
          f"tokens={total_tokens/1e6:.0f}M  steps={step}  vram={peak_vram/1024:.0f}GB  mfu={mfu:.0f}%")

    del model, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tsv = "phase1_results.tsv"
    with open(tsv, "w") as f:
        f.write("name\tparams_M\tval_bpb\ttrain_loss\tvram_gb\ttokens_M\tsteps\tmfu%\ttime_s\tarch\n")

    print(f"Phase 1: Model Size Sweep  |  {len(PHASE1)} configs x {TIME_BUDGET}s each")
    print(f"Output: {tsv}\n")
    all_results = []
    t_total = time.time()

    for i, cfg in enumerate(PHASE1):
        print(f"\n[{i+1}/{len(PHASE1)}] >>>")
        try:
            r = run_experiment(cfg)
            all_results.append(r)
            with open(tsv, "a") as f:
                f.write(f"{r['name']}\t{r['params_M']}\t{r['val_bpb']}\t{r['train_loss']}\t"
                        f"{r['vram_gb']}\t{r['tokens_M']}\t{r['steps']}\t{r['mfu']}\t"
                        f"{r['time_s']}\t{r['arch']}\n")
        except Exception as exc:
            print(f"  CRASH: {exc}")
            with open(tsv, "a") as f:
                f.write(f"{cfg.name}\t0\t0\t0\t0\t0\t0\t0\t0\tCRASH\n")

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*70}")
    print(f"  PHASE 1 DONE — {elapsed:.1f} min total")
    print(f"{'='*70}\n")
    print(f"{'name':<20} {'params':>8} {'val_bpb':>10} {'tokens':>10} {'steps':>7} {'vram':>7}")
    print("-" * 65)

    best = None
    for r in all_results:
        tag = ""
        if best is None or r["val_bpb"] < best["val_bpb"]:
            best = r
        print(f"{r['name']:<20} {r['params_M']:>7.0f}M {r['val_bpb']:>10.6f} "
              f"{r['tokens_M']:>9.0f}M {r['steps']:>7} {r['vram_gb']:>6.0f}G")

    if best:
        print(f"\n>>> BEST: {best['name']} — val_bpb = {best['val_bpb']:.6f}")

    with open("phase1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
