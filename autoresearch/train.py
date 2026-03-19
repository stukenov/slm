"""
Autoresearch Kazakh pretraining script. Single-GPU, single-file.
Llama architecture for Kazakh language modeling on A100 80GB.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
    vocab_size: int = VOCAB_SIZE
    seq_len: int = MAX_SEQ_LEN
    n_layer: int = 24
    n_head: int = 24
    n_kv_head: int = 24
    n_embd: int = 1536
    intermediate_size: int = 5376  # 3.5 * n_embd
    tie_embeddings: bool = True


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

        # GQA: expand kv heads
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

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = self.rotary(T)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    device = "cuda"
    compile_model = True

    # Model config (exp019: Llama 897M for Kazakh)
    config = LlamaConfig()

    # Training hyperparameters
    batch_size = 4
    learning_rate = 3e-4
    weight_decay = 0.1
    warmup_steps = 100
    max_grad_norm = 1.0

    # Create model
    model = Llama(config).to(device=device, dtype=torch.bfloat16)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_M = num_params / 1e6
    print(f"Model: {num_params_M:.1f}M params, {config.n_layer}L/{config.n_embd}d/{config.n_head}h")

    if compile_model:
        model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )

    # Dataloader
    train_loader = make_dataloader("train", batch_size, config.seq_len)

    # LR schedule: linear warmup + cosine decay to 10%
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, 5000 - warmup_steps)
        return learning_rate * 0.1 + 0.5 * learning_rate * 0.9 * (1 + math.cos(math.pi * min(progress, 1.0)))

    # Training loop (time-budgeted)
    model.train()
    step = 0
    total_tokens = 0

    print(f"Training for {TIME_BUDGET}s on {device}...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    while True:
        elapsed = time.time() - t0
        if elapsed >= TIME_BUDGET:
            break

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = next(train_loader)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += x.numel()
        step += 1

        if step % 50 == 0:
            tps = total_tokens / (time.time() - t0)
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {tps:.0f} tok/s | {elapsed:.0f}s")

    training_seconds = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6

    # Final validation
    print("Running validation...")
    t_val = time.time()
    model.eval()
    val_bpb = compute_val_bpb(model, batch_size, device, config.seq_len)
    val_seconds = time.time() - t_val
    total_seconds = training_seconds + val_seconds

    # MFU estimate (A100 bf16 peak = 312 TFLOPS)
    flops = 6 * num_params * total_tokens
    mfu = flops / (training_seconds * 312e12) * 100

    # Print results (autoresearch format)
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"mfu_percent:      {mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params_M:.1f}")
    print(f"depth:            {config.n_layer}")


if __name__ == "__main__":
    main()
