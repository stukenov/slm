"""
Optimized 150M Kazakh LM training script.
Incorporates Parameter Golf techniques adapted for Kazakh pretraining:
- LeakyReLU(a)^2 activation (replaces SwiGLU)
- XSA (Exclusive Self-Attention) on deep layers
- Partial RoPE (16/64 dims)
- LN Scale (1/sqrt(i+1) per layer)
- U-Net skip connections
- SmearGate + BigramHash embeddings
- Value Residual Learning
- Gated Attention (per-head sigmoid gates)
- Muon optimizer for matrix params
- EMA weight averaging

Architecture: ~130M params (768d, 18L, 12H/4KV GQA, 3x MLP)
Dataset: stukenov/sozkz-corpus-tokenized-kk-llama50k-v3
Tokenizer: 50257 vocab (llama50k)

Usage:
  Single GPU:  python train_150m_optimized.py
  Multi-GPU:   torchrun --standalone --nproc_per_node=2 train_150m_optimized.py
"""
from __future__ import annotations
import math
import os
import sys
import time
import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func
    _HAS_FA = True
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        _HAS_FA = True
    except ImportError:
        _HAS_FA = False
        flash_attn_func = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    hf_repo: str = "stukenov/sozkz-corpus-tokenized-kk-llama50k-v3"
    data_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-kazakh", "data")
    num_train_shards: int = 44
    num_val_shards: int = 1

    # Model architecture
    vocab_size: int = 50257
    n_layer: int = 18
    n_head: int = 12
    n_kv_head: int = 4  # GQA
    n_embd: int = 768
    mlp_mult: float = 3.0
    seq_len: int = 1024
    tie_embeddings: bool = True
    logit_softcap: float = 30.0

    # Parameter Golf techniques
    leaky_alpha: float = 0.5  # LeakyReLU(a)^2 — try 0.9 too
    xsa_last_n: int = 6  # XSA on last N layers
    rope_dims: int = 16  # Partial RoPE (16 of 64 head dims)
    ln_scale: bool = True  # 1/sqrt(i+1) per layer
    bigram_vocab_size: int = 2048  # BigramHash embedding size
    bigram_dim: int = 64
    gated_attention: bool = True
    value_residual: bool = True
    recur_layers: list[int] = field(default_factory=list)  # e.g. [7,8] for depth recurrence

    # Training
    batch_tokens: int = 131_072  # ~128K tokens per step
    learning_rate: float = 3e-4
    matrix_lr: float = 0.02  # Muon LR for matrix params
    scalar_lr: float = 0.02
    embed_lr: float = 0.6
    muon_momentum: float = 0.95
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500
    warmup_steps: int = 500
    weight_decay: float = 0.04
    grad_clip: float = 0.3
    beta1: float = 0.9
    beta2: float = 0.95
    max_steps: int = 50_000
    warmdown_steps: int = 5000
    ema_decay: float = 0.997
    qk_gain_init: float = 1.5

    # Logging & saving
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 5000
    output_dir: str = "./outputs/exp_150m_optimized"
    seed: int = 42

    # HuggingFace
    hf_model_name: str = "stukenov/sozkz-core-llama-150m-kk-optimized-v1"


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization (for Muon)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix parameters."""
    def __init__(self, params, lr: float, momentum: float, backend_steps: int = 5,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    # Fallback to SGD for 1D params
                    p.data.add_(g, alpha=-lr)
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                g_ns = zeropower_via_newtonschulz5(g, steps=ns_steps)
                p.data.add_(g_ns.to(dtype=p.dtype), alpha=-lr)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def prepare_data(cfg: Config):
    """Download and cache tokenized data from HF."""
    train_bin = os.path.join(cfg.data_dir, "train.bin")
    val_bin = os.path.join(cfg.data_dir, "val.bin")

    if os.path.exists(train_bin) and os.path.exists(val_bin):
        train_tok = os.path.getsize(train_bin) // 2
        val_tok = os.path.getsize(val_bin) // 2
        print(f"Data cached: train={train_tok:,} val={val_tok:,}")
        return train_bin, val_bin

    os.makedirs(cfg.data_dir, exist_ok=True)
    shard_dir = os.path.join(cfg.data_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    from huggingface_hub import HfApi
    import pyarrow.parquet as pq
    import requests

    api = HfApi()
    files = api.list_repo_files(cfg.hf_repo, repo_type="dataset")
    train_files = sorted([f for f in files if "/train/" in f and f.endswith(".parquet")])
    val_files = sorted([f for f in files if "/validation/" in f and f.endswith(".parquet")])
    if not train_files:
        train_files = sorted([f for f in files if f.startswith("data/train-") and f.endswith(".parquet")])
    if not val_files:
        val_files = sorted([f for f in files if f.startswith("data/validation-") and f.endswith(".parquet")])

    def download_shard(shard_file, idx, total):
        local = os.path.join(shard_dir, f"shard_{idx:04d}.npy")
        if os.path.exists(local):
            return np.load(local)
        url = f"https://huggingface.co/datasets/{cfg.hf_repo}/resolve/main/{shard_file}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        import io
        table = pq.read_table(io.BytesIO(resp.content))
        tokens = np.concatenate([np.array(row, dtype=np.uint16) for row in table["input_ids"].to_pylist()])
        np.save(local, tokens)
        print(f"  [{idx+1}/{total}] downloaded: {len(tokens):,} tokens")
        return tokens

    print(f"Downloading {len(train_files[:cfg.num_train_shards])} train shards...")
    train_arrs = [download_shard(f, i, cfg.num_train_shards)
                  for i, f in enumerate(train_files[:cfg.num_train_shards])]
    train_all = np.concatenate(train_arrs)
    train_all.tofile(train_bin)
    print(f"Train: {len(train_all):,} tokens -> {train_bin}")

    print(f"Downloading {len(val_files[:cfg.num_val_shards])} val shards...")
    val_arrs = [download_shard(f, i, cfg.num_val_shards)
                for i, f in enumerate(val_files[:cfg.num_val_shards])]
    val_all = np.concatenate(val_arrs)
    val_all.tofile(val_bin)
    print(f"Val: {len(val_all):,} tokens -> {val_bin}")

    return train_bin, val_bin


class TokenLoader:
    """Stream tokens from a flat binary file."""
    def __init__(self, bin_path: str, rank: int = 0, world_size: int = 1, device: torch.device = None):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device("cuda")
        self.pos = rank  # stagger starting positions

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        local_tokens = batch_tokens // self.world_size
        n_seqs = local_tokens // seq_len
        total_need = n_seqs * seq_len + 1
        if self.pos + total_need > self.n_tokens:
            self.pos = self.rank
        chunk = torch.from_numpy(self.data[self.pos:self.pos + total_need].astype(np.int64))
        self.pos += total_need * self.world_size
        x = chunk[:-1].reshape(n_seqs, seq_len).to(self.device, non_blocking=True)
        y = chunk[1:].reshape(n_seqs, seq_len).to(self.device, non_blocking=True)
        return x, y


# ---------------------------------------------------------------------------
# Transformer modules (Parameter Golf optimized)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int | None = None, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: tuple[Tensor, Tensor] | None = None
        self._cached_len: int = 0

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cache is None or self._cached_len != seq_len or self._cache[0].device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :])
            self._cached_len = seq_len
        return self._cache[0].to(dtype=dtype), self._cache[1].to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int = 0):
        super().__init__()
        dim = cfg.n_embd
        self.num_heads = cfg.n_head
        self.num_kv_heads = cfg.n_kv_head
        self.head_dim = dim // cfg.n_head
        self.q_proj = nn.Linear(dim, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((cfg.n_head,), cfg.qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, rope_dims=cfg.rope_dims)
        self.rope_dims = cfg.rope_dims
        self.use_xsa = False  # set by GPT.__init__
        # Gated attention
        self.gated_attention = cfg.gated_attention
        if cfg.gated_attention:
            self.attn_gate = nn.Linear(dim, cfg.n_head, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        # Value residual
        self.value_residual = cfg.value_residual
        if cfg.value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, x: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if _HAS_FA:
            y = flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16),
                                v.to(torch.bfloat16), causal=True).to(q.dtype)
        else:
            qt = q.transpose(1, 2)
            kt = k.transpose(1, 2)
            vt = v.transpose(1, 2)
            if kt.size(1) != qt.size(1):
                reps = qt.size(1) // kt.size(1)
                kt = kt.repeat_interleave(reps, dim=1)
                vt = vt.repeat_interleave(reps, dim=1)
            y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=1.0).transpose(1, 2)
        # XSA: subtract self-value projection
        if self.use_xsa:
            B, T, H, D = y.shape
            Hkv = v.size(-2)
            group = H // Hkv
            y_g = y.reshape(B, T, Hkv, group, D)
            vn = F.normalize(v, dim=-1).unsqueeze(-2)
            proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
            y = (y_g - proj).reshape(B, T, H, D)
        # Gated attention
        if self.gated_attention:
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y = y * gate
        y = y.reshape(bsz, seqlen, dim)
        return self.o_proj(y), raw_v


class MLP(nn.Module):
    """LeakyReLU(a)^2 MLP from Parameter Golf."""
    def __init__(self, cfg: Config):
        super().__init__()
        mlp_dim = int(cfg.mlp_mult * cfg.n_embd)
        self.up_proj = nn.Linear(cfg.n_embd, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, cfg.n_embd, bias=False)
        self.alpha = cfg.leaky_alpha

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.up_proj(x), negative_slope=self.alpha)
        return self.down_proj(x.square())


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = nn.Linear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        dim = cfg.n_embd
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if cfg.ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, raw_v = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v0=v0)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out, raw_v


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.bigram = BigramHashEmbedding(cfg.bigram_vocab_size, cfg.bigram_dim, cfg.n_embd) if cfg.bigram_vocab_size > 0 else None
        self.smear = SmearGate(cfg.n_embd)
        # Depth recurrence
        self.recur_layers = sorted(cfg.recur_layers) if cfg.recur_layers else []
        if self.recur_layers:
            cutoff = max(self.recur_layers) + 1
            self.v2p = list(range(cutoff)) + self.recur_layers + list(range(cutoff, cfg.n_layer))
            virtual_n = cfg.n_layer + len(self.recur_layers)
        else:
            self.v2p = list(range(cfg.n_layer))
            virtual_n = cfg.n_layer
        self.virtual_n = virtual_n
        self.n_enc = virtual_n // 2
        self.n_dec = virtual_n - self.n_enc
        self.n_skip = min(self.n_enc, self.n_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.n_skip, cfg.n_embd, dtype=torch.float32))
        # Blocks
        self.blocks = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(virtual_n)])
        # XSA on last N layers
        if cfg.xsa_last_n > 0:
            for i in range(max(0, virtual_n - cfg.xsa_last_n), virtual_n):
                self.blocks[i].attn.use_xsa = True
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.logit_softcap = cfg.logit_softcap
        self.value_residual = cfg.value_residual
        self._init_weights()

    def _init_weights(self):
        if self.cfg.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        n = self.cfg.n_layer
        proj_scale = 1.0 / math.sqrt(2 * n)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
        for block in self.blocks:
            if hasattr(block.attn, 'o_proj'):
                block.attn.o_proj.weight.data.mul_(proj_scale)
            block.mlp.down_proj.weight.data.mul_(proj_scale)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        for i in range(self.n_enc):
            x, raw_v = self.blocks[i](x, x0, v0=v0)
            if v0 is None and raw_v is not None and self.value_residual:
                v0 = raw_v
            skips.append(x)
        for i in range(self.n_dec):
            bi = self.n_enc + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, _ = self.blocks[bi](x, x0, v0=v0)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if target_ids is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   target_ids.reshape(-1), reduction="mean")
            return logits, loss
        return logits


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: TokenLoader, cfg: Config,
             device: torch.device, max_tokens: int = 10_000_000) -> float:
    model_to_eval = model.module if hasattr(model, 'module') else model
    model_to_eval.eval()
    total_loss = 0.0
    total_tokens = 0
    while total_tokens < max_tokens:
        x, y = val_loader.next_batch(cfg.batch_tokens, cfg.seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model_to_eval(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    bpb = avg_loss / math.log(2.0)
    model_to_eval.train()
    return bpb


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    cfg = Config()

    # Override from env
    for attr in ['n_layer', 'n_head', 'n_kv_head', 'n_embd', 'seq_len', 'max_steps',
                 'batch_tokens', 'seed', 'log_every', 'eval_every', 'save_every',
                 'warmup_steps', 'warmdown_steps', 'num_train_shards', 'bigram_vocab_size']:
        env_val = os.environ.get(attr.upper())
        if env_val is not None:
            setattr(cfg, attr, int(env_val))
    for attr in ['learning_rate', 'matrix_lr', 'scalar_lr', 'embed_lr', 'weight_decay',
                 'grad_clip', 'leaky_alpha', 'mlp_mult', 'ema_decay', 'logit_softcap',
                 'muon_momentum']:
        env_val = os.environ.get(attr.upper())
        if env_val is not None:
            setattr(cfg, attr, float(env_val))
    for attr in ['ln_scale', 'gated_attention', 'value_residual', 'tie_embeddings']:
        env_val = os.environ.get(attr.upper())
        if env_val is not None:
            setattr(cfg, attr, bool(int(env_val)))
    env_xsa = os.environ.get("XSA_LAST_N")
    if env_xsa:
        cfg.xsa_last_n = int(env_xsa)
    env_recur = os.environ.get("RECUR_LAYERS")
    if env_recur:
        cfg.recur_layers = [int(x) for x in env_recur.split(",") if x.strip()]
    env_out = os.environ.get("OUTPUT_DIR")
    if env_out:
        cfg.output_dir = env_out

    # DDP setup
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl")
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Data
    if master:
        train_bin, val_bin = prepare_data(cfg)
    if distributed:
        dist.barrier()
    train_bin = os.path.join(cfg.data_dir, "train.bin")
    val_bin = os.path.join(cfg.data_dir, "val.bin")
    train_loader = TokenLoader(train_bin, rank, world_size, device)
    val_loader = TokenLoader(val_bin, rank, world_size, device)

    # Model
    model = GPT(cfg).to(device=device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    if master:
        print(f"Model: {n_params/1e6:.1f}M params | {cfg.n_layer}L {cfg.n_embd}d {cfg.n_head}h/{cfg.n_kv_head}kv")
        print(f"  LeakyReLU(a={cfg.leaky_alpha})^2 | XSA last {cfg.xsa_last_n} | RoPE {cfg.rope_dims}/{cfg.n_embd // cfg.n_head}")
        print(f"  LN Scale={cfg.ln_scale} | GatedAttn={cfg.gated_attention} | VRL={cfg.value_residual}")
        print(f"  BigramHash({cfg.bigram_vocab_size}x{cfg.bigram_dim}) | Softcap={cfg.logit_softcap}")
        if cfg.recur_layers:
            print(f"  Depth Recurrence: layers {cfg.recur_layers} -> {model.virtual_n} virtual")
        print(f"  Batch: {cfg.batch_tokens:,} tokens | Steps: {cfg.max_steps:,} | Warmdown: {cfg.warmdown_steps}")
        print(f"  Data: {train_loader.n_tokens:,} train tokens")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    base_model = model.module if distributed else model

    # Optimizer: Muon for matrix params, AdamW for embeddings/scalars
    matrix_params = []
    scalar_params = []
    embed_params = []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if 'tok_emb' in name or ('lm_head' in name and not cfg.tie_embeddings):
            embed_params.append(p)
        elif 'bigram' in name and 'embed' in name:
            embed_params.append(p)
        elif p.ndim >= 2 and p.shape[0] >= 64 and p.shape[1] >= 64:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    optimizer_muon = Muon(
        [{"params": matrix_params, "lr": cfg.matrix_lr, "base_lr": cfg.matrix_lr}],
        lr=cfg.matrix_lr, momentum=cfg.muon_momentum, weight_decay=cfg.weight_decay,
    )
    optimizer_adam = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": cfg.embed_lr, "base_lr": cfg.embed_lr},
            {"params": scalar_params, "lr": cfg.scalar_lr, "base_lr": cfg.scalar_lr},
        ],
        betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay, fused=True,
    )
    optimizers = [optimizer_muon, optimizer_adam]

    if master:
        print(f"  Muon params: {sum(p.numel() for p in matrix_params)/1e6:.1f}M | "
              f"Adam params: {sum(p.numel() for p in embed_params + scalar_params)/1e6:.1f}M")

    # EMA
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}

    # LR schedule
    def lr_scale(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        warmdown_start = cfg.max_steps - cfg.warmdown_steps
        if step >= warmdown_start:
            return max((cfg.max_steps - step) / cfg.warmdown_steps, 0.0)
        return 1.0

    # Training loop
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.train()
    t0 = time.perf_counter()
    best_bpb = float("inf")

    for step in range(cfg.max_steps):
        scale = lr_scale(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        frac = min(step / cfg.muon_momentum_warmup_steps, 1.0) if cfg.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * cfg.muon_momentum_warmup_start + frac * cfg.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        x, y = train_loader.next_batch(cfg.batch_tokens, cfg.seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()

        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(cfg.ema_decay).add_(t.detach().float(), alpha=1.0 - cfg.ema_decay)

        if master and (step % cfg.log_every == 0 or step == cfg.max_steps - 1):
            elapsed = time.perf_counter() - t0
            tps = (step + 1) * cfg.batch_tokens / elapsed
            print(f"step {step:6d}/{cfg.max_steps} | loss {loss.item():.4f} | "
                  f"lr {scale * cfg.matrix_lr:.2e} | {tps:.0f} tok/s | {elapsed:.0f}s")

        if master and cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
            orig_state = {k: v.clone() for k, v in base_model.state_dict().items()}
            ema_applied = {name: t.to(dtype=orig_state[name].dtype) for name, t in ema_state.items()}
            base_model.load_state_dict(ema_applied, strict=True)
            bpb = evaluate(model, val_loader, cfg, device)
            print(f"  EVAL step {step+1} | val_bpb: {bpb:.4f}")
            if bpb < best_bpb:
                best_bpb = bpb
                torch.save(base_model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
                print(f"  -> NEW BEST: {bpb:.4f}")
            base_model.load_state_dict(orig_state, strict=True)
            model.train()

        if master and cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            ckpt = {
                "step": step + 1,
                "model": base_model.state_dict(),
                "ema": ema_state,
                "config": {k: v for k, v in cfg.__dict__.items() if not callable(v)},
            }
            torch.save(ckpt, os.path.join(cfg.output_dir, f"checkpoint_{step+1}.pt"))
            print(f"  Saved checkpoint at step {step+1}")

    # Final eval with EMA
    if master:
        ema_applied = {name: t.to(dtype=base_model.state_dict()[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(ema_applied, strict=True)
        final_bpb = evaluate(model, val_loader, cfg, device)
        print(f"\nFinal val_bpb (EMA): {final_bpb:.4f}")
        print(f"Best val_bpb: {best_bpb:.4f}")
        torch.save(base_model.state_dict(), os.path.join(cfg.output_dir, "final_model.pt"))
        n_params = sum(p.numel() for p in base_model.parameters())
        print(f"\nModel: {n_params/1e6:.1f}M params")
        print(f"Architecture: {cfg.n_layer}L {cfg.n_embd}d {cfg.n_head}h/{cfg.n_kv_head}kv")
        print(f"Techniques: LeakyReLU({cfg.leaky_alpha})^2, XSA-{cfg.xsa_last_n}, PartialRoPE-{cfg.rope_dims}, "
              f"LNScale, GatedAttn, VRL, SmearGate, BigramHash({cfg.bigram_vocab_size}), "
              f"U-Net, Muon, EMA({cfg.ema_decay})")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
