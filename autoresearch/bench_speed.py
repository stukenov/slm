"""
Speed benchmark: test different optimizations for 900M Llama on A100 80GB.
Measures tokens/sec for each configuration over 30 seconds.
Usage: uv run bench_speed.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc, math, time, subprocess, sys
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from prepare import MAX_SEQ_LEN, VOCAB_SIZE, make_dataloader

@dataclass
class Cfg:
    vocab_size: int = VOCAB_SIZE; seq_len: int = MAX_SEQ_LEN
    n_layer: int = 24; n_head: int = 24; n_kv_head: int = 24
    n_embd: int = 1536; intermediate_size: int = 5376; tie_embeddings: bool = True

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :].to(torch.bfloat16))
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :].to(torch.bfloat16))
    def forward(self, seq_len):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]

def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head; self.n_kv_head = c.n_kv_head; self.hd = c.n_embd // c.n_head
        self.q = nn.Linear(c.n_embd, c.n_head * self.hd, bias=False)
        self.k = nn.Linear(c.n_embd, c.n_kv_head * self.hd, bias=False)
        self.v = nn.Linear(c.n_embd, c.n_kv_head * self.hd, bias=False)
        self.o = nn.Linear(c.n_head * self.hd, c.n_embd, bias=False)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        q = apply_rotary(q, cos, sin); k = apply_rotary(k, cos, sin)
        if self.n_kv_head < self.n_head:
            r = self.n_head // self.n_kv_head
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
        x = self.emb(idx); cos, sin = self.rot(idx.shape[1])
        for layer in self.layers: x = layer(x, cos, sin)
        return self.head(self.norm(x))

def bench(name, batch_size, use_compile, use_gqa=False, duration=30):
    """Run a 30-second benchmark, return tokens/sec and VRAM."""
    cfg = Cfg()
    if use_gqa: cfg.n_kv_head = 6
    model = Llama(cfg).to("cuda", dtype=torch.bfloat16)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    if use_compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.1, fused=True)
    loader = make_dataloader("train", batch_size)
    model.train(); torch.cuda.reset_peak_memory_stats()
    # Warmup (3 steps)
    for _ in range(3):
        x, y = next(loader); x, y = x.to("cuda"), y.to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    # Timed run
    total_tok = 0; t0 = time.time()
    while time.time() - t0 < duration:
        x, y = next(loader); x, y = x.to("cuda"), y.to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        total_tok += x.numel()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tps = total_tok / elapsed
    vram = torch.cuda.max_memory_allocated() / 1e9
    mfu = (6 * nparams * 1e6 * total_tok) / (elapsed * 312e12) * 100
    print(f"  {name:<35} {tps:>8.0f} tok/s  {vram:>5.1f}GB  MFU={mfu:.0f}%  loss={loss.item():.2f}")
    del model, opt, loader; gc.collect(); torch.cuda.empty_cache()
    return dict(name=name, tps=tps, vram_gb=round(vram,1), mfu=round(mfu,1), params_M=round(nparams,1))

def main():
    print("=" * 75)
    print("  Speed Benchmark: 900M Llama on A100 80GB (30s each)")
    print("=" * 75)
    # Check triton
    has_triton = False
    try:
        import triton; has_triton = True
        print(f"  Triton: {triton.__version__}")
    except ImportError:
        print("  Triton: NOT INSTALLED — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "triton==3.0.0", "-q"])
        try:
            import triton; has_triton = True
            print(f"  Triton: {triton.__version__} (just installed)")
        except:
            print("  Triton: FAILED to install")

    results = []
    print(f"\n{'name':<35} {'tok/s':>10} {'VRAM':>7} {'MFU':>6}  {'loss':>6}")
    print("-" * 75)

    # 1. Baseline: no compile, BS=4
    results.append(bench("baseline_bs4", 4, False))

    # 2. BS=16 no compile
    results.append(bench("bs16_no_compile", 16, False))

    # 3. BS=16 + compile (if triton available)
    if has_triton:
        results.append(bench("bs16_compile", 16, True))

    # 4. BS=24 + GQA kv=6 no compile
    results.append(bench("bs24_gqa6", 24, False, use_gqa=True))

    # 5. BS=24 + GQA + compile
    if has_triton:
        results.append(bench("bs24_gqa6_compile", 24, True, use_gqa=True))

    # 6. Try BS=32 GQA
    try:
        results.append(bench("bs32_gqa6", 32, False, use_gqa=True))
    except RuntimeError as e:
        print(f"  {'bs32_gqa6':<35} OOM: {e}")

    # 7. BS=32 GQA + compile
    if has_triton:
        try:
            results.append(bench("bs32_gqa6_compile", 32, True, use_gqa=True))
        except RuntimeError as e:
            print(f"  {'bs32_gqa6_compile':<35} OOM: {e}")

    # Summary
    print(f"\n{'='*75}")
    print("  SUMMARY")
    print(f"{'='*75}")
    best = max(results, key=lambda r: r["tps"])
    for r in results:
        tag = " <-- FASTEST" if r is best else ""
        hours = 17.8e9 / r["tps"] / 3600
        cost = hours * 0.81  # current instance rate
        print(f"  {r['name']:<35} {r['tps']:>8.0f} tok/s  {hours:>5.1f}h for 17.8B  ~${cost:.0f}{tag}")
    print(f"\n  Fastest: {best['name']} — {best['tps']:.0f} tok/s")

if __name__ == "__main__":
    main()
