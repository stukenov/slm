"""
Benchmark 300M Llama on RTX 5090. Tests eager vs compile, different BS.
Usage: uv run bench_5090.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc, math, time
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from prepare import MAX_SEQ_LEN, VOCAB_SIZE, make_dataloader

@dataclass
class Cfg:
    name: str = "300M"
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

def bench(name, bs, use_compile, dur=30):
    cfg = Cfg()
    model = Llama(cfg).to("cuda", dtype=torch.bfloat16)
    np_ = sum(p.numel() for p in model.parameters()) / 1e6
    if use_compile: model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.1, fused=True)
    ld = make_dataloader("train", bs)
    model.train(); torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(3):
        x, y = next(ld); x, y = x.to("cuda"), y.to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    # Bench
    tok = 0; t0 = time.time()
    while time.time() - t0 < dur:
        x, y = next(ld); x, y = x.to("cuda"), y.to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        tok += x.numel()
    torch.cuda.synchronize(); el = time.time() - t0
    tps = tok / el; vram = torch.cuda.max_memory_allocated() / 1e9
    # 5090 bf16 peak ~280 TFLOPS (estimate)
    mfu = (6 * np_ * 1e6 * tok) / (el * 280e12) * 100
    hrs_9b = 9e9 / tps / 3600
    print(f"  {name:<25} {tps:>9.0f} tok/s  {vram:>5.1f}GB  MFU={mfu:>4.0f}%  9B in {hrs_9b:>5.1f}h  loss={loss.item():.2f}")
    del model, opt, ld; gc.collect(); torch.cuda.empty_cache()
    return dict(name=name, tps=int(tps), vram=round(vram,1), mfu=round(mfu,1), hours_9b=round(hrs_9b,1))

def main():
    print("="*80)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    cap = torch.cuda.get_device_capability()
    print(f"  Compute capability: sm_{cap[0]}{cap[1]}")
    try:
        import triton; print(f"  Triton: {triton.__version__}")
    except: print("  Triton: not available")

    cfg = Cfg()
    model = Llama(cfg)
    np_ = sum(p.numel() for p in model.parameters()) / 1e6
    del model
    print(f"  Model: {cfg.name} ({np_:.1f}M params)")
    print(f"  Arch: {cfg.n_embd}d/{cfg.n_layer}L/{cfg.n_head}h, inter={cfg.intermediate_size}")
    print("="*80)

    print(f"\n  {'name':<25} {'tok/s':>11} {'VRAM':>7} {'MFU':>6} {'9B time':>9}")
    print("  " + "-"*65)
    R = []

    # Eager tests
    for bs in [8, 16, 32, 64]:
        try:
            R.append(bench(f"bs{bs}_eager", bs, False))
        except RuntimeError as e:
            print(f"  bs{bs}_eager               OOM: {e}")
            gc.collect(); torch.cuda.empty_cache()
            break

    # Compile tests
    for bs in [8, 16, 32, 64]:
        try:
            R.append(bench(f"bs{bs}_compile", bs, True))
        except RuntimeError as e:
            print(f"  bs{bs}_compile              OOM: {e}")
            gc.collect(); torch.cuda.empty_cache()
            break

    print(f"\n{'='*80}")
    if R:
        best = max(R, key=lambda r: r["tps"])
        print(f"  FASTEST: {best['name']} — {best['tps']:,} tok/s")
        print(f"  9B tokens in {best['hours_9b']}h on 1×RTX 5090")
        print(f"  Estimated 8×5090: {best['hours_9b']/8:.1f}h")
    import json
    with open("bench_5090_results.json", "w") as f: json.dump(R, f, indent=2)

if __name__ == "__main__":
    main()
