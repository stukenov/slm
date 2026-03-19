"""
Phase 2: Hyperparameter Optimization for 900M model (7 experiments x 5 min)
Focus: batch size, LR, GQA, schedule on A100 80GB.
Usage: uv run run_phase2.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc, json, math, time
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from prepare import MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, make_dataloader, compute_val_bpb

@dataclass
class LlamaConfig:
    name: str = ""; vocab_size: int = VOCAB_SIZE; seq_len: int = MAX_SEQ_LEN
    n_layer: int = 24; n_head: int = 24; n_kv_head: int = 24
    n_embd: int = 1536; intermediate_size: int = 5376; tie_embeddings: bool = True
    batch_size: int = 4; learning_rate: float = 3e-4
    warmup_steps: int = 100; schedule: str = "cosine"

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :].to(torch.bfloat16))
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :].to(torch.bfloat16))
    def forward(self, seq_len):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]

def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head; self.n_kv_head = config.n_kv_head
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
        q = apply_rotary(q, cos, sin); k = apply_rotary(k, cos, sin)
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, -1))

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
        self.ln1 = RMSNorm(config.n_embd); self.attn = Attention(config)
        self.ln2 = RMSNorm(config.n_embd); self.mlp = SwiGLU(config)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(config.n_embd // config.n_head, config.seq_len)
        if config.tie_embeddings: self.lm_head.weight = self.embed_tokens.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, idx):
        B, T = idx.shape; x = self.embed_tokens(idx)
        cos, sin = self.rotary(T)
        for layer in self.layers: x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))

# Phase 2 configs — all 900M, vary hparams
PHASE2 = [
    LlamaConfig(name="01_baseline_bs4_lr3e4", batch_size=4, learning_rate=3e-4),
    LlamaConfig(name="02_bs16_lr3e4", batch_size=16, learning_rate=3e-4),
    LlamaConfig(name="03_bs16_lr6e4", batch_size=16, learning_rate=6e-4, warmup_steps=150),
    LlamaConfig(name="04_bs16_lr1e3", batch_size=16, learning_rate=1e-3, warmup_steps=200),
    LlamaConfig(name="05_gqa6_bs24_lr6e4", n_kv_head=6, batch_size=24, learning_rate=6e-4, warmup_steps=150),
    LlamaConfig(name="06_bs16_lr6e4_wsd", batch_size=16, learning_rate=6e-4, warmup_steps=100, schedule="wsd"),
    LlamaConfig(name="07_gqa6_bs24_lr1e3_wsd", n_kv_head=6, batch_size=24, learning_rate=1e-3, warmup_steps=150, schedule="wsd"),
]

def make_lr_fn(config, total_steps=5000):
    lr, warmup = config.learning_rate, config.warmup_steps
    if config.schedule == "wsd":
        stable_end = int(total_steps * 0.8)
        def fn(step):
            if step < warmup: return lr * (step + 1) / warmup
            if step < stable_end: return lr
            return lr * (1 - 0.9 * (step - stable_end) / max(1, total_steps - stable_end))
        return fn
    def fn(step):
        if step < warmup: return lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * min(p, 1.0)))
    return fn

def run_experiment(config, device="cuda"):
    print(f"\n{'='*70}")
    print(f"  {config.name}: BS={config.batch_size} LR={config.learning_rate} kv={config.n_kv_head} {config.schedule}")
    print(f"{'='*70}")
    model = Llama(config).to(device=device, dtype=torch.bfloat16)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_M = num_params / 1e6
    print(f"  Params: {num_params_M:.1f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    loader = make_dataloader("train", config.batch_size, config.seq_len)
    lr_fn = make_lr_fn(config)
    model.train(); step = 0; total_tokens = 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed >= TIME_BUDGET: break
        lr = lr_fn(step)
        for pg in optimizer.param_groups: pg["lr"] = lr
        x, y = next(loader); x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        total_tokens += x.numel(); step += 1
        if step % 50 == 0:
            tps = total_tokens / (time.time() - t0)
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {tps:.0f} tok/s | {elapsed:.0f}s")
    train_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    print("  Validating...")
    model.eval()
    val_bpb = compute_val_bpb(model, min(config.batch_size, 8), device, config.seq_len)
    mfu = (6 * num_params * total_tokens) / (train_time * 312e12) * 100
    result = dict(name=config.name, params_M=round(num_params_M, 1), val_bpb=round(val_bpb, 6),
        train_loss=round(loss.item(), 4), vram_gb=round(peak_vram / 1024, 1),
        tokens_M=round(total_tokens / 1e6, 1), steps=step, mfu=round(mfu, 1),
        time_s=round(train_time, 0),
        desc=f"BS={config.batch_size} LR={config.learning_rate} kv={config.n_kv_head} {config.schedule}")
    print(f"\n  RESULT: val_bpb={val_bpb:.6f}  tokens={total_tokens/1e6:.0f}M  steps={step}  vram={peak_vram/1024:.0f}GB  mfu={mfu:.0f}%")
    del model, optimizer, loader; gc.collect(); torch.cuda.empty_cache()
    return result

def main():
    tsv = "phase2_results.tsv"
    with open(tsv, "w") as f:
        f.write("name\tparams_M\tval_bpb\ttrain_loss\tvram_gb\ttokens_M\tsteps\tmfu%\ttime_s\tdesc\n")
    print(f"Phase 2: 900M Hparam Sweep | {len(PHASE2)} configs x {TIME_BUDGET}s")
    all_results = []; t_total = time.time()
    for i, cfg in enumerate(PHASE2):
        print(f"\n[{i+1}/{len(PHASE2)}] >>>")
        try:
            r = run_experiment(cfg); all_results.append(r)
            with open(tsv, "a") as f:
                f.write(f"{r['name']}\t{r['params_M']}\t{r['val_bpb']}\t{r['train_loss']}\t{r['vram_gb']}\t{r['tokens_M']}\t{r['steps']}\t{r['mfu']}\t{r['time_s']}\t{r['desc']}\n")
        except Exception as exc:
            print(f"  CRASH: {exc}"); import traceback; traceback.print_exc()
            with open(tsv, "a") as f: f.write(f"{cfg.name}\t0\t0\t0\t0\t0\t0\t0\t0\tCRASH\n")
            gc.collect(); torch.cuda.empty_cache()
    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*70}\n  PHASE 2 DONE — {elapsed:.1f} min\n{'='*70}\n")
    print(f"{'name':<30} {'val_bpb':>10} {'tokens':>8} {'steps':>6} {'vram':>6} {'mfu':>5}")
    print("-" * 70)
    best = None
    for r in all_results:
        if r.get("val_bpb", 99) > 0 and (best is None or r["val_bpb"] < best["val_bpb"]): best = r
        tag = " <-- BEST" if r is best else ""
        print(f"{r['name']:<30} {r['val_bpb']:>10.6f} {r['tokens_M']:>7.0f}M {r['steps']:>6} {r['vram_gb']:>5.0f}G {r['mfu']:>4.0f}%{tag}")
    if best: print(f"\n>>> BEST: {best['name']} — val_bpb = {best['val_bpb']:.6f}")
    with open("phase2_results.json", "w") as f: json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
