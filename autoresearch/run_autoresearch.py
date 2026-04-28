"""
Autonomous autoresearch runner for Kazakh LM optimization.
Runs ~96 experiments of 5 minutes each over 8 hours.
After each experiment, uploads results to HuggingFace for persistence.

Usage: python run_autoresearch.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HF_RESULTS_REPO = "stukenov/autoresearch-kazakh-logs"
RESULTS_FILE = "results.tsv"
MAX_TOTAL_HOURS = 8
RUN_TIMEOUT = 660  # 11 min max per run (5 min train + startup + eval)

def upload_to_hf(files: list[str], commit_msg: str):
    """Upload files to HF dataset repo for persistence."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.repo_info(HF_RESULTS_REPO, repo_type="dataset")
        except Exception:
            api.create_repo(HF_RESULTS_REPO, repo_type="dataset", private=True)
        for f in files:
            if os.path.exists(f):
                api.upload_file(
                    path_or_fileobj=f,
                    path_in_repo=os.path.basename(f),
                    repo_id=HF_RESULTS_REPO,
                    repo_type="dataset",
                    commit_message=commit_msg,
                )
        print(f"  [HF] Uploaded {len(files)} files: {commit_msg}")
    except Exception as e:
        print(f"  [HF] Upload failed (non-fatal): {e}")


def write_train_py(cfg: dict):
    """Generate train.py from experiment config."""
    n_layer = cfg.get("n_layer", 12)
    n_head = cfg.get("n_head", 12)
    n_kv_head = cfg.get("n_kv_head", n_head)
    n_embd = cfg.get("n_embd", 768)
    inter_size = cfg.get("intermediate_size", int(3.5 * n_embd))
    batch_size = cfg.get("batch_size", 8)
    lr = cfg.get("lr", 3e-4)
    wd = cfg.get("weight_decay", 0.1)
    warmup_frac = cfg.get("warmup_frac", 0.1)
    beta1 = cfg.get("beta1", 0.9)
    beta2 = cfg.get("beta2", 0.95)
    grad_clip = cfg.get("grad_clip", 1.0)
    tie_emb = cfg.get("tie_embeddings", True)
    activation = cfg.get("activation", "swiglu")  # swiglu or leaky_relu_sq
    leaky_alpha = cfg.get("leaky_alpha", 0.5)
    use_compile = cfg.get("use_compile", True)
    log_every = cfg.get("log_every", 50)

    act_code = ""
    if activation == "swiglu":
        act_code = """
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
"""
    elif activation == "leaky_relu_sq":
        act_code = f"""
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
    def forward(self, x):
        x = F.leaky_relu(self.up_proj(x), negative_slope={leaky_alpha})
        return self.down_proj(x * x)
"""

    compile_line = "model = torch.compile(model)" if use_compile else "# torch.compile disabled"

    code = f'''"""Auto-generated train.py — experiment config embedded."""
import math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, make_dataloader, compute_val_bpb

@torch.jit.script
def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class LlamaConfig:
    vocab_size = {cfg.get("vocab_size", 50257)}
    seq_len = {cfg.get("seq_len", 1024)}
    n_layer = {n_layer}
    n_head = {n_head}
    n_kv_head = {n_kv_head}
    n_embd = {n_embd}
    intermediate_size = {inter_size}
    tie_embeddings = {tie_emb}

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
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :].to(torch.bfloat16))
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :].to(torch.bfloat16))
    def forward(self, seq_len):
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]

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
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, -1))
{act_code}
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = self.rotary(T)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

def main():
    device = "cuda"
    config = LlamaConfig()
    batch_size = {batch_size}
    learning_rate = {lr}

    model = Llama(config).to(device=device, dtype=torch.bfloat16)
    num_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {{num_params_M:.1f}}M params, {{config.n_layer}}L/{{config.n_embd}}d/{{config.n_head}}h")
    {compile_line}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=({beta1}, {beta2}),
        weight_decay={wd},
        fused=True,
    )
    train_loader = make_dataloader("train", batch_size, config.seq_len)
    total_steps = 100_000
    warmup_steps = int({warmup_frac} * total_steps)
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return learning_rate * 0.1 + 0.9 * learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    step = 0
    t0 = time.perf_counter()
    model.train()
    while True:
        elapsed = time.perf_counter() - t0
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
        if {grad_clip} > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), {grad_clip})
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % {log_every} == 0 or step == 1:
            print(f"  step {{step:5d}} | loss {{loss.item():.4f}} | lr {{lr:.2e}} | {{elapsed:.0f}}s")

    training_seconds = time.perf_counter() - t0
    print(f"Training done: {{step}} steps in {{training_seconds:.1f}}s")
    val_bpb = compute_val_bpb(model, batch_size, device, config.seq_len)
    total_seconds = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    tokens_per_step = batch_size * config.seq_len
    total_tokens_M = step * tokens_per_step / 1e6
    A100_TFLOPS = 312e12
    params = sum(p.numel() for p in model.parameters())
    flops_per_step = 6 * params * tokens_per_step
    mfu = (flops_per_step * step / training_seconds) / A100_TFLOPS * 100
    print("---")
    print(f"val_bpb:          {{val_bpb:.6f}}")
    print(f"training_seconds: {{training_seconds:.1f}}")
    print(f"total_seconds:    {{total_seconds:.1f}}")
    print(f"peak_vram_mb:     {{peak_vram_mb:.1f}}")
    print(f"mfu_percent:      {{mfu:.2f}}")
    print(f"total_tokens_M:   {{total_tokens_M:.1f}}")
    print(f"num_steps:        {{step}}")
    print(f"num_params_M:     {{num_params_M:.1f}}")
    print(f"depth:            {{config.n_layer}}")

if __name__ == "__main__":
    main()
'''
    with open("train.py", "w") as f:
        f.write(code)


# ============================================================================
# Experiment definitions — 96 experiments in 7 phases
# ============================================================================

def build_experiments() -> list[dict]:
    exps = []
    exp_id = 0

    def add(desc, **kwargs):
        nonlocal exp_id
        exp_id += 1
        cfg = {"id": exp_id, "description": desc}
        cfg.update(kwargs)
        exps.append(cfg)

    # Phase 1: Baseline + Model size sweep (6 runs)
    add("baseline 897M (24L/1536d/24h)", n_layer=24, n_head=24, n_kv_head=24, n_embd=1536, intermediate_size=5376, batch_size=4, lr=3e-4)
    add("500M (20L/1280d/20h)", n_layer=20, n_head=20, n_kv_head=4, n_embd=1280, intermediate_size=4480, batch_size=8, lr=3e-4)
    add("300M (16L/1024d/16h)", n_layer=16, n_head=16, n_kv_head=4, n_embd=1024, intermediate_size=3584, batch_size=12, lr=3e-4)
    add("150M (18L/768d/12h)", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4)
    add("100M (12L/768d/12h)", n_layer=12, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=24, lr=3e-4)
    add("50M (8L/512d/8h)", n_layer=8, n_head=8, n_kv_head=4, n_embd=512, intermediate_size=1792, batch_size=32, lr=6e-4)

    # Phase 2: LR sweep on best sizes (9 runs — will run for top 3 sizes)
    for sz_name, sz in [("150M", {"n_layer": 18, "n_head": 12, "n_kv_head": 4, "n_embd": 768, "intermediate_size": 2688, "batch_size": 16}),
                         ("300M", {"n_layer": 16, "n_head": 16, "n_kv_head": 4, "n_embd": 1024, "intermediate_size": 3584, "batch_size": 12}),
                         ("100M", {"n_layer": 12, "n_head": 12, "n_kv_head": 4, "n_embd": 768, "intermediate_size": 2688, "batch_size": 24})]:
        for lr in [1e-4, 6e-4, 1e-3]:
            add(f"{sz_name} lr={lr}", lr=lr, **sz)

    # Phase 3: Batch size sweep (6 runs)
    for bs in [8, 12, 24, 32, 48, 64]:
        add(f"150M bs={bs}", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=bs, lr=3e-4)

    # Phase 4: Architecture variations (12 runs)
    # Width vs depth
    add("150M wide (12L/896d)", n_layer=12, n_head=14, n_kv_head=7, n_embd=896, intermediate_size=3136, batch_size=16, lr=3e-4)
    add("150M deep (24L/640d)", n_layer=24, n_head=10, n_kv_head=5, n_embd=640, intermediate_size=2240, batch_size=16, lr=3e-4)
    add("150M square (16L/768d)", n_layer=16, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4)
    # GQA variations
    add("150M GQA kv=2", n_layer=18, n_head=12, n_kv_head=2, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4)
    add("150M GQA kv=6", n_layer=18, n_head=12, n_kv_head=6, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4)
    add("150M MHA kv=12", n_layer=18, n_head=12, n_kv_head=12, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4)
    # MLP ratio
    add("150M mlp=2.5x", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=1920, batch_size=16, lr=3e-4)
    add("150M mlp=4x", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=3072, batch_size=16, lr=3e-4)
    # No tie embeddings
    add("150M no-tie", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, tie_embeddings=False)
    # Activation: LeakyReLU^2
    add("150M leaky_relu_sq a=0.5", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, activation="leaky_relu_sq", leaky_alpha=0.5)
    add("150M leaky_relu_sq a=0.9", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, activation="leaky_relu_sq", leaky_alpha=0.9)
    add("150M leaky_relu_sq a=0.1", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, activation="leaky_relu_sq", leaky_alpha=0.1)

    # Phase 5: Optimizer tuning (8 runs)
    for wd in [0.01, 0.05, 0.2]:
        add(f"150M wd={wd}", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, weight_decay=wd)
    for gc in [0.3, 0.5, 2.0]:
        add(f"150M grad_clip={gc}", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, grad_clip=gc)
    add("150M beta2=0.99", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, beta2=0.99)
    add("150M warmup=0.02", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, warmup_frac=0.02)

    # Phase 6: Scaling with best config (adaptive — use best from phases 1-5)
    # These use placeholder configs — will be overridden at runtime
    for sz_name, n_l, n_e, n_h, inter, bs in [
        ("200M", 14, 896, 14, 3136, 14),
        ("250M", 16, 960, 12, 3360, 12),
        ("350M", 18, 1024, 16, 3584, 10),
        ("400M", 20, 1024, 16, 3584, 8),
        ("175M", 20, 768, 12, 2688, 14),
        ("125M", 16, 704, 11, 2464, 18),
    ]:
        add(f"{sz_name} refined", n_layer=n_l, n_head=n_h, n_kv_head=max(2, n_h//3), n_embd=n_e, intermediate_size=inter, batch_size=bs, lr=3e-4)

    # Phase 7: torch.compile on/off + misc (fill remaining)
    add("150M no-compile", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=3e-4, use_compile=False)
    add("150M compile+lr=5e-4", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=5e-4)

    # Repeat experiments with best found config at different sizes (padding to ~96)
    for extra_lr in [2e-4, 4e-4, 8e-4]:
        add(f"300M lr={extra_lr}", n_layer=16, n_head=16, n_kv_head=4, n_embd=1024, intermediate_size=3584, batch_size=12, lr=extra_lr)
    for extra_lr in [2e-4, 4e-4, 8e-4]:
        add(f"500M lr={extra_lr}", n_layer=20, n_head=20, n_kv_head=4, n_embd=1280, intermediate_size=4480, batch_size=8, lr=extra_lr)
    # Fill to ~96
    add("300M leaky_sq a=0.5", n_layer=16, n_head=16, n_kv_head=4, n_embd=1024, intermediate_size=3584, batch_size=12, lr=3e-4, activation="leaky_relu_sq")
    add("300M leaky_sq a=0.9", n_layer=16, n_head=16, n_kv_head=4, n_embd=1024, intermediate_size=3584, batch_size=12, lr=3e-4, activation="leaky_relu_sq", leaky_alpha=0.9)
    add("500M leaky_sq a=0.5", n_layer=20, n_head=20, n_kv_head=4, n_embd=1280, intermediate_size=4480, batch_size=8, lr=3e-4, activation="leaky_relu_sq")
    add("100M leaky_sq lr=6e-4", n_layer=12, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=24, lr=6e-4, activation="leaky_relu_sq")
    add("150M deep 22L/704d", n_layer=22, n_head=11, n_kv_head=11, n_embd=704, intermediate_size=2464, batch_size=16, lr=3e-4)
    add("150M bs=16 lr=5e-4 wd=0.05", n_layer=18, n_head=12, n_kv_head=4, n_embd=768, intermediate_size=2688, batch_size=16, lr=5e-4, weight_decay=0.05)
    add("200M bs=12 lr=4e-4", n_layer=14, n_head=14, n_kv_head=7, n_embd=896, intermediate_size=3136, batch_size=12, lr=4e-4)
    add("300M GQA kv=2 lr=4e-4", n_layer=16, n_head=16, n_kv_head=2, n_embd=1024, intermediate_size=3584, batch_size=12, lr=4e-4)

    return exps


def run_experiment(exp: dict) -> dict:
    """Run one experiment and return results."""
    desc = exp["description"]
    exp_id = exp["id"]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {exp_id}: {desc}")
    print(f"{'='*60}")

    # Generate train.py
    write_train_py(exp)

    # Run
    t0 = time.time()
    try:
        result = subprocess.run(
            ["python", "train.py"],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )
        elapsed = time.time() - t0
        stdout = result.stdout
        stderr = result.stderr

        # Save full log
        log_file = f"logs/exp_{exp_id:03d}.log"
        with open(log_file, "w") as f:
            f.write(f"=== Experiment {exp_id}: {desc} ===\n")
            f.write(f"Config: {json.dumps({k:v for k,v in exp.items() if k not in ('id','description')}, default=str)}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)

        # Parse results
        val_bpb = 0.0
        peak_vram = 0.0
        num_steps = 0
        num_params = 0.0
        for line in stdout.split("\n"):
            if line.startswith("val_bpb:"):
                val_bpb = float(line.split()[-1])
            elif line.startswith("peak_vram_mb:"):
                peak_vram = float(line.split()[-1])
            elif line.startswith("num_steps:"):
                num_steps = int(line.split()[-1])
            elif line.startswith("num_params_M:"):
                num_params = float(line.split()[-1])

        if val_bpb > 0:
            status = "ok"
            print(f"  RESULT: val_bpb={val_bpb:.6f} | vram={peak_vram/1e3:.1f}GB | steps={num_steps} | params={num_params:.1f}M | {elapsed:.0f}s")
        else:
            status = "crash"
            print(f"  CRASHED after {elapsed:.0f}s")
            # Print last 20 lines of stderr
            for line in stderr.strip().split("\n")[-20:]:
                print(f"    {line}")

        return {
            "id": exp_id,
            "description": desc,
            "val_bpb": val_bpb,
            "peak_vram_gb": peak_vram / 1e3,
            "num_steps": num_steps,
            "num_params_M": num_params,
            "elapsed_s": elapsed,
            "status": status,
            "log_file": log_file,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT after {elapsed:.0f}s")
        return {
            "id": exp_id, "description": desc,
            "val_bpb": 0.0, "peak_vram_gb": 0.0,
            "num_steps": 0, "num_params_M": 0.0,
            "elapsed_s": elapsed, "status": "timeout",
            "log_file": "",
        }


def main():
    os.makedirs("logs", exist_ok=True)

    # Initialize results
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("id\tval_bpb\tmemory_gb\tsteps\tparams_M\telapsed_s\tstatus\tdescription\n")

    experiments = build_experiments()
    print(f"Total experiments planned: {len(experiments)}")
    print(f"Max runtime: {MAX_TOTAL_HOURS} hours")

    best_bpb = float("inf")
    best_desc = ""
    global_t0 = time.time()

    for exp in experiments:
        # Check time budget
        elapsed_hours = (time.time() - global_t0) / 3600
        if elapsed_hours >= MAX_TOTAL_HOURS:
            print(f"\n{'='*60}")
            print(f"TIME LIMIT REACHED: {elapsed_hours:.1f}h")
            break

        remaining = MAX_TOTAL_HOURS - elapsed_hours
        print(f"\n[Time: {elapsed_hours:.1f}h / {MAX_TOTAL_HOURS}h | Remaining: {remaining:.1f}h | Best: {best_bpb:.6f}]")

        result = run_experiment(exp)

        # Log to TSV
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{result['id']}\t{result['val_bpb']:.6f}\t{result['peak_vram_gb']:.1f}\t"
                    f"{result['num_steps']}\t{result['num_params_M']:.1f}\t{result['elapsed_s']:.0f}\t"
                    f"{result['status']}\t{result['description']}\n")

        # Track best
        if result["val_bpb"] > 0 and result["val_bpb"] < best_bpb:
            best_bpb = result["val_bpb"]
            best_desc = result["description"]
            print(f"  >>> NEW BEST: {best_bpb:.6f} ({best_desc})")

        # Upload to HF after each run
        upload_files = [RESULTS_FILE]
        if result.get("log_file") and os.path.exists(result["log_file"]):
            upload_files.append(result["log_file"])
        upload_to_hf(upload_files, f"exp {result['id']}: {result['description']} -> {result['val_bpb']:.6f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"Total time: {(time.time() - global_t0)/3600:.1f} hours")
    print(f"Experiments run: {len(experiments)}")
    print(f"Best val_bpb: {best_bpb:.6f} ({best_desc})")
    print(f"{'='*60}")

    # Final upload
    upload_to_hf([RESULTS_FILE], f"FINAL: best={best_bpb:.6f} ({best_desc})")


if __name__ == "__main__":
    main()
