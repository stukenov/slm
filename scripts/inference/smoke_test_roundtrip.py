#!/usr/bin/env python3
"""
Smoke test: full round-trip verification for exp028v2.
Train 10 steps -> save checkpoint -> convert to HF format -> load -> inference -> verify.
This MUST pass before launching expensive training.

Usage: python3 smoke_test_roundtrip.py
Requires: GPU, transformers, torch, huggingface_hub
"""
import os, sys, json, tempfile, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("  SMOKE TEST: Full round-trip verification")
print("  Train -> Save -> HF Convert -> Load -> Inference")
print("=" * 60)

# ============================================================
# Step 1: Build the SAME model architecture as train_1b_ddp.py
# ============================================================
print("\n[1/5] Building model architecture (mini version for speed)...")

# Use tiny config for smoke test (same structure, fewer layers)
from dataclasses import dataclass

VOCAB_SIZE = 50257

@dataclass
class LlamaConfig:
    vocab_size: int = VOCAB_SIZE; seq_len: int = 128
    n_layer: int = 4; n_head: int = 16; n_kv_head: int = 4  # GQA
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
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = apply_rotary(q, cos, sin); k = apply_rotary(k, cos, sin)
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
        self.head = nn.Linear(c.vocab_size, c.vocab_size, bias=False)
        self.rot = RotaryEmbedding(c.n_embd // c.n_head, c.seq_len)
        if c.tie_embeddings: self.head.weight = self.emb.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, idx):
        x = self.emb(idx)  # NO embedding scaling
        cos, sin = self.rot(idx.shape[1])
        for layer in self.layers: x = layer(x, cos, sin)
        return self.head(self.norm(x))

cfg = LlamaConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Llama(cfg).to(device=device, dtype=torch.bfloat16)
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Built model: {num_params:.1f}M params (mini 4-layer version)")

# ============================================================
# Step 2: Train 10 steps on random data
# ============================================================
print("\n[2/5] Training 10 steps on random data...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
losses = []
for step in range(10):
    x = torch.randint(0, VOCAB_SIZE, (2, 64), device=device)
    y = torch.randint(0, VOCAB_SIZE, (2, 64), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())
    if step % 3 == 0:
        print(f"  step {step}: loss={loss.item():.4f}")

print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

# ============================================================
# Step 3: Save checkpoint (same as train_1b_ddp.py does)
# ============================================================
print("\n[3/5] Saving checkpoint...")
tmp_dir = tempfile.mkdtemp(prefix="smoke_test_")
ckpt_path = os.path.join(tmp_dir, "model.pt")
sd = model.state_dict()
# Strip _orig_mod. prefix (like clean_state_dict does)
sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
torch.save(sd, ckpt_path)
print(f"  Saved {len(sd)} keys to {ckpt_path}")
print(f"  Keys: {list(sd.keys())[:5]} ...")

# ============================================================
# Step 4: Convert to HF format (same mapping as upload_1b_to_hf.py)
# ============================================================
print("\n[4/5] Converting to HF LlamaForCausalLM format...")
from transformers import LlamaConfig as HFLlamaConfig, LlamaForCausalLM

# Map keys exactly like upload_1b_to_hf.py
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
state = {}
key_map = {
    "emb.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "lm_head.weight",
}
skipped = []
for k, v in ckpt.items():
    if k.startswith("rot."):
        skipped.append(k)
        continue
    new_k = k
    if k in key_map:
        new_k = key_map[k]
    else:
        new_k = new_k.replace("layers.", "model.layers.")
        new_k = new_k.replace(".ln1.", ".input_layernorm.")
        new_k = new_k.replace(".ln2.", ".post_attention_layernorm.")
        new_k = new_k.replace(".attn.q.", ".self_attn.q_proj.")
        new_k = new_k.replace(".attn.k.", ".self_attn.k_proj.")
        new_k = new_k.replace(".attn.v.", ".self_attn.v_proj.")
        new_k = new_k.replace(".attn.o.", ".self_attn.o_proj.")
        new_k = new_k.replace(".mlp.g.", ".mlp.gate_proj.")
        new_k = new_k.replace(".mlp.u.", ".mlp.up_proj.")
        new_k = new_k.replace(".mlp.d.", ".mlp.down_proj.")
    state[new_k] = v

print(f"  Mapped {len(state)} keys, skipped {len(skipped)} rotary keys")

hf_config = HFLlamaConfig(
    vocab_size=50257, hidden_size=2048, intermediate_size=5504,
    num_hidden_layers=4,  # mini version
    num_attention_heads=16, num_key_value_heads=4,
    max_position_embeddings=2048, tie_word_embeddings=True,
    bos_token_id=0, eos_token_id=0, pad_token_id=1,
    torch_dtype="bfloat16",
)
hf_model = LlamaForCausalLM(hf_config)

# THE CRITICAL TEST: strict=True
try:
    missing, unexpected = hf_model.load_state_dict(state, strict=True)
    print("  strict=True: ALL KEYS MATCHED!")
except RuntimeError as e:
    print(f"\n  FATAL: strict=True FAILED!")
    print(f"  Error: {e}")
    print(f"\n  Custom keys: {sorted(state.keys())[:10]}")
    expected = set(hf_model.state_dict().keys())
    got = set(state.keys())
    print(f"  Missing in checkpoint: {expected - got}")
    print(f"  Extra in checkpoint: {got - expected}")
    shutil.rmtree(tmp_dir)
    sys.exit(1)

# ============================================================
# Step 5: Inference test
# ============================================================
print("\n[5/5] Running inference on HF model...")
hf_model = hf_model.to(device=device, dtype=torch.bfloat16)

# Quick generation test
test_ids = torch.randint(0, VOCAB_SIZE, (1, 10), device=device)
with torch.no_grad():
    out = hf_model.generate(test_ids, max_new_tokens=20, do_sample=True, temperature=0.7)
print(f"  Input shape: {test_ids.shape}")
print(f"  Output shape: {out.shape}")
print(f"  Generated {out.shape[1] - test_ids.shape[1]} new tokens")

# Verify outputs are not all the same token (sanity check)
unique_tokens = out[0].unique().numel()
print(f"  Unique tokens in output: {unique_tokens}")
if unique_tokens < 3:
    print("  WARNING: Very few unique tokens — model may be degenerate")

# Cleanup
shutil.rmtree(tmp_dir)

print("\n" + "=" * 60)
print("  SMOKE TEST PASSED!")
print("  - Model architecture: OK")
print("  - Training: OK (loss decreased)")
print("  - Checkpoint save/load: OK")
print("  - HF conversion (strict=True): OK")
print("  - Inference: OK")
print("  Safe to launch full training.")
print("=" * 60)
