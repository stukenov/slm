"""
exp029: 1.08B Llama with GQA + Z-loss on ~16.2B Kazakh tokens, TPU via PyTorch/XLA FSDP.
Architecture: 2048h, 22L, 16 heads, 4 KV heads (GQA), 5504 inter, tied embeddings.
NO QK-Norm, NO embedding scaling — both are NOT HF-compatible (exp028 lesson).

FSDP (Fully Sharded Data Parallel) shards the model across all TPU chips.
Checkpoints are portable: consolidated state_dict saved to GCS, can resume on any chip count.

TPU/XLA COMPATIBILITY NOTES (torch_xla >= 2.7):
  - xm.xrt_world_size()    -> xr.world_size()          [removed in 2.7]
  - xm.get_ordinal()       -> xr.global_ordinal()      [removed in 2.7]
  - xm.is_master_ordinal() -> xr.global_ordinal() == 0 [removed in 2.7]
  - xm.reduce_gradients()  -> NOT needed with FSDP (handles its own reduce-scatter)
  - F.rms_norm              -> manual implementation     [not in XLA]
  - F.scaled_dot_product_attention -> manual attention   [unreliable on XLA]
  - torch.compile           -> skip (XLA has its own JIT)
  - AdamW(fused=True)       -> not supported on TPU
  - With FSDP: use optimizer.step(), NOT xm.optimizer_step()

Usage (single-host, e.g. v6e-4):
  PJRT_DEVICE=TPU python3 train_1b_tpu.py --max-steps 10
  PJRT_DEVICE=TPU python3 train_1b_tpu.py --resume gs://bucket/step_5000

Usage (multi-host pod, e.g. v6e-64):
  gcloud compute tpus tpu-vm ssh NODE --worker=all --command="
    cd ~ && PJRT_DEVICE=TPU python3 train_1b_tpu.py
  "
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")  # must be set before torch_xla import

import sys, math, time, json, signal, subprocess, glob, shutil
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_backend  # registers 'xla' backend

from torch_xla.distributed.fsdp import (
    XlaFullyShardedDataParallel as FSDP,
    checkpoint_module,
)
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

from prepare_1b import MAX_SEQ_LEN, VOCAB_SIZE, Dataloader

Z_LOSS_COEFF = 1e-4  # logit regularization (PaLM/Gemma)

# Graceful shutdown for spot preemption
_SHUTDOWN_REQUESTED = False
def _sigterm_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print("\n  !!! SIGTERM received — saving checkpoint and exiting gracefully !!!")
signal.signal(signal.SIGTERM, _sigterm_handler)

GCS_BUCKET = os.environ.get("GCS_CHECKPOINT_BUCKET", "gs://sozkz-trc-checkpoints/exp029")
LOCAL_CKPT_DIR = "/tmp/checkpoints/exp029_1b"


@dataclass
class LlamaConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = MAX_SEQ_LEN
    n_layer: int = 22
    n_head: int = 16
    n_kv_head: int = 4  # GQA: 4 KV heads
    n_embd: int = 2048
    intermediate_size: int = 5504
    tie_embeddings: bool = True


# ============================================================================
# Model components — all XLA-compatible (no CUDA-only ops)
# ============================================================================

class RMSNorm(nn.Module):
    """Manual RMSNorm — F.rms_norm is not available in XLA."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])

    def forward(self, T):
        return self.cos_cached[:, :, :T], self.sin_cached[:, :, :T]


def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    """Manual causal attention — F.scaled_dot_product_attention is unreliable on XLA."""
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_head, c.n_kv_head, c.n_embd // c.n_head
        self.scale = 1.0 / math.sqrt(self.hd)
        self.q = nn.Linear(c.n_embd, c.n_head * self.hd, bias=False)
        self.k = nn.Linear(c.n_embd, c.n_kv_head * self.hd, bias=False)
        self.v = nn.Linear(c.n_embd, c.n_kv_head * self.hd, bias=False)
        self.o = nn.Linear(c.n_head * self.hd, c.n_embd, bias=False)
        mask = torch.triu(torch.ones(c.seq_len, c.seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.nkv < self.nh:
            r = self.nh // self.nkv
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[None, None, :T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Linear(c.n_embd, c.intermediate_size, bias=False)
        self.u = nn.Linear(c.n_embd, c.intermediate_size, bias=False)
        self.d = nn.Linear(c.intermediate_size, c.n_embd, bias=False)

    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))


class Block(nn.Module):
    """Transformer block — FSDP wraps each Block as a sharding unit."""
    def __init__(self, c):
        super().__init__()
        self.ln1 = RMSNorm(c.n_embd)
        self.attn = Attention(c)
        self.ln2 = RMSNorm(c.n_embd)
        self.mlp = SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class Llama(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.norm = RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.rot = RotaryEmbedding(c.n_embd // c.n_head, c.seq_len)
        if c.tie_embeddings:
            self.head.weight = self.emb.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx):
        x = self.emb(idx)
        cos, sin = self.rot(idx.shape[1])
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.head(self.norm(x))


# ============================================================================
# FSDP Checkpoint utils — portable across different chip counts
# ============================================================================

def save_fsdp_checkpoint(model, optimizer, step, tokens, loss, lr, local_dir, gcs_path=None):
    """Save FSDP sharded checkpoint (each rank saves its shard) + consolidated on rank 0.

    Sharded checkpoints are fast but tied to world_size.
    Consolidated checkpoint is portable — can resume on any number of chips.
    """
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    os.makedirs(local_dir, exist_ok=True)

    # Each rank saves its own shard
    shard_ckpt = {
        'model': model.state_dict(),
        'shard_metadata': model.get_shard_metadata(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'tokens': tokens,
        'loss': loss,
        'lr': lr,
        'world_size': world_size,
    }
    shard_path = os.path.join(local_dir, f"rank-{rank}-of-{world_size}.pt")
    xm.save(shard_ckpt, shard_path, master_only=False)

    # Rank 0 also saves consolidated (portable) checkpoint
    if rank == 0:
        try:
            from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints
            full_sd, _ = consolidate_sharded_model_checkpoints(
                ckpt_prefix=os.path.join(local_dir, "rank-"),
                ckpt_suffix=f"*-of-{world_size}.pt",
            )
            consolidated_path = os.path.join(local_dir, "consolidated.pt")
            torch.save({
                'model': full_sd,
                'step': step,
                'tokens': tokens,
                'loss': loss,
                'lr': lr,
            }, consolidated_path)
            print(f"  >>> Consolidated checkpoint saved ({os.path.getsize(consolidated_path)/1e9:.1f}GB)")
        except Exception as e:
            print(f"  >>> Consolidation failed (non-fatal, shards OK): {e}")

        # Meta file for quick inspection
        meta = dict(step=step, tokens=tokens, loss=loss, lr=lr, world_size=world_size)
        with open(os.path.join(local_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        # Sync to GCS
        if gcs_path:
            try:
                subprocess.run(
                    ["gsutil", "-m", "cp", "-r", local_dir, gcs_path],
                    timeout=300, capture_output=True
                )
                print(f"  >>> Synced to GCS: {gcs_path}")
            except Exception as e:
                print(f"  >>> GCS sync failed (non-fatal): {e}")

    xm.rendezvous(f"save_{step}")


def load_fsdp_checkpoint(model, optimizer, ckpt_dir, world_size):
    """Load checkpoint. Tries consolidated (portable) first, then sharded.

    Consolidated checkpoint works regardless of chip count.
    Sharded checkpoint only works if world_size matches.
    """
    rank = xr.global_ordinal()

    # Try consolidated checkpoint first (portable)
    consolidated_path = os.path.join(ckpt_dir, "consolidated.pt")
    if os.path.exists(consolidated_path):
        if rank == 0:
            print(f"  Loading consolidated checkpoint (portable)...")
        ckpt = torch.load(consolidated_path, map_location="cpu", weights_only=False)

        # Load into the UNWRAPPED model inside FSDP, then let FSDP re-shard
        # FSDP's load_state_dict handles the re-sharding automatically
        # when loading a full state dict into a sharded model
        model.load_state_dict(ckpt['model'])

        if rank == 0:
            print(f"  Loaded consolidated model from step {ckpt['step']}")
        # Note: optimizer state is lost when loading consolidated — will re-warm
        return ckpt['step']

    # Try sharded checkpoint (must match world_size)
    shard_path = os.path.join(ckpt_dir, f"rank-{rank}-of-{world_size}.pt")
    if os.path.exists(shard_path):
        if rank == 0:
            print(f"  Loading sharded checkpoint (world_size={world_size})...")
        ckpt = torch.load(shard_path, map_location="cpu", weights_only=False)
        if ckpt.get('world_size') != world_size:
            if rank == 0:
                print(f"  WARNING: Checkpoint world_size={ckpt.get('world_size')} != current {world_size}")
                print(f"  Falling back — need consolidated checkpoint for cross-machine resume")
            return 0
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if rank == 0:
            print(f"  Loaded sharded model + optimizer from step {ckpt['step']}")
        return ckpt['step']

    # Check meta.json to see if checkpoint exists but wrong world_size
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if rank == 0:
            print(f"  Checkpoint exists (step {meta['step']}) but no compatible format found")
            print(f"  Checkpoint world_size={meta.get('world_size')}, current={world_size}")
    return 0


def download_gcs_checkpoint(gcs_bucket, local_dir):
    """Download latest checkpoint from GCS. Returns local path or None."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", f"{gcs_bucket}/step_*/meta.json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None

        # Find latest step
        paths = result.stdout.strip().split("\n")
        if not paths or not paths[0]:
            return None

        steps = []
        for p in paths:
            try:
                step = int(p.split("/step_")[1].split("/")[0])
                steps.append((step, p.rsplit("/meta.json")[0]))
            except (IndexError, ValueError):
                continue

        if not steps:
            return None

        latest_step, gcs_dir = max(steps, key=lambda x: x[0])
        local_ckpt = os.path.join(local_dir, f"step_{latest_step}")
        os.makedirs(local_ckpt, exist_ok=True)

        print(f"  Downloading checkpoint step {latest_step} from GCS...")
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{gcs_dir}/*", f"{local_ckpt}/"],
            timeout=600, capture_output=True
        )
        if os.path.exists(os.path.join(local_ckpt, "meta.json")):
            print(f"  Downloaded checkpoint step {latest_step}")
            return local_ckpt
    except Exception as e:
        print(f"  GCS download failed: {e}")
    return None


# ============================================================================
# Training
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path or GCS URI to checkpoint dir to resume from")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total steps (for smoke test)")
    args, _ = parser.parse_known_args()

    # Initialize distributed — FSDP requires torch.distributed
    dist.init_process_group('xla', init_method='xla://')

    device = torch_xla.device()
    world_size = xr.world_size()
    rank = xr.global_ordinal()
    is_main = (rank == 0)

    cfg = LlamaConfig()

    # Batch sizing — with FSDP, model is sharded so memory per chip is much lower
    # FSDP shards weights + optimizer states across chips
    # v6e-4 (4 chips): ~0.55GB weights/chip + ~3.3GB optimizer/chip = ~4GB base
    # Remaining ~27GB for activations → batch_size=4 fits easily
    if world_size >= 32:
        batch_size = 8
        grad_accum = 2   # 8*64*2*1024 = 1,048,576 tok/step
    elif world_size >= 8:
        batch_size = 4
        grad_accum = 4   # 4*8*4*1024 = 131,072 tok/step
    else:
        # v6e-4: FSDP shards model across 4 chips
        batch_size = 4
        grad_accum = 4   # 4*4*4*1024 = 65,536 tok/step
    lr = 2e-4
    warmup_steps = 1000
    total_tokens_target = 16_200_000_000
    tokens_per_step = batch_size * grad_accum * cfg.seq_len * world_size
    total_steps = total_tokens_target // tokens_per_step
    if args.max_steps:
        total_steps = args.max_steps
    save_steps = min(500, max(total_steps // 20, 100))
    log_steps = 25 if total_steps > 50 else 1

    if is_main:
        print(f"{'=' * 70}")
        print(f"  exp029: Llama 1.08B (GQA+Zloss) on ~16.2B tokens — TPU FSDP")
        print(f"  NO QK-Norm, NO embedding scaling (HF-compatible)")
        print(f"  {world_size}x TPU chips, torch_xla {torch_xla.__version__}")
        print(f"  GQA: {cfg.n_head} heads, {cfg.n_kv_head} KV heads")
        print(f"  BS={batch_size}x{world_size}TPUx{grad_accum}accum = {tokens_per_step:,} tok/step")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Save every: {save_steps} steps")
        print(f"  GCS bucket: {GCS_BUCKET}")
        print(f"  FSDP: sharding model across {world_size} chips")
        print(f"{'=' * 70}")

    # Build model on CPU first
    model = Llama(cfg)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"  Params: {num_params:.1f}M (before FSDP sharding)")

    # Wrap with FSDP — each Block becomes a sharding unit
    # checkpoint_module enables gradient checkpointing (recompute activations, save memory)
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )
    auto_wrapper_callable = lambda m, *args, **kwargs: FSDP(
        checkpoint_module(m), *args, **kwargs
    )

    model = FSDP(
        model,
        compute_dtype=torch.bfloat16,
        fp32_reduce_scatter=True,
        flatten_parameters=True,
        reshard_after_forward=True,
        pin_layout_in_collective_ops=True,
        auto_wrap_policy=auto_wrap_policy,
        auto_wrapper_callable=auto_wrapper_callable,
    )

    if is_main:
        print(f"  FSDP wrapped: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params per shard")

    # Optimizer — MUST be created AFTER FSDP wrapping (params are sharded)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
        foreach=False,  # foreach can cause XLA issues
    )

    # Resume from checkpoint
    start_step = 0
    resume_dir = None

    if args.resume:
        if args.resume.startswith("gs://"):
            resume_dir = download_gcs_checkpoint(args.resume.rsplit("/step_")[0] if "/step_" in args.resume else args.resume, LOCAL_CKPT_DIR)
        elif os.path.exists(args.resume):
            resume_dir = args.resume
    else:
        # Auto-detect: check GCS for latest checkpoint
        resume_dir = download_gcs_checkpoint(GCS_BUCKET, LOCAL_CKPT_DIR)

    if resume_dir:
        start_step = load_fsdp_checkpoint(model, optimizer, resume_dir, world_size)
        if is_main:
            print(f"  Resuming from step {start_step}")
    else:
        if is_main:
            print(f"  Starting fresh (no checkpoint found)")

    # Move model to device (FSDP handles the sharding)
    model = model.to(device)

    torch.manual_seed(42 + rank + start_step)
    base_loader = Dataloader("train", batch_size, cfg.seq_len)

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * min(progress, 1.0)))

    model.train()
    total_tokens = start_step * tokens_per_step
    t0 = time.time()
    running_loss = 0.0

    # Skip steps if resuming
    if start_step > 0:
        for _ in range(start_step):
            next(base_loader)
        if is_main:
            print(f"  Skipped {start_step} dataloader batches")

    for step in range(start_step, total_steps):
        if _SHUTDOWN_REQUESTED:
            if is_main:
                print(f"  Emergency save at step {step}...")
            ckpt_dir = os.path.join(LOCAL_CKPT_DIR, f"step_{step}")
            save_fsdp_checkpoint(model, optimizer, step, total_tokens, running_loss,
                                 get_lr(step), ckpt_dir, f"{GCS_BUCKET}/step_{step}")
            if is_main:
                print(f"  >>> Emergency checkpoint saved: {ckpt_dir}")
            break

        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(grad_accum):
            x, y = next(base_loader)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            z_loss = Z_LOSS_COEFF * logits.float().logsumexp(-1).pow(2).mean()
            loss = (ce_loss + z_loss) / grad_accum
            loss.backward()
            accum_loss += ce_loss.item() / grad_accum
            # mark_step per micro-batch to prevent XLA graph unrolling OOM
            xm.mark_step()

        # FSDP handles gradient reduction internally (reduce-scatter)
        # Do NOT call sync_gradients or xm.reduce_gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # NOT xm.optimizer_step() — FSDP handles it
        xm.mark_step()

        total_tokens += tokens_per_step
        running_loss = 0.9 * running_loss + 0.1 * accum_loss if running_loss > 0 else accum_loss

        if is_main and (step + 1) % log_steps == 0:
            elapsed = time.time() - t0
            steps_done = step - start_step + 1
            tps = (steps_done * tokens_per_step) / elapsed
            eta_h = (total_steps - step - 1) / steps_done * elapsed / 3600
            print(f"  step {step+1:>6}/{total_steps} | loss {running_loss:.4f} | "
                  f"lr {current_lr:.2e} | {tps:.0f} tok/s | {elapsed/60:.0f}m | ETA {eta_h:.1f}h")

        # Save checkpoint
        if (step + 1) % save_steps == 0:
            ckpt_dir = os.path.join(LOCAL_CKPT_DIR, f"step_{step+1}")
            save_fsdp_checkpoint(model, optimizer, step + 1, total_tokens, running_loss,
                                 current_lr, ckpt_dir, f"{GCS_BUCKET}/step_{step+1}")
            if is_main:
                print(f"  >>> Saved checkpoint: step_{step+1}")
                # Cleanup old local checkpoints (keep last 3)
                all_ckpts = sorted(
                    glob.glob(os.path.join(LOCAL_CKPT_DIR, "step_*")),
                    key=lambda x: int(os.path.basename(x).split("_")[-1])
                )
                for old in all_ckpts[:-3]:
                    shutil.rmtree(old)

    train_time = time.time() - t0

    if is_main:
        print(f"\n  Training done: {train_time/3600:.1f}h, {total_tokens/1e9:.1f}B tokens")

        # Validation
        print("  Running validation...")
        model.eval()
        val_loader = Dataloader("val", batch_size, cfg.seq_len)
        total_val_loss = 0.0
        total_val_tokens = 0
        target_val = 10_000_000
        while total_val_tokens < target_val:
            vx, vy = next(val_loader)
            vx, vy = vx.to(device), vy.to(device)
            with torch.no_grad():
                vlogits = model(vx)
            vloss = F.cross_entropy(
                vlogits.view(-1, vlogits.size(-1)), vy.view(-1), reduction="sum"
            )
            total_val_loss += vloss.item()
            total_val_tokens += vy.numel()
            xm.mark_step()
        BYTES_PER_TOKEN = 5.2
        val_bpb = (total_val_loss / total_val_tokens) / math.log(2) / BYTES_PER_TOKEN
        print(f"  val_bpb: {val_bpb:.6f}")

        # Save final consolidated checkpoint
        final_dir = os.path.join(LOCAL_CKPT_DIR, "final")
        save_fsdp_checkpoint(model, optimizer, total_steps, total_tokens, running_loss,
                             0.0, final_dir, f"{GCS_BUCKET}/final")
        results = dict(
            val_bpb=val_bpb, train_loss=running_loss, total_steps=total_steps,
            total_tokens=total_tokens, train_hours=round(train_time / 3600, 2),
            world_size=world_size, params_M=round(num_params, 1),
            arch="GQA-4kv+Zloss+FSDP", hardware=f"TPU-{world_size}chips",
        )
        with open(os.path.join(final_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Final model saved: {final_dir}")
        print(f"  Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
