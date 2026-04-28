"""
OmniAudio v2: Train encoder (+LLM) on TPU — Stage 1 CTC and Stage 2 E2E.
Single-device mode (1 TPU chip) — no xmp.spawn, no multiprocess crashes.

Stage 1 (CTC pretrain):
  PJRT_DEVICE=TPU python3 train_ctc_tpu.py --size 50m --stage ctc

Stage 2 (E2E):
  PJRT_DEVICE=TPU python3 train_ctc_tpu.py --size 50m --stage e2e
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ["TPU_RUNTIME_METRICS_PORTS"] = ""
os.environ["TPU_STDERR_LOG_LEVEL"] = "0"

import argparse
import io
import math
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp

# ============================================================================
# Model components — all XLA-compatible
# ============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, cos, sin):
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def manual_attention(q, k, v, dropout_p=0.0):
    scale = math.sqrt(q.size(-1))
    attn = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return torch.matmul(attn, v)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        ffn_dim = int(d_model * 8 / 3)
        ffn_dim = ((ffn_dim + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        attn_out = manual_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.o_proj(attn_out)
        if self.training and self.dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.dropout_p)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class AudioEncoderV2(nn.Module):
    def __init__(self, n_mels=80, d_model=256, n_heads=4, n_layers=6, n_conv=2, dropout=0.1):
        super().__init__()
        convs = []
        in_ch = n_mels
        for _ in range(n_conv):
            convs.append(nn.Conv1d(in_ch, d_model, kernel_size=3, stride=2, padding=1))
            convs.append(nn.GELU())
            in_ch = d_model
        self.conv_stack = nn.Sequential(*convs)
        self.norm = RMSNorm(d_model)
        self.rope = RotaryEmbedding(dim=d_model // n_heads)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, mel):
        x = self.conv_stack(mel)
        x = x.transpose(1, 2)
        x = self.norm(x)
        cos, sin = self.rope(x.size(1))
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class AudioProjectorV2(nn.Module):
    def __init__(self, audio_dim, llm_dim):
        super().__init__()
        self.linear = nn.Linear(audio_dim, llm_dim)
        self.norm = RMSNorm(llm_dim)

    def forward(self, x):
        return self.norm(self.linear(x))


class OmniAudioTPU(nn.Module):
    def __init__(self, encoder_config, vocab_size, stage, llm_name=None, llm_dim=768):
        super().__init__()
        self.stage = stage
        audio_dim = encoder_config["d_model"]
        self.encoder = AudioEncoderV2(**encoder_config)
        self.ctc_head = nn.Linear(audio_dim, vocab_size)
        self.llm = None
        self.projector = None
        if stage == "e2e" and llm_name:
            self.projector = AudioProjectorV2(audio_dim, llm_dim)
            from transformers import LlamaForCausalLM
            self.llm = LlamaForCausalLM.from_pretrained(
                llm_name, attn_implementation="eager", torch_dtype=torch.bfloat16,
            )
            for p in self.llm.parameters():
                p.requires_grad = False

    def forward_ctc(self, mel, targets, target_lengths):
        enc_out = self.encoder(mel)
        logits = self.ctc_head(enc_out)
        # CTC loss requires float32 (not bf16)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        input_lengths = torch.full((mel.size(0),), log_probs.size(0),
                                   dtype=torch.long, device=mel.device)
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          blank=0, zero_infinity=True)

    def forward_e2e(self, mel, text_ids, ctc_weight=0.3,
                    ctc_targets=None, ctc_target_lengths=None):
        enc_out = self.encoder(mel)
        audio_embeds = self.projector(enc_out)
        text_ids_safe = text_ids.clone()
        text_ids_safe[text_ids_safe < 0] = 0
        text_embeds = self.llm.model.embed_tokens(text_ids_safe)
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
        B, T_audio = audio_embeds.shape[:2]
        ignore = torch.full((B, T_audio), -100, dtype=torch.long, device=mel.device)
        labels = torch.cat([ignore, text_ids], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
        ce_loss = outputs.loss
        if ctc_weight > 0.0 and ctc_targets is not None:
            ctc_loss = self.forward_ctc(mel, ctc_targets, ctc_target_lengths)
            return (1.0 - ctc_weight) * ce_loss + ctc_weight * ctc_loss
        return ce_loss


# ============================================================================
# Configs
# ============================================================================

CONFIGS = {
    "50m": {
        "d_model": 256, "n_heads": 4, "n_layers": 6, "n_conv": 2,
        "llm_name": "stukenov/sozkz-core-llama-50m-kk-base-v2", "llm_dim": 512,
        "ctc_batch": 32, "e2e_batch": 16,  # per-device; effective = batch * num_chips
        "ctc_lr": 3e-4, "e2e_lr": 3e-4,
        "ctc_repo": "stukenov/sozkz-core-omniaudio-50m-kk-ctc-v1",
        "e2e_repo": "stukenov/sozkz-core-omniaudio-50m-kk-asr-v1",
    },
    "150m": {
        "d_model": 512, "n_heads": 8, "n_layers": 12, "n_conv": 2,
        "llm_name": "stukenov/sozkz-core-llama-150m-kk-base-v1", "llm_dim": 768,
        "ctc_batch": 8, "e2e_batch": 4,
        "ctc_lr": 3e-4, "e2e_lr": 3e-4,
        "ctc_repo": "stukenov/sozkz-core-omniaudio-150m-kk-ctc-v1",
        "e2e_repo": "stukenov/sozkz-core-omniaudio-150m-kk-asr-v1",
    },
    "600m": {
        "d_model": 768, "n_heads": 12, "n_layers": 16, "n_conv": 2,
        "llm_name": "stukenov/sozkz-core-llama-600m-kk-base-v1", "llm_dim": 1280,
        "ctc_batch": 4, "e2e_batch": 2,
        "ctc_lr": 3e-4, "e2e_lr": 2e-4,
        "ctc_repo": "stukenov/sozkz-core-omniaudio-600m-kk-ctc-v1",
        "e2e_repo": "stukenov/sozkz-core-omniaudio-600m-kk-asr-v1",
    },
    "1b": {
        "d_model": 1024, "n_heads": 16, "n_layers": 24, "n_conv": 2,
        "llm_name": "stukenov/sozkz-core-llama-1b-kk-base-v1", "llm_dim": 2048,
        "ctc_batch": 2, "e2e_batch": 1,
        "ctc_lr": 3e-4, "e2e_lr": 1e-4,
        "ctc_repo": "stukenov/sozkz-core-omniaudio-1b-kk-ctc-v1",
        "e2e_repo": "stukenov/sozkz-core-omniaudio-1b-kk-asr-v1",
    },
}

VOCAB_SIZE = 50257
N_MELS = 80
MAX_AUDIO_LEN = 10.0  # 10s instead of 15s — saves ~33% HBM on XLA
MAX_TEXT_LEN = 256
SAMPLE_RATE = 16000
NUM_EPOCHS_CTC = 2
NUM_EPOCHS_E2E = 5
SAVE_STEPS = 5000
LOG_STEPS = 50
WARMUP_RATIO = 0.05
CTC_WEIGHT_E2E = 0.3
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.expanduser("~/tokenizers/kazakh-gpt2-50k"))
DATASET_NAME = "stukenov/sozkz-asr-mels-kk-v1"


# ============================================================================
# Data
# ============================================================================

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def load_dataset_splits():
    from datasets import load_dataset
    # Use streaming to avoid filling disk (dataset is ~77GB)
    streaming = os.environ.get("STREAM_DATASET", "1") == "1"
    if streaming:
        print("  Using streaming mode (no disk cache)")
        ds = load_dataset(DATASET_NAME, streaming=True)
        return ds["train"], None  # no val in streaming
    ds = load_dataset(DATASET_NAME)
    train_ds = ds["train"]
    val_ds = ds.get("validation") or ds.get("test")
    if val_ds is None:
        split = train_ds.train_test_split(test_size=0.01, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]
    return train_ds, val_ds


class MelCollator:
    def __init__(self, tokenizer, stage, max_audio_len=15.0, max_text_len=256):
        self.tokenizer = tokenizer
        self.stage = stage
        self.max_mel_frames = int(max_audio_len * SAMPLE_RATE / 160)
        self.max_text_len = max_text_len
        self._tag_re = re.compile(r"<[^>]+>")

    def __call__(self, batch):
        # CRITICAL for XLA: all tensors must have FIXED shapes across batches.
        # Variable shapes cause XLA recompilation → OOM.
        mels, ctc_targets_list, ctc_lengths, text_ids_list = [], [], [], []
        for item in batch:
            mel_bytes = item["mel"]
            if isinstance(mel_bytes, dict) and "bytes" in mel_bytes:
                mel_bytes = mel_bytes["bytes"]
            mel = np.load(io.BytesIO(mel_bytes))
            mel = torch.from_numpy(mel).float()
            if mel.size(1) > self.max_mel_frames:
                mel = mel[:, :self.max_mel_frames]
            text = item.get("text", "")
            text = self._tag_re.sub("", text).strip()
            if not text:
                continue
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > self.max_text_len:
                token_ids = token_ids[:self.max_text_len]
            if len(token_ids) == 0:
                continue
            mels.append(mel)
            ctc_targets_list.append(torch.tensor(token_ids, dtype=torch.long))
            ctc_lengths.append(len(token_ids))
            if self.stage == "e2e":
                text_ids_list.append(torch.tensor(token_ids + [0], dtype=torch.long))

        if not mels:
            return None

        # FIXED shape: always pad to max_mel_frames (XLA needs static shapes)
        padded_mels = torch.zeros(len(mels), N_MELS, self.max_mel_frames)
        for i, m in enumerate(mels):
            padded_mels[i, :, :m.size(1)] = m

        # FIXED shape CTC targets: pad each to max_text_len, use flat concat
        ctc_targets = torch.cat(ctc_targets_list)
        ctc_target_lengths = torch.tensor(ctc_lengths, dtype=torch.long)

        result = {
            "mel": padded_mels,
            "ctc_targets": ctc_targets,
            "ctc_target_lengths": ctc_target_lengths,
        }
        if self.stage == "e2e" and text_ids_list:
            # Fixed shape text_ids
            padded_text = torch.full((len(text_ids_list), self.max_text_len + 1), -100, dtype=torch.long)
            for i, t in enumerate(text_ids_list):
                padded_text[i, :t.size(0)] = t
            result["text_ids"] = padded_text
        return result


# ============================================================================
# Training — xmp.spawn for all chips
# ============================================================================

def train(index, args):
    device = xm.xla_device()
    world_size = xr.world_size()
    ordinal = xr.global_ordinal()
    is_main = ordinal == 0
    if is_main:
        print(f"=== OmniAudio TPU: {args.size} / stage={args.stage} ===")
        print(f"  device: {device}, world_size: {world_size}")

    cfg = CONFIGS[args.size]
    stage = args.stage
    is_e2e = stage == "e2e"
    batch_size = cfg["e2e_batch"] if is_e2e else cfg["ctc_batch"]
    lr = cfg["e2e_lr"] if is_e2e else cfg["ctc_lr"]
    num_epochs = NUM_EPOCHS_E2E if is_e2e else NUM_EPOCHS_CTC
    hf_repo = cfg["e2e_repo"] if is_e2e else cfg["ctc_repo"]

    if is_main:
        print(f"  encoder: d={cfg['d_model']}, h={cfg['n_heads']}, L={cfg['n_layers']}")
        if is_e2e:
            print(f"  LLM: {cfg['llm_name']} (dim={cfg['llm_dim']}, frozen)")
        print(f"  batch={batch_size}/device, effective={batch_size*world_size}, lr={lr}, epochs={num_epochs}")

    # Load data FIRST (before model, to avoid OOM during download)
    if is_main:
        print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    if is_main:
        print("Loading dataset...")
    train_ds, val_ds = load_dataset_splits()
    train_len = len(train_ds) if hasattr(train_ds, '__len__') else "streaming"
    val_len = len(val_ds) if val_ds is not None and hasattr(val_ds, '__len__') else "none"
    if is_main:
        print(f"  train: {train_len}, val: {val_len}")

    collator = MelCollator(tokenizer, stage, max_audio_len=MAX_AUDIO_LEN, max_text_len=MAX_TEXT_LEN)
    is_streaming = hasattr(train_ds, '__iter__') and not hasattr(train_ds, '__len__')
    # For multi-chip: each process gets different data shard
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=not is_streaming,
        collate_fn=collator, num_workers=0, drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, collate_fn=collator, num_workers=0, drop_last=False,
        )

    # Build model
    encoder_config = {
        "n_mels": N_MELS, "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"], "n_layers": cfg["n_layers"], "n_conv": cfg["n_conv"],
    }
    llm_name = cfg["llm_name"] if is_e2e else None
    llm_dim = cfg["llm_dim"] if is_e2e else 768
    model = OmniAudioTPU(encoder_config, VOCAB_SIZE, stage, llm_name, llm_dim)

    # Load CTC checkpoint for E2E
    if is_e2e:
        ctc_path = args.ctc_checkpoint
        if not ctc_path:
            try:
                from huggingface_hub import hf_hub_download
                ctc_path = hf_hub_download(cfg["ctc_repo"], "model.pt")
                print(f"  Downloaded CTC checkpoint from {cfg['ctc_repo']}")
            except Exception as e:
                print(f"  WARNING: no CTC checkpoint: {e}")
        if ctc_path:
            ckpt = torch.load(ctc_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"  Loaded CTC (missing={len(missing)}, unexpected={len(unexpected)})")

    # Freeze/unfreeze
    if stage == "ctc":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.ctc_head.parameters():
            p.requires_grad = True
    else:
        for p in model.encoder.parameters():
            p.requires_grad = True
        if model.projector:
            for p in model.projector.parameters():
                p.requires_grad = True
        for p in model.ctc_head.parameters():
            p.requires_grad = True

    # Resume from HF checkpoint (all processes load — HF hub caches after first download)
    resume_step = 0
    if args.resume:
        if is_main:
            print(f"  Checking HF for resume checkpoint...")
        resume_step, resume_path = find_latest_hf_checkpoint(hf_repo)
        if resume_path:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if is_main:
                print(f"  Resumed from step {resume_step} (missing={len(missing)}, unexpected={len(unexpected)})")
        elif is_main:
            print(f"  No checkpoint found, starting from scratch")

    model = model.to(device).to(torch.bfloat16)  # bf16 saves ~50% HBM
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  params: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01, foreach=False)
    # For streaming: estimate ~1M samples / batch_size
    if is_streaming:
        est_steps_per_epoch = 1_000_000 // batch_size
    else:
        est_steps_per_epoch = len(train_loader)
    total_steps = est_steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    if is_main:
        print(f"  total_steps~{total_steps}, warmup={warmup_steps}, streaming={is_streaming}, chips={world_size}")
        if resume_step > 0:
            print(f"  Resuming from step {resume_step} (optimizer fresh, scheduler fast-forwarded)")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ctc_weight = CTC_WEIGHT_E2E if is_e2e else 0.0

    output_dir = f"/tmp/omniaudio_{stage}_{args.size}"
    os.makedirs(output_dir, exist_ok=True)

    global_step = resume_step
    best_val_loss = float("inf")
    # Fast-forward scheduler to resume_step
    if resume_step > 0:
        for _ in range(resume_step):
            scheduler.step()
    start_time = time.time()
    if is_main:
        print(f"Starting training from step {global_step}...")

    for epoch in range(num_epochs):
        model.train()
        if model.llm is not None:
            model.llm.eval()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if batch is None:
                continue
            mel = batch["mel"].to(device=device, dtype=torch.bfloat16)
            ctc_targets = batch["ctc_targets"].to(device)
            ctc_lengths = batch["ctc_target_lengths"].to(device)

            if is_e2e:
                text_ids = batch["text_ids"].to(device)
                loss = model.forward_e2e(mel, text_ids, ctc_weight, ctc_targets, ctc_lengths)
            else:
                loss = model.forward_ctc(mel, ctc_targets, ctc_lengths)

            loss.backward()
            # Sync gradients across all chips
            if world_size > 1:
                gradients = [p.grad for p in trainable_params if p.grad is not None]
                xm.all_reduce("sum", gradients)
                for g in gradients:
                    g.div_(world_size)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            xm.mark_step()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1

            if args.max_steps > 0 and global_step >= args.max_steps:
                if is_main:
                    print(f"  Reached max_steps={args.max_steps}")
                break

            if is_main and global_step % LOG_STEPS == 0:
                avg = epoch_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"  Step {global_step} | Loss: {avg:.4f} | LR: {lr_now:.2e} | {elapsed:.0f}s")

            if global_step % SAVE_STEPS == 0:
                save_checkpoint(model, output_dir, global_step, hf_repo)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        avg_train = epoch_loss / max(n_batches, 1)
        if is_main:
            print(f"  Epoch {epoch+1}/{num_epochs} | Train loss: {avg_train:.4f}")

        # Validation (skip if streaming — no val set)
        if val_loader is not None:
            val_loss = run_validation(model, val_loader, device, stage, ctc_weight)
            if is_main:
                print(f"  Epoch {epoch+1}/{num_epochs} | Val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, output_dir, "best", hf_repo)
        else:
            if avg_train < best_val_loss:
                best_val_loss = avg_train
                save_checkpoint(model, output_dir, "best", hf_repo)

    save_checkpoint(model, output_dir, "final", hf_repo)
    if is_main:
        elapsed = time.time() - start_time
        print(f"=== DONE: {args.size}/{stage} | Best val: {best_val_loss:.4f} | {elapsed:.0f}s ===")


def run_validation(model, val_loader, device, stage, ctc_weight):
    model.train(False)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            mel = batch["mel"].to(device=device, dtype=torch.bfloat16)
            ctc_targets = batch["ctc_targets"].to(device)
            ctc_lengths = batch["ctc_target_lengths"].to(device)
            if stage == "e2e":
                text_ids = batch["text_ids"].to(device)
                loss = model.forward_e2e(mel, text_ids, ctc_weight, ctc_targets, ctc_lengths)
            else:
                loss = model.forward_ctc(mel, ctc_targets, ctc_lengths)
            xm.mark_step()
            total_loss += loss.item()
            n += 1
            if n >= 50:
                break
    model.train()
    return total_loss / max(n, 1)


def find_latest_hf_checkpoint(hf_repo_id):
    """Find the latest numbered checkpoint on HF. Returns (step, path) or (0, None)."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = api.list_repo_files(hf_repo_id)
        steps = []
        for f in files:
            # Match checkpoint-NNNNN/model.pt
            if f.startswith("checkpoint-") and f.endswith("/model.pt"):
                try:
                    s = int(f.split("/")[0].replace("checkpoint-", ""))
                    steps.append(s)
                except ValueError:
                    pass
        if not steps:
            return 0, None
        latest = max(steps)
        path = hf_hub_download(hf_repo_id, f"checkpoint-{latest}/model.pt")
        return latest, path
    except Exception as e:
        print(f"  Resume: could not fetch from HF: {e}")
        return 0, None


def save_checkpoint(model, output_dir, step, hf_repo_id):
    """Must be called by ALL xmp processes — xm.save is collective."""
    ckpt_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_path, exist_ok=True)
    state = {name: param.data.cpu() for name, param in model.named_parameters()
             if not name.startswith("llm.")}
    model_path = os.path.join(ckpt_path, "model.pt")
    xm.save(state, model_path)  # collective op — all processes call, only master writes
    is_main = xm.is_master_ordinal()
    if is_main:
        print(f"  Saved checkpoint-{step}")
        if hf_repo_id:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(repo_id=hf_repo_id, exist_ok=True, repo_type="model")
                api.upload_file(path_or_fileobj=model_path,
                                path_in_repo=f"checkpoint-{step}/model.pt", repo_id=hf_repo_id)
                api.upload_file(path_or_fileobj=model_path,
                                path_in_repo="model.pt", repo_id=hf_repo_id)
                print(f"  Pushed checkpoint-{step} to {hf_repo_id}")
            except Exception as e:
                print(f"  HF push failed (continuing): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=["50m", "150m", "600m", "1b"])
    parser.add_argument("--stage", required=True, choices=["ctc", "e2e"])
    parser.add_argument("--ctc-checkpoint", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from latest HF checkpoint")
    parser.add_argument("--single-device", action="store_true", help="Use 1 chip only (fallback)")
    args = parser.parse_args()
    if args.single_device:
        train(0, args)
    else:
        xmp.spawn(train, args=(args,))
