# Matcha-TTS для казахского языка — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Реализовать Matcha-TTS (flow matching + MAS + duration predictor + 1D U-Net) для казахского TTS, запустить на kaznu (2x A10 DDP).

**Architecture:** Text encoder -> MAS alignment -> duration predictor -> expand features to mel length -> 1D U-Net flow matching decoder -> Vocos vocoder. Минимальный порт ключевых компонентов Matcha-TTS в существующий kz-calm проект.

**Tech Stack:** PyTorch, torchaudio, Vocos, DDP (torchrun), bf16, streaming HF datasets.

**Design doc:** `docs/plans/2026-02-22-matcha-tts-kazakh-design.md`

---

### Task 1: Monotonic Alignment Search (MAS)

**Files:**
- Create: `src/kzcalm/model/mas.py`
- Create: `tests/test_mas.py`

**Step 1: Write the test**

```python
# tests/test_mas.py
import torch
from kzcalm.model.mas import monotonic_alignment_search, expand_durations


def test_mas_output_shape():
    """MAS returns durations that sum to T_mel."""
    B, S, T, D = 2, 5, 20, 100
    mu = torch.randn(B, S, D)
    mel = torch.randn(B, T, D)
    text_mask = torch.ones(B, S)
    mel_mask = torch.ones(B, T)

    durations = monotonic_alignment_search(mu, mel, text_mask, mel_mask)

    assert durations.shape == (B, S)
    assert durations.dtype == torch.long
    # Durations must sum to T_mel for each sample
    for b in range(B):
        assert durations[b].sum().item() == T


def test_mas_monotonic():
    """Each text token gets at least 1 frame, monotonically."""
    B, S, T, D = 1, 3, 10, 16
    mu = torch.randn(B, S, D)
    mel = torch.randn(B, T, D)
    text_mask = torch.ones(B, S)
    mel_mask = torch.ones(B, T)

    durations = monotonic_alignment_search(mu, mel, text_mask, mel_mask)
    assert (durations >= 1).all(), "Every token must have at least 1 frame"


def test_expand_durations():
    """Expand features using durations."""
    B, S, D = 1, 3, 8
    features = torch.tensor([[[1.0]*D, [2.0]*D, [3.0]*D]])  # (1, 3, 8)
    durations = torch.tensor([[2, 3, 1]])  # sum=6

    expanded = expand_durations(features, durations)  # (1, 6, 8)
    assert expanded.shape == (1, 6, 8)
    assert (expanded[0, 0] == 1.0).all()
    assert (expanded[0, 1] == 1.0).all()
    assert (expanded[0, 2] == 2.0).all()
    assert (expanded[0, 4] == 2.0).all()
    assert (expanded[0, 5] == 3.0).all()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_mas.py -v`
Expected: FAIL with ImportError

**Step 3: Implement MAS**

```python
# src/kzcalm/model/mas.py
"""Monotonic Alignment Search for Matcha-TTS."""
from __future__ import annotations

import torch


@torch.no_grad()
def monotonic_alignment_search(
    mu: torch.Tensor,
    mel: torch.Tensor,
    text_mask: torch.Tensor,
    mel_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MAS alignment between text features and mel.

    Uses log-likelihood based on Gaussian assumption:
        log p(mel_t | mu_s) = -0.5 * ||mel_t - mu_s||^2

    Args:
        mu: (B, S, D) text encoder output (per-phoneme mel statistics)
        mel: (B, T, D) mel spectrogram
        text_mask: (B, S) 1=valid, 0=pad
        mel_mask: (B, T) 1=valid, 0=pad

    Returns:
        durations: (B, S) frames per text token, sums to T per sample
    """
    B, S, D = mu.shape
    T = mel.shape[1]

    # Log-likelihood: (B, S, T)
    mu_exp = mu.unsqueeze(2)    # (B, S, 1, D)
    mel_exp = mel.unsqueeze(1)  # (B, 1, T, D)
    log_prob = -0.5 * ((mu_exp - mel_exp) ** 2).sum(dim=-1)  # (B, S, T)

    durations = torch.zeros(B, S, dtype=torch.long, device=mu.device)

    for b in range(B):
        s_len = int(text_mask[b].sum().item())
        t_len = int(mel_mask[b].sum().item())
        if s_len == 0 or t_len == 0:
            continue

        lp = log_prob[b, :s_len, :t_len]  # (s_len, t_len)
        durations[b, :s_len] = _mas_single(lp, s_len, t_len)

    return durations


def _mas_single(log_prob: torch.Tensor, S: int, T: int) -> torch.Tensor:
    """Viterbi-style MAS for a single sample.

    Args:
        log_prob: (S, T) log-likelihood matrix
        S: number of text tokens
        T: number of mel frames

    Returns:
        durations: (S,) frames per token
    """
    Q = torch.full((S, T), -1e9, device=log_prob.device)

    Q[0, 0] = log_prob[0, 0]
    for t in range(1, T):
        Q[0, t] = Q[0, t - 1] + log_prob[0, t]

    for s in range(1, S):
        for t in range(s, T):
            Q[s, t] = torch.max(Q[s, t - 1], Q[s - 1, t - 1]) + log_prob[s, t]

    # Backtrack
    durations = torch.zeros(S, dtype=torch.long, device=log_prob.device)
    s = S - 1
    t = T - 1
    durations[s] = 1

    while t > 0:
        if s > 0 and Q[s - 1, t - 1] >= Q[s, t - 1]:
            s -= 1
            durations[s] = 1
        else:
            durations[s] += 1
        t -= 1

    return durations


def expand_durations(features: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Expand per-phoneme features to per-frame using durations.

    Args:
        features: (B, S, D)
        durations: (B, S) integer durations

    Returns:
        expanded: (B, T, D) where T = sum(durations) per sample (max across batch)
    """
    B, S, D = features.shape
    T = durations.sum(dim=-1).max().item()
    expanded = torch.zeros(B, T, D, device=features.device, dtype=features.dtype)

    for b in range(B):
        pos = 0
        for s in range(S):
            dur = durations[b, s].item()
            if dur > 0:
                expanded[b, pos:pos + dur] = features[b, s]
                pos += dur

    return expanded
```

**Step 4: Run tests**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_mas.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/kzcalm/model/mas.py tests/test_mas.py
git commit -m "feat(matcha): add monotonic alignment search"
```

---

### Task 2: Duration Predictor

**Files:**
- Create: `src/kzcalm/model/duration_predictor.py`
- Create: `tests/test_duration_predictor.py`

**Step 1: Write the test**

```python
# tests/test_duration_predictor.py
import torch
from kzcalm.model.duration_predictor import DurationPredictor


def test_duration_predictor_shape():
    B, S, D = 2, 10, 256
    pred = DurationPredictor(d_model=D)
    x = torch.randn(B, S, D)
    mask = torch.ones(B, S)

    log_dur = pred(x, mask)
    assert log_dur.shape == (B, S)


def test_duration_predictor_masked():
    """Padded positions should be zero."""
    B, S, D = 1, 5, 256
    pred = DurationPredictor(d_model=D)
    x = torch.randn(B, S, D)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)

    log_dur = pred(x, mask)
    assert (log_dur[0, 3:] == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_duration_predictor.py -v`
Expected: FAIL with ImportError

**Step 3: Implement**

```python
# src/kzcalm/model/duration_predictor.py
"""Duration predictor for Matcha-TTS."""
from __future__ import annotations

import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    """Predicts log-duration per phoneme from encoder hidden states.

    Architecture: 2x (Conv1d + ReLU + LayerNorm + Dropout) + Linear(1)
    """

    def __init__(self, d_model: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) encoder hidden states
            mask: (B, S) 1=valid, 0=pad

        Returns:
            log_duration: (B, S)
        """
        h = x.transpose(1, 2)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                h = layer(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = layer(h)
        h = h.transpose(1, 2)  # (B, S, D)
        log_dur = self.proj(h).squeeze(-1)  # (B, S)
        return log_dur * mask
```

**Step 4: Run tests**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_duration_predictor.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/kzcalm/model/duration_predictor.py tests/test_duration_predictor.py
git commit -m "feat(matcha): add duration predictor"
```

---

### Task 3: 1D U-Net Flow Decoder

**Files:**
- Create: `src/kzcalm/model/unet_1d.py`
- Create: `tests/test_unet_1d.py`

**Step 1: Write the test**

```python
# tests/test_unet_1d.py
import torch
from kzcalm.model.unet_1d import UNet1DDecoder


def test_unet_output_shape():
    B, T, D = 2, 64, 100
    unet = UNet1DDecoder(mel_dim=D, cond_dim=D, channels=[256, 256, 512])

    x_t = torch.randn(B, T, D)
    cond = torch.randn(B, T, D)
    t = torch.rand(B)
    mask = torch.ones(B, T)

    out = unet(x_t, t, cond, mask)
    assert out.shape == (B, T, D)


def test_unet_different_lengths():
    """U-Net should handle non-power-of-2 lengths via padding."""
    B, T, D = 1, 37, 100
    unet = UNet1DDecoder(mel_dim=D, cond_dim=D, channels=[256, 256, 512])

    x_t = torch.randn(B, T, D)
    cond = torch.randn(B, T, D)
    t = torch.rand(B)
    mask = torch.ones(B, T)

    out = unet(x_t, t, cond, mask)
    assert out.shape == (B, T, D)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_unet_1d.py -v`
Expected: FAIL with ImportError

**Step 3: Implement 1D U-Net**

```python
# src/kzcalm/model/unet_1d.py
"""1D U-Net decoder for flow matching (Matcha-TTS style)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock1d(nn.Module):
    """Residual block: Conv1d + GroupNorm + SiLU + FiLM conditioning."""

    def __init__(self, channels: int, time_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_mlp = nn.Linear(time_dim, channels * 2)  # scale + shift

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T), t_emb: (B, time_dim)"""
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM conditioning
        scale_shift = self.time_mlp(t_emb).unsqueeze(-1)  # (B, 2C, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return x + h


class UNet1DDecoder(nn.Module):
    """1D U-Net for flow matching decoder.

    Input: x_t (noisy mel) + cond (expanded encoder output), both (B, T, D)
    Output: velocity (B, T, D)
    """

    def __init__(
        self,
        mel_dim: int = 100,
        cond_dim: int = 100,
        channels: list[int] | None = None,
        time_dim: int = 256,
    ):
        super().__init__()
        if channels is None:
            channels = [256, 256, 512]

        self.mel_dim = mel_dim
        in_channels = mel_dim + cond_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, channels[0], 1)

        # Encoder (downsample)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(ResBlock1d(channels[i], time_dim))
            self.downsamples.append(
                nn.Conv1d(channels[i], channels[i + 1], 4, stride=2, padding=1)
            )

        # Bottleneck
        self.bottleneck = ResBlock1d(channels[-1], time_dim)

        # Decoder (upsample)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose1d(channels[i], channels[i - 1], 4, stride=2, padding=1)
            )
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels[i - 1] * 2, channels[i - 1], 1),
                    ResBlock1d(channels[i - 1], time_dim),
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], mel_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, T, mel_dim) noisy mel
            t: (B,) timestep [0, 1]
            cond: (B, T, cond_dim) expanded encoder features
            mask: (B, T) 1=valid

        Returns:
            velocity: (B, T, mel_dim)
        """
        T_orig = x_t.shape[1]

        # Concat and transpose to (B, C, T)
        h = torch.cat([x_t, cond], dim=-1).transpose(1, 2)
        t_emb = self.time_mlp(t)

        h = self.input_proj(h)

        # Pad T to be divisible by 2^num_downsamples
        n_down = len(self.downsamples)
        factor = 2 ** n_down
        T_cur = h.shape[2]
        pad_len = (factor - T_cur % factor) % factor
        if pad_len > 0:
            h = F.pad(h, (0, pad_len))

        # Encoder
        skips = []
        for block, down in zip(self.encoder_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder
        for up, dec_block, skip in zip(self.upsamples, self.decoder_blocks, reversed(skips)):
            h = up(h)
            if h.shape[2] != skip.shape[2]:
                h = h[:, :, :skip.shape[2]]
            h = torch.cat([h, skip], dim=1)
            h = dec_block[0](h)  # reduce channels
            h = dec_block[1](h, t_emb)  # ResBlock

        h = self.output_proj(h)

        # Remove padding and transpose back
        h = h[:, :, :T_orig].transpose(1, 2)  # (B, T, mel_dim)
        h = h * mask.unsqueeze(-1)

        return h
```

**Step 4: Run tests**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_unet_1d.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/kzcalm/model/unet_1d.py tests/test_unet_1d.py
git commit -m "feat(matcha): add 1D U-Net flow decoder"
```

---

### Task 4: Text Encoder

**Files:**
- Create: `src/kzcalm/model/text_encoder.py`
- Create: `tests/test_text_encoder.py`

**Step 1: Write the test**

```python
# tests/test_text_encoder.py
import torch
from kzcalm.model.text_encoder import TextEncoder


def test_text_encoder_shape():
    B, S = 2, 10
    enc = TextEncoder(vocab_size=4096, d_model=256, mel_dim=100, num_layers=2, num_heads=4)
    text = torch.randint(0, 4096, (B, S))
    mask = torch.ones(B, S)

    mu, log_sigma, hidden = enc(text, mask)
    assert mu.shape == (B, S, 100)
    assert log_sigma.shape == (B, S, 100)
    assert hidden.shape == (B, S, 256)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_text_encoder.py -v`
Expected: FAIL

**Step 3: Implement**

```python
# src/kzcalm/model/text_encoder.py
"""Text encoder for Matcha-TTS."""
from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Transformer text encoder that produces per-phoneme mel statistics.

    Returns:
        mu: (B, S, mel_dim)
        log_sigma: (B, S, mel_dim)
        hidden: (B, S, d_model) for duration predictor
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        mel_dim: int = 100,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_text_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_text_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mu_proj = nn.Linear(d_model, mel_dim)
        self.log_sigma_proj = nn.Linear(d_model, mel_dim)

    def forward(
        self, text: torch.Tensor, text_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text: (B, S) token ids
            text_mask: (B, S) 1=valid, 0=pad
        """
        S = text.shape[1]
        pos_ids = torch.arange(S, device=text.device)
        h = self.embed(text) + self.pos(pos_ids)

        pad_mask = (text_mask == 0)
        hidden = self.encoder(h, src_key_padding_mask=pad_mask)

        mu = self.mu_proj(hidden)
        log_sigma = self.log_sigma_proj(hidden)

        return mu, log_sigma, hidden
```

**Step 4: Run tests**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_text_encoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/kzcalm/model/text_encoder.py tests/test_text_encoder.py
git commit -m "feat(matcha): add text encoder with mu/sigma projections"
```

---

### Task 5: MatchaTTS Orchestrator

**Files:**
- Create: `src/kzcalm/model/matcha.py`
- Create: `tests/test_matcha.py`

**Step 1: Write the test**

```python
# tests/test_matcha.py
import torch
from kzcalm.model.matcha import MatchaTTS


def test_matcha_forward_training():
    """Training forward pass returns flow_loss and duration_loss."""
    model = MatchaTTS(
        vocab_size=4096, mel_dim=100, encoder_dim=256,
        encoder_layers=2, encoder_heads=4, encoder_ff=512,
        unet_channels=[128, 128, 256], dropout=0.1,
    )
    B, S, T = 2, 8, 32
    text = torch.randint(0, 4096, (B, S))
    text_mask = torch.ones(B, S)
    mel = torch.randn(B, T, 100)
    mel_mask = torch.ones(B, T)

    flow_loss, dur_loss = model(text, text_mask, mel, mel_mask)
    assert flow_loss.shape == ()
    assert dur_loss.shape == ()
    assert flow_loss.item() > 0
    assert flow_loss.requires_grad


def test_matcha_synthesize():
    """Inference produces mel of predicted length."""
    model = MatchaTTS(
        vocab_size=4096, mel_dim=100, encoder_dim=256,
        encoder_layers=2, encoder_heads=4, encoder_ff=512,
        unet_channels=[128, 128, 256], dropout=0.1,
    )
    model.eval()
    B, S = 1, 5
    text = torch.randint(0, 4096, (B, S))
    text_mask = torch.ones(B, S)

    with torch.no_grad():
        mel = model.synthesize(text, text_mask, num_steps=4)

    assert mel.dim() == 3
    assert mel.shape[0] == B
    assert mel.shape[2] == 100


def test_matcha_param_count():
    """Sanity check: model should be less than 30M params."""
    model = MatchaTTS(
        vocab_size=4096, mel_dim=100, encoder_dim=256,
        encoder_layers=4, encoder_heads=4, encoder_ff=1024,
        unet_channels=[256, 256, 512], dropout=0.1,
    )
    n = sum(p.numel() for p in model.parameters())
    assert n < 30_000_000, f"Model too large: {n/1e6:.1f}M"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_matcha.py -v`
Expected: FAIL

**Step 3: Implement**

```python
# src/kzcalm/model/matcha.py
"""Matcha-TTS: Flow matching TTS with MAS and duration predictor."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kzcalm.model.text_encoder import TextEncoder
from kzcalm.model.duration_predictor import DurationPredictor
from kzcalm.model.mas import monotonic_alignment_search, expand_durations
from kzcalm.model.unet_1d import UNet1DDecoder


class MatchaTTS(nn.Module):
    """Matcha-TTS model.

    Training: text + mel -> MAS alignment -> duration loss + flow matching loss
    Inference: text -> predicted durations -> ODE sampling -> mel
    """

    def __init__(
        self,
        vocab_size: int,
        mel_dim: int = 100,
        encoder_dim: int = 256,
        encoder_layers: int = 4,
        encoder_heads: int = 4,
        encoder_ff: int = 1024,
        unet_channels: list[int] | None = None,
        dropout: float = 0.1,
        max_text_len: int = 512,
    ):
        super().__init__()
        self.mel_dim = mel_dim

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=encoder_dim,
            mel_dim=mel_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            d_ff=encoder_ff,
            dropout=dropout,
            max_text_len=max_text_len,
        )

        self.duration_predictor = DurationPredictor(
            d_model=encoder_dim, dropout=dropout,
        )

        self.unet = UNet1DDecoder(
            mel_dim=mel_dim,
            cond_dim=mel_dim,
            channels=unet_channels,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        mel: torch.Tensor,
        mel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Returns:
            flow_loss: scalar
            duration_loss: scalar
        """
        # 1. Encode text
        mu, log_sigma, hidden = self.text_encoder(text, text_mask)

        # 2. MAS: find alignment
        with torch.no_grad():
            durations = monotonic_alignment_search(mu, mel, text_mask, mel_mask)

        # 3. Duration predictor loss
        log_dur_pred = self.duration_predictor(hidden.detach(), text_mask)
        log_dur_target = torch.log(durations.float().clamp(min=1))
        dur_loss = F.mse_loss(log_dur_pred * text_mask, log_dur_target * text_mask)

        # 4. Expand mu to mel length
        mu_expanded = expand_durations(mu, durations)  # (B, T', mel_dim)
        T = mel.shape[1]
        if mu_expanded.shape[1] > T:
            mu_expanded = mu_expanded[:, :T]
        elif mu_expanded.shape[1] < T:
            pad = T - mu_expanded.shape[1]
            mu_expanded = F.pad(mu_expanded, (0, 0, 0, pad))

        # 5. Flow matching
        x0 = torch.randn_like(mel)
        t = torch.rand(mel.shape[0], device=mel.device)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * x0 + t_expand * mel

        velocity = self.unet(x_t, t, mu_expanded, mel_mask)
        target = mel - x0

        # Masked MSE
        diff = (velocity - target) ** 2
        diff = diff.mean(dim=-1)  # (B, T)
        flow_loss = (diff * mel_mask).sum() / mel_mask.sum().clamp(min=1)

        return flow_loss, dur_loss

    @torch.no_grad()
    def synthesize(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        num_steps: int = 8,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate mel spectrogram from text."""
        mu, log_sigma, hidden = self.text_encoder(text, text_mask)

        # Predict durations
        log_dur = self.duration_predictor(hidden, text_mask)
        durations = torch.round(torch.exp(log_dur) * length_scale).long().clamp(min=1)
        durations = durations * text_mask.long()

        # Expand
        mu_expanded = expand_durations(mu, durations)
        T = mu_expanded.shape[1]
        mel_mask = torch.ones(mu_expanded.shape[0], T, device=text.device)

        # ODE sampling (Euler)
        x = torch.randn_like(mu_expanded)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = self.unet(x, t, mu_expanded, mel_mask)
            x = x + v * dt

        return x
```

**Step 4: Run tests**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_matcha.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/kzcalm/model/matcha.py tests/test_matcha.py
git commit -m "feat(matcha): add MatchaTTS orchestrator model"
```

---

### Task 6: Experiment Config

**Files:**
- Create: `configs/experiments/exp003_matcha.yaml`
- Create: `configs/experiments/exp003_matcha_overfit.yaml`

**Step 1: Create configs**

exp003_matcha.yaml:
```yaml
inherits: base
experiment_name: exp003_matcha

data:
  hf_audio_dataset: stukenov/kzcalm-tts-kk-v1
  split: "train"

codec:
  type: mel
  latent_dim: 100
  n_mels: 100
  hop_length: 256
  n_fft: 1024

tokenizer:
  hf_repo: "stukenov/kzcalm-sp-tokenizer-4k-kk-v1"

model:
  type: matcha
  encoder_layers: 4
  encoder_dim: 256
  encoder_heads: 4
  encoder_ff: 1024
  unet_channels: [256, 256, 512]
  dropout: 0.1
  max_text_len: 512
  duration_loss_weight: 1.0

flow:
  num_sampling_steps: 8
  sigma_min: 0.001
  loss_type: mse

training:
  batch_size: 32
  gradient_accumulation_steps: 1
  max_steps: 200000
  learning_rate: 1.0e-4
  warmup_steps: 4000
  weight_decay: 0.01
  grad_clip: 1.0
  bf16: true
  num_workers: 4
  save_steps: 5000
  logging_steps: 100
```

exp003_matcha_overfit.yaml:
```yaml
inherits: base
experiment_name: exp003_matcha_overfit

data:
  hf_audio_dataset: stukenov/kzcalm-tts-kk-v1
  split: "train[:50]"

codec:
  type: mel
  latent_dim: 100
  n_mels: 100
  hop_length: 256
  n_fft: 1024

tokenizer:
  hf_repo: "stukenov/kzcalm-sp-tokenizer-4k-kk-v1"

model:
  type: matcha
  encoder_layers: 4
  encoder_dim: 256
  encoder_heads: 4
  encoder_ff: 1024
  unet_channels: [256, 256, 512]
  dropout: 0.0
  max_text_len: 512
  duration_loss_weight: 1.0

flow:
  num_sampling_steps: 8
  sigma_min: 0.001
  loss_type: mse

training:
  batch_size: 8
  gradient_accumulation_steps: 1
  max_steps: 1000
  learning_rate: 1.0e-4
  warmup_steps: 100
  weight_decay: 0.01
  grad_clip: 1.0
  bf16: true
  num_workers: 2
  save_steps: 500
  logging_steps: 10
```

**Step 2: Commit**

```bash
git add configs/experiments/exp003_matcha.yaml configs/experiments/exp003_matcha_overfit.yaml
git commit -m "feat(matcha): add exp003 configs (full + overfit)"
```

---

### Task 7: DDP Training Script

**Files:**
- Create: `src/kzcalm/train_matcha.py`
- Create: `tests/test_train_matcha.py`

**Step 1: Write smoke test**

```python
# tests/test_train_matcha.py
import torch
from kzcalm.model.matcha import MatchaTTS


def test_training_step_backward():
    """Smoke test: one training step with backward pass."""
    model = MatchaTTS(
        vocab_size=100, mel_dim=100, encoder_dim=128,
        encoder_layers=1, encoder_heads=2, encoder_ff=256,
        unet_channels=[64, 64, 128], dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B, S, T = 2, 5, 32
    text = torch.randint(0, 100, (B, S))
    text_mask = torch.ones(B, S)
    mel = torch.randn(B, T, 100)
    mel_mask = torch.ones(B, T)

    flow_loss, dur_loss = model(text, text_mask, mel, mel_mask)
    loss = flow_loss + dur_loss
    loss.backward()
    optimizer.step()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients computed"
```

**Step 2: Implement DDP training script**

See `src/kzcalm/train_matcha.py` in design doc. Key points:
- DDP init via `torchrun` environment variables (LOCAL_RANK, WORLD_SIZE)
- `dist.init_process_group("nccl")` for multi-GPU
- Falls back to single-GPU if WORLD_SIZE=1
- MelDataset streaming with mel_collate_fn (reused from exp002)
- Two losses: flow_loss + duration_loss_weight * dur_loss
- text_mask conversion: dataset returns bool (True=pad), model expects float (1=valid)
- Checkpoint saves raw_model (unwrapped from DDP)
- bf16 autocast, AdamW, cosine LR with warmup

**Step 3: Run smoke test**

Run: `cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/test_train_matcha.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/kzcalm/train_matcha.py tests/test_train_matcha.py
git commit -m "feat(matcha): add DDP training script"
```

---

### Task 8: Inference Script

**Files:**
- Create: `scripts/test_matcha_checkpoint.py`

**Step 1: Create script**

Key logic:
- Load checkpoint, extract config
- Build MatchaTTS from config params
- Load Vocos vocoder
- For each test text: encode -> synthesize -> denormalize (mel * 3.80 + (-1.42)) -> Vocos decode -> save WAV via soundfile
- Support --num_steps and --length_scale CLI args

**Step 2: Commit**

```bash
git add scripts/test_matcha_checkpoint.py
git commit -m "feat(matcha): add inference script"
```

---

### Task 9: Deploy and Run on kaznu

**Step 1: Run all tests locally**

```bash
cd /Users/sakentukenov/slm/kz-calm && python -m pytest tests/ -v
```
Expected: All pass

**Step 2: Deploy to kaznu**

```bash
rsync -avz --exclude='.venv' --exclude='outputs' --exclude='__pycache__' \
  /Users/sakentukenov/slm/kz-calm/ kaznu:/root/slm/kz-calm/
```

**Step 3: Install vocos on server**

```bash
ssh kaznu "source /root/slm/.venv/bin/activate && pip install vocos"
```

**Step 4: Run tests on server**

```bash
ssh kaznu "cd /root/slm/kz-calm && source /root/slm/.venv/bin/activate && python -m pytest tests/ -v"
```

**Step 5: Run overfit test (single GPU)**

```bash
ssh kaznu "cd /root/slm/kz-calm && source /root/slm/.venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -m kzcalm.train_matcha \
  --config configs/experiments/exp003_matcha_overfit.yaml"
```

Criteria: flow_loss and dur_loss both approach 0 by 1000 steps.

**Step 6: If overfit works, run DDP full training**

```bash
ssh kaznu "cd /root/slm/kz-calm && source /root/slm/.venv/bin/activate && \
  screen -dmS matcha_exp003 torchrun --nproc_per_node=2 \
  -m kzcalm.train_matcha --config configs/experiments/exp003_matcha.yaml"
```

**Step 7: Monitor**

```bash
ssh kaznu "screen -r matcha_exp003"
```

**Step 8: Listen to checkpoint at 10K steps**

```bash
# On server:
python scripts/test_matcha_checkpoint.py \
  --checkpoint outputs/exp003_matcha/checkpoint-10000/model.pt \
  --output_dir /root/slm/kz-calm/inference_test
# Download:
scp kaznu:/root/slm/kz-calm/inference_test/*.wav ./
```

Criteria at 10K: recognizable words in Kazakh.
