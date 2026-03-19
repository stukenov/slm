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
    def __init__(self, channels: int, time_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_mlp = nn.Linear(time_dim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        scale_shift = self.time_mlp(t_emb).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class UNet1DDecoder(nn.Module):
    def __init__(self, mel_dim: int = 100, cond_dim: int = 100, channels: list[int] | None = None, time_dim: int = 256):
        super().__init__()
        if channels is None:
            channels = [256, 256, 512]
        self.mel_dim = mel_dim
        in_channels = mel_dim + cond_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input_proj = nn.Conv1d(in_channels, channels[0], 1)
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(ResBlock1d(channels[i], time_dim))
            self.downsamples.append(nn.Conv1d(channels[i], channels[i + 1], 4, stride=2, padding=1))
        self.bottleneck = ResBlock1d(channels[-1], time_dim)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.upsamples.append(nn.ConvTranspose1d(channels[i], channels[i - 1], 4, stride=2, padding=1))
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv1d(channels[i - 1] * 2, channels[i - 1], 1),
                ResBlock1d(channels[i - 1], time_dim),
            ))
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], mel_dim, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T_orig = x_t.shape[1]
        h = torch.cat([x_t, cond], dim=-1).transpose(1, 2)
        t_emb = self.time_mlp(t)
        h = self.input_proj(h)
        n_down = len(self.downsamples)
        factor = 2 ** n_down
        T_cur = h.shape[2]
        pad_len = (factor - T_cur % factor) % factor
        if pad_len > 0:
            h = F.pad(h, (0, pad_len))
        skips = []
        for block, down in zip(self.encoder_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h, t_emb)
        for up, dec_block, skip in zip(self.upsamples, self.decoder_blocks, reversed(skips)):
            h = up(h)
            if h.shape[2] != skip.shape[2]:
                h = h[:, :, :skip.shape[2]]
            h = torch.cat([h, skip], dim=1)
            h = dec_block[0](h)
            h = dec_block[1](h, t_emb)
        h = self.output_proj(h)
        h = h[:, :, :T_orig].transpose(1, 2)
        h = h * mask.unsqueeze(-1)
        return h
