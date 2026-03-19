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
    def __init__(self, vocab_size: int, mel_dim: int = 100, encoder_dim: int = 256,
                 encoder_layers: int = 4, encoder_heads: int = 4, encoder_ff: int = 1024,
                 unet_channels: list[int] | None = None, dropout: float = 0.1, max_text_len: int = 512):
        super().__init__()
        self.mel_dim = mel_dim
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, d_model=encoder_dim, mel_dim=mel_dim,
            num_layers=encoder_layers, num_heads=encoder_heads, d_ff=encoder_ff,
            dropout=dropout, max_text_len=max_text_len,
        )
        self.duration_predictor = DurationPredictor(d_model=encoder_dim, dropout=dropout)
        self.unet = UNet1DDecoder(mel_dim=mel_dim, cond_dim=mel_dim, channels=unet_channels)

    def forward(self, text, text_mask, mel, mel_mask):
        mu, log_sigma, hidden = self.text_encoder(text, text_mask)

        with torch.no_grad():
            durations = monotonic_alignment_search(mu, mel, text_mask, mel_mask)

        # Duration predictor (detached — standard in Matcha-TTS)
        log_dur_pred = self.duration_predictor(hidden.detach(), text_mask)
        log_dur_target = torch.log(durations.float().clamp(min=1))
        dur_loss = F.mse_loss(log_dur_pred * text_mask, log_dur_target * text_mask)

        # Expand mu and log_sigma together (single expand call)
        mu_sigma = torch.cat([mu, log_sigma], dim=-1)  # (B, S, 2*mel_dim)
        mu_sigma_expanded = expand_durations(mu_sigma, durations)
        T = mel.shape[1]
        if mu_sigma_expanded.shape[1] > T:
            mu_sigma_expanded = mu_sigma_expanded[:, :T]
        elif mu_sigma_expanded.shape[1] < T:
            pad = T - mu_sigma_expanded.shape[1]
            mu_sigma_expanded = F.pad(mu_sigma_expanded, (0, 0, 0, pad))
        mu_expanded = mu_sigma_expanded[:, :, :self.mel_dim]
        log_sigma_expanded = mu_sigma_expanded[:, :, self.mel_dim:]

        # Prior loss — forces mu to represent per-phoneme mel means
        # -log N(mel | mu, sigma) = 0.5 * (2*log_sigma + (mel - mu)^2 * exp(-2*log_sigma)) + const
        prior_loss = 0.5 * (2.0 * log_sigma_expanded + (mel - mu_expanded) ** 2 * torch.exp(-2.0 * log_sigma_expanded))
        prior_loss = (prior_loss.mean(dim=-1) * mel_mask).sum() / mel_mask.sum().clamp(min=1)

        # OT-CFM: x0 ~ N(mu, sigma^2 I)
        noise = torch.randn_like(mel)
        x0 = mu_expanded + noise * torch.exp(log_sigma_expanded)
        t = torch.rand(mel.shape[0], device=mel.device)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * x0 + t_expand * mel

        velocity = self.unet(x_t, t, mu_expanded, mel_mask)
        target = mel - x0

        diff = (velocity - target) ** 2
        diff = diff.mean(dim=-1)
        flow_loss = (diff * mel_mask).sum() / mel_mask.sum().clamp(min=1)

        return flow_loss, dur_loss, prior_loss

    @torch.no_grad()
    def synthesize(self, text, text_mask, num_steps=8, length_scale=1.0):
        mu, log_sigma, hidden = self.text_encoder(text, text_mask)

        log_dur = self.duration_predictor(hidden, text_mask)
        durations = torch.round(torch.exp(log_dur) * length_scale).long().clamp(min=1)
        durations = durations * text_mask.long()

        mu_sigma = torch.cat([mu, log_sigma], dim=-1)
        mu_sigma_expanded = expand_durations(mu_sigma, durations)
        T = mu_sigma_expanded.shape[1]
        mu_expanded = mu_sigma_expanded[:, :, :self.mel_dim]
        log_sigma_expanded = mu_sigma_expanded[:, :, self.mel_dim:]
        mel_mask = torch.ones(mu_expanded.shape[0], T, device=text.device)

        # OT-CFM: start from N(mu, sigma^2 I)
        x = mu_expanded + torch.randn_like(mu_expanded) * torch.exp(log_sigma_expanded)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = self.unet(x, t, mu_expanded, mel_mask)
            x = x + v * dt

        return x
