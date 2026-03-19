"""Text encoder for Matcha-TTS."""
from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, mel_dim: int = 100,
                 num_layers: int = 4, num_heads: int = 4, d_ff: int = 1024,
                 dropout: float = 0.1, max_text_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_text_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mu_proj = nn.Linear(d_model, mel_dim)
        self.log_sigma_proj = nn.Linear(d_model, mel_dim)

    def forward(self, text: torch.Tensor, text_mask: torch.Tensor):
        S = text.shape[1]
        pos_ids = torch.arange(S, device=text.device)
        h = self.embed(text) + self.pos(pos_ids)
        pad_mask = (text_mask == 0)
        hidden = self.encoder(h, src_key_padding_mask=pad_mask)
        mu = self.mu_proj(hidden)
        log_sigma = self.log_sigma_proj(hidden)
        return mu, log_sigma, hidden
