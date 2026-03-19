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
        h = x.transpose(1, 2)
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                h = layer(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = layer(h)
        h = h.transpose(1, 2)
        log_dur = self.proj(h).squeeze(-1)
        return log_dur * mask
