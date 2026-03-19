"""Voice prompt conditioning for zero-shot speaker transfer (P1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class VoicePromptEncoder(nn.Module):
    """Encodes voice prompt codec latents into a speaker conditioning vector.

    Takes a few seconds of reference audio (already encoded to codec latents)
    and produces a fixed-size speaker embedding that conditions the backbone.
    """

    def __init__(self, latent_dim: int = 256, d_model: int = 1024, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(latent_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, prompt_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompt_latents: (B, N_prompt, D_latent) codec latents of reference audio

        Returns:
            speaker_emb: (B, d_model) speaker conditioning vector
        """
        x = self.proj(prompt_latents)
        x = self.encoder(x)
        # Mean pooling
        x = x.mean(dim=1)
        return self.pool(x)
