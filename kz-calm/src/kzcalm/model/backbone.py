"""Transformer backbone for text-conditioned latent generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class TTSBackbone(nn.Module):
    """Transformer that takes text embeddings + noisy latents + timestep
    and predicts the flow velocity (or denoised target).

    Architecture:
        - Text encoder: embedding + positional + transformer layers
        - Cross-attention from latent sequence to text
        - Timestep conditioning via adaptive layer norm
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 256,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
        max_text_len: int = 512,
        max_audio_frames: int = 1500,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Text encoder
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.text_pos = nn.Embedding(max_text_len, d_model)

        # Latent input projection
        self.latent_in = nn.Linear(latent_dim, d_model)
        self.latent_pos = nn.Embedding(max_audio_frames, d_model)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Transformer decoder layers (self-attn on latents + cross-attn to text)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Text encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Output projection back to latent dim
        self.latent_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        text_tokens: torch.Tensor,
        timestep: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        latent_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_latents: (B, N, D_latent) noisy codec latents
            text_tokens: (B, S) text token ids
            timestep: (B,) diffusion timestep in [0, 1]
            text_padding_mask: (B, S) True for padded positions
            latent_padding_mask: (B, N) True for padded positions

        Returns:
            velocity: (B, N, D_latent) predicted flow velocity
        """
        B, N, _ = noisy_latents.shape

        # Encode text
        text_emb = self.text_embed(text_tokens) + self.text_pos(
            torch.arange(text_tokens.shape[1], device=text_tokens.device)
        )
        text_hidden = self.text_encoder(text_emb, src_key_padding_mask=text_padding_mask)

        # Project latents
        lat_emb = self.latent_in(noisy_latents) + self.latent_pos(
            torch.arange(N, device=noisy_latents.device)
        )

        # Add timestep conditioning
        t_emb = self.time_embed(timestep)  # (B, d_model)
        lat_emb = lat_emb + t_emb.unsqueeze(1)

        # Decode
        out = self.decoder(
            lat_emb,
            text_hidden,
            tgt_key_padding_mask=latent_padding_mask,
            memory_key_padding_mask=text_padding_mask,
        )

        return self.latent_out(out)
