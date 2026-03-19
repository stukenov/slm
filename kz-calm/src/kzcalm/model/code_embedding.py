"""Extract pre-decoder latents from Mimi: codes (B,8,T) -> (B,2T,512)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MimiLatentExtractor(nn.Module):
    """Extracts pre-decoder continuous latents from Mimi by hooking into decode().

    Mimi's decode path: codes -> quantizer (lookup + transformer + upsample 2x)
    -> 512-dim latents -> decoder -> waveform.

    We hook the decoder to capture its input: (B, 512, 2T) tensor with
    range ~[-5, 5], which is the correct target for flow matching.
    """

    def __init__(self, mimi_model_name: str = "kyutai/mimi", device: str = "cpu"):
        super().__init__()
        self._load_mimi(mimi_model_name, device)

    def _load_mimi(self, model_name: str, device: str):
        from transformers import MimiModel

        logger.info(f"Loading Mimi model from {model_name} for latent extraction...")
        self.mimi = MimiModel.from_pretrained(model_name).to(device)
        for p in self.mimi.parameters():
            p.requires_grad = False

        self._captured = {}

        # Hook to capture decoder input
        orig_decoder_forward = self.mimi.decoder.forward

        def hooked_forward(hidden_states):
            self._captured["decoder_input"] = hidden_states.detach()
            return orig_decoder_forward(hidden_states)

        self.mimi.decoder.forward = hooked_forward
        logger.info("Mimi loaded with decoder hook (target: 512-dim, 2x upsampled)")

    @torch.no_grad()
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: (B, K, T) integer codes (K=8 codebooks)

        Returns:
            latents: (B, 2T, 512) pre-decoder continuous latents
        """
        self.mimi.decode(audio_codes=codes)
        # captured shape: (B, 512, 2T)
        latents = self._captured["decoder_input"]
        return latents.transpose(1, 2)  # (B, 2T, 512)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode continuous latents directly through Mimi decoder.

        Args:
            latents: (B, 2T, 512) continuous latents

        Returns:
            waveform: (B, 1, num_samples) at 24kHz
        """
        lat_t = latents.transpose(1, 2)  # (B, 512, 2T)
        return self.mimi.decoder(lat_t)

    @property
    def latent_dim(self) -> int:
        return 512

    @property
    def upsample_factor(self) -> int:
        return 2
