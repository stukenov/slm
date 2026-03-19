"""Mimi codec wrapper: encode audio to latents, decode latents to waveform."""

from __future__ import annotations

import torch
import torch.nn as nn


class MimiCodec(nn.Module):
    """Wrapper around Kyutai's Mimi neural audio codec.

    Encodes 24kHz audio into continuous latent representations
    and decodes them back to waveform. Codec weights are frozen.
    """

    def __init__(self, model_name: str = "kyutai/mimi", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the Mimi model."""
        if self._model is not None:
            return
        try:
            from moshi.models import loaders
            self._model = loaders.get_mimi(self.model_name, device=self.device)
            self._model.set_num_codebooks(8)
            for p in self._model.parameters():
                p.requires_grad = False
        except ImportError:
            raise ImportError(
                "moshi package required for Mimi codec. "
                "Install with: pip install moshi"
            )

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveform to codec codes.

        Args:
            waveform: (B, T) or (B, 1, T) audio at 24kHz

        Returns:
            codes: codec output tensor
        """
        self._load_model()
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        waveform = waveform.to(self.device)
        return self._model.encode(waveform)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes back to waveform.

        Args:
            codes: codec output from encode

        Returns:
            waveform: (B, 1, T) at 24kHz
        """
        self._load_model()
        codes = codes.to(self.device)
        return self._model.decode(codes)

    @property
    def sample_rate(self) -> int:
        return 24000
