"""Mel spectrogram extraction for TTS flow matching."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class MelExtractor(nn.Module):
    """Extract log-mel spectrograms from waveforms.

    Output: (T, n_mels) log-mel frames, globally normalized.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        # Global normalization stats (log-mel on speech data)
        # Will be updated from data if needed; reasonable defaults for 24kHz speech
        self.register_buffer("mel_mean", torch.tensor(-1.42))
        self.register_buffer("mel_std", torch.tensor(3.80))

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract normalized log-mel spectrogram.

        Args:
            waveform: (T_samples,) or (1, T_samples) mono audio

        Returns:
            mel: (T_frames, n_mels) normalized log-mel
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel_spec(waveform)  # (1, n_mels, T)
        mel = torch.log(mel.clamp(min=1e-5))
        mel = (mel - self.mel_mean) / self.mel_std
        return mel.squeeze(0).transpose(0, 1)  # (T, n_mels)
