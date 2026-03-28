"""Data augmentation for audio: SpecAugment and speed perturbation."""

import random

import torch
import torchaudio


def spec_augment(mel: torch.Tensor, freq_mask_param: int = 27, time_mask_param: int = 100,
                 num_freq_masks: int = 2, num_time_masks: int = 2) -> torch.Tensor:
    """Apply SpecAugment to mel spectrogram. mel: (n_mels, time)."""
    augmented = mel.clone()
    n_mels, n_time = augmented.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, n_mels - 1))
        f0 = random.randint(0, n_mels - f)
        augmented[f0:f0 + f, :] = 0

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, n_time - 1))
        t0 = random.randint(0, n_time - t)
        augmented[:, t0:t0 + t] = 0

    return augmented


def speed_perturb(waveform: torch.Tensor, sample_rate: int = 16000, factor: float = 1.0) -> torch.Tensor:
    """Speed perturbation via resampling. waveform: (samples,)."""
    if factor == 1.0:
        return waveform
    new_sr = int(sample_rate * factor)
    return torchaudio.functional.resample(waveform, new_sr, sample_rate)
