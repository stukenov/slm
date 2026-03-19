"""Audio quality control: VAD, SNR, clipping detection."""

from __future__ import annotations

import numpy as np
import torch


def compute_snr(waveform: torch.Tensor | np.ndarray, frame_length: int = 2048) -> float:
    """Estimate SNR in dB using simple energy-based VAD."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    waveform = waveform.astype(np.float32).flatten()

    # Frame energies
    n_frames = len(waveform) // frame_length
    if n_frames < 2:
        return 0.0
    frames = waveform[: n_frames * frame_length].reshape(n_frames, frame_length)
    energies = (frames**2).mean(axis=1)

    threshold = np.median(energies) * 0.1
    speech_energy = energies[energies > threshold].mean() if (energies > threshold).any() else 1e-10
    noise_energy = energies[energies <= threshold].mean() if (energies <= threshold).any() else 1e-10

    return float(10 * np.log10(speech_energy / max(noise_energy, 1e-10)))


def detect_clipping(waveform: torch.Tensor | np.ndarray, threshold: float = 0.99) -> float:
    """Return fraction of samples that are clipped."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    waveform = np.abs(waveform.flatten())
    if len(waveform) == 0:
        return 0.0
    return float((waveform >= threshold).sum() / len(waveform))


def check_audio_quality(
    waveform: torch.Tensor | np.ndarray,
    min_snr: float = 15.0,
    max_clip_fraction: float = 0.001,
) -> dict:
    """Run QC checks, return dict with pass/fail and metrics."""
    snr = compute_snr(waveform)
    clip_frac = detect_clipping(waveform)

    return {
        "snr_db": snr,
        "clip_fraction": clip_frac,
        "snr_pass": snr >= min_snr,
        "clip_pass": clip_frac <= max_clip_fraction,
        "pass": snr >= min_snr and clip_frac <= max_clip_fraction,
    }
