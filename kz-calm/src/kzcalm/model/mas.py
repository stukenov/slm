"""Monotonic Alignment Search for Matcha-TTS."""
from __future__ import annotations

import numpy as np
import torch

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


@torch.no_grad()
def monotonic_alignment_search(
    mu: torch.Tensor,
    mel: torch.Tensor,
    text_mask: torch.Tensor,
    mel_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MAS alignment between text features and mel.

    Uses log-likelihood based on Gaussian assumption:
        log p(mel_t | mu_s) = -0.5 * ||mel_t - mu_s||^2

    Args:
        mu: (B, S, D) text encoder output (per-phoneme mel statistics)
        mel: (B, T, D) mel spectrogram
        text_mask: (B, S) 1=valid, 0=pad
        mel_mask: (B, T) 1=valid, 0=pad

    Returns:
        durations: (B, S) frames per text token, sums to T per sample
    """
    B, S, D = mu.shape
    T = mel.shape[1]

    mu_exp = mu.unsqueeze(2)
    mel_exp = mel.unsqueeze(1)
    log_prob = -0.5 * ((mu_exp - mel_exp) ** 2).sum(dim=-1)

    if _HAS_NUMBA:
        log_prob_np = log_prob.cpu().float().numpy()
        s_lens = text_mask.sum(dim=-1).cpu().long().numpy()
        t_lens = mel_mask.sum(dim=-1).cpu().long().numpy()
        dur_np = _mas_batch_numba(log_prob_np, s_lens, t_lens)
        durations = torch.from_numpy(dur_np).to(device=mu.device)
    else:
        durations = torch.zeros(B, S, dtype=torch.long, device=mu.device)
        for b in range(B):
            s_len = int(text_mask[b].sum().item())
            t_len = int(mel_mask[b].sum().item())
            if s_len == 0 or t_len == 0:
                continue
            lp = log_prob[b, :s_len, :t_len]
            durations[b, :s_len] = _mas_single(lp, s_len, t_len)

    # Clamp degenerate alignments: min 2 frames per token, max 50% of total
    durations = _clamp_durations(durations, text_mask, mel_mask)
    return durations


def _clamp_durations(
    durations: torch.Tensor,
    text_mask: torch.Tensor,
    mel_mask: torch.Tensor,
    min_dur: int = 2,
) -> torch.Tensor:
    """Prevent degenerate MAS: clamp min/max durations, redistribute to match total."""
    B, S = durations.shape
    for b in range(B):
        s_len = int(text_mask[b].sum().item())
        t_len = int(mel_mask[b].sum().item())
        if s_len == 0 or t_len == 0:
            continue

        durs = durations[b, :s_len].clone()
        max_dur = max(t_len // s_len * 4, min_dur + 1)

        # Clamp
        durs = durs.clamp(min=min_dur, max=max_dur)

        # Redistribute to match t_len
        diff = t_len - durs.sum().item()
        if diff > 0:
            # Spread extra frames proportionally to original durations
            orig = durations[b, :s_len].float()
            orig = orig / orig.sum().clamp(min=1)
            extra = (orig * diff).long()
            extra[extra.argmax()] += diff - extra.sum().item()
            durs += extra
        elif diff < 0:
            # Remove excess from largest durations
            for _ in range(-diff):
                idx = durs.argmax()
                if durs[idx] > min_dur:
                    durs[idx] -= 1

        durations[b, :s_len] = durs

    return durations


def _mas_single(log_prob: torch.Tensor, S: int, T: int) -> torch.Tensor:
    Q = torch.full((S, T), -1e9, device=log_prob.device)

    Q[0, 0] = log_prob[0, 0]
    for t in range(1, T):
        Q[0, t] = Q[0, t - 1] + log_prob[0, t]

    for s in range(1, S):
        for t in range(s, T):
            Q[s, t] = torch.max(Q[s, t - 1], Q[s - 1, t - 1]) + log_prob[s, t]

    durations = torch.zeros(S, dtype=torch.long, device=log_prob.device)
    s = S - 1
    t = T - 1
    durations[s] = 1

    while t > 0:
        if s > 0 and Q[s - 1, t - 1] >= Q[s, t - 1]:
            s -= 1
            durations[s] = 1
        else:
            durations[s] += 1
        t -= 1

    return durations


if _HAS_NUMBA:
    @njit()
    def _mas_single_numba(log_prob, S, T):
        Q = np.full((S, T), -1e9, dtype=np.float32)
        Q[0, 0] = log_prob[0, 0]
        for t in range(1, T):
            Q[0, t] = Q[0, t - 1] + log_prob[0, t]
        for s in range(1, S):
            for t in range(s, T):
                stay = Q[s, t - 1]
                move = Q[s - 1, t - 1]
                Q[s, t] = (stay if stay > move else move) + log_prob[s, t]

        durations = np.zeros(S, dtype=np.int64)
        s = S - 1
        t = T - 1
        durations[s] = 1
        while t > 0:
            if s > 0 and Q[s - 1, t - 1] >= Q[s, t - 1]:
                s -= 1
                durations[s] = 1
            else:
                durations[s] += 1
            t -= 1
        return durations

    @njit()
    def _mas_batch_numba(log_prob, s_lens, t_lens):
        B, S, T = log_prob.shape
        durations = np.zeros((B, S), dtype=np.int64)
        for b in range(B):
            sl = s_lens[b]
            tl = t_lens[b]
            if sl > 0 and tl > 0:
                durations[b, :sl] = _mas_single_numba(log_prob[b, :sl, :tl], sl, tl)
        return durations


def expand_durations(features: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Expand features by durations. Vectorized via torch.repeat_interleave."""
    B, S, D = features.shape
    T = durations.sum(dim=-1).max().item()
    if T == 0:
        return torch.zeros(B, 0, D, device=features.device, dtype=features.dtype)

    expanded = torch.zeros(B, T, D, device=features.device, dtype=features.dtype)
    for b in range(B):
        durs_b = durations[b, :S]
        total = durs_b.sum().item()
        if total > 0:
            exp_b = torch.repeat_interleave(features[b], durs_b, dim=0)
            expanded[b, :exp_b.shape[0]] = exp_b

    return expanded
