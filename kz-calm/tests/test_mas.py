# tests/test_mas.py
import torch
from kzcalm.model.mas import monotonic_alignment_search, expand_durations


def test_mas_output_shape():
    """MAS returns durations that sum to T_mel."""
    B, S, T, D = 2, 5, 20, 100
    mu = torch.randn(B, S, D)
    mel = torch.randn(B, T, D)
    text_mask = torch.ones(B, S)
    mel_mask = torch.ones(B, T)

    durations = monotonic_alignment_search(mu, mel, text_mask, mel_mask)

    assert durations.shape == (B, S)
    assert durations.dtype == torch.long
    for b in range(B):
        assert durations[b].sum().item() == T


def test_mas_monotonic():
    """Each text token gets at least 1 frame, monotonically."""
    B, S, T, D = 1, 3, 10, 16
    mu = torch.randn(B, S, D)
    mel = torch.randn(B, T, D)
    text_mask = torch.ones(B, S)
    mel_mask = torch.ones(B, T)

    durations = monotonic_alignment_search(mu, mel, text_mask, mel_mask)
    assert (durations >= 1).all(), "Every token must have at least 1 frame"


def test_expand_durations():
    """Expand features using durations."""
    B, S, D = 1, 3, 8
    features = torch.tensor([[[1.0]*D, [2.0]*D, [3.0]*D]])
    durations = torch.tensor([[2, 3, 1]])

    expanded = expand_durations(features, durations)
    assert expanded.shape == (1, 6, 8)
    assert (expanded[0, 0] == 1.0).all()
    assert (expanded[0, 1] == 1.0).all()
    assert (expanded[0, 2] == 2.0).all()
    assert (expanded[0, 4] == 2.0).all()
    assert (expanded[0, 5] == 3.0).all()
