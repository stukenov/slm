# tests/test_matcha.py
import torch
from kzcalm.model.matcha import MatchaTTS


def test_matcha_forward_training():
    model = MatchaTTS(
        vocab_size=4096, mel_dim=100, encoder_dim=256,
        encoder_layers=2, encoder_heads=4, encoder_ff=512,
        unet_channels=[128, 128, 256], dropout=0.1,
    )
    B, S, T = 2, 8, 32
    text = torch.randint(0, 4096, (B, S))
    text_mask = torch.ones(B, S)
    mel = torch.randn(B, T, 100)
    mel_mask = torch.ones(B, T)

    flow_loss, dur_loss = model(text, text_mask, mel, mel_mask)
    assert flow_loss.shape == ()
    assert dur_loss.shape == ()
    assert flow_loss.item() > 0
    assert flow_loss.requires_grad


def test_matcha_synthesize():
    model = MatchaTTS(
        vocab_size=4096, mel_dim=100, encoder_dim=256,
        encoder_layers=2, encoder_heads=4, encoder_ff=512,
        unet_channels=[128, 128, 256], dropout=0.1,
    )
    model.eval()
    B, S = 1, 5
    text = torch.randint(0, 4096, (B, S))
    text_mask = torch.ones(B, S)

    with torch.no_grad():
        mel = model.synthesize(text, text_mask, num_steps=4)

    assert mel.dim() == 3
    assert mel.shape[0] == B
    assert mel.shape[2] == 100


def test_matcha_param_count():
    model = MatchaTTS(
        vocab_size=4096, mel_dim=100, encoder_dim=256,
        encoder_layers=4, encoder_heads=4, encoder_ff=1024,
        unet_channels=[256, 256, 512], dropout=0.1,
    )
    n = sum(p.numel() for p in model.parameters())
    assert n < 30_000_000, f"Model too large: {n/1e6:.1f}M"
