import torch
from kzcalm.model.text_encoder import TextEncoder


def test_text_encoder_shape():
    B, S = 2, 10
    enc = TextEncoder(vocab_size=4096, d_model=256, mel_dim=100, num_layers=2, num_heads=4)
    text = torch.randint(0, 4096, (B, S))
    mask = torch.ones(B, S)
    mu, log_sigma, hidden = enc(text, mask)
    assert mu.shape == (B, S, 100)
    assert log_sigma.shape == (B, S, 100)
    assert hidden.shape == (B, S, 256)
