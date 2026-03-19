import torch
from kzcalm.model.matcha import MatchaTTS


def test_training_step_backward():
    model = MatchaTTS(
        vocab_size=100, mel_dim=100, encoder_dim=128,
        encoder_layers=1, encoder_heads=2, encoder_ff=256,
        unet_channels=[64, 64, 128], dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B, S, T = 2, 5, 32
    text = torch.randint(0, 100, (B, S))
    text_mask = torch.ones(B, S)
    mel = torch.randn(B, T, 100)
    mel_mask = torch.ones(B, T)

    flow_loss, dur_loss = model(text, text_mask, mel, mel_mask)
    loss = flow_loss + dur_loss
    loss.backward()
    optimizer.step()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients computed"
