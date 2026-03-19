import torch
from kzcalm.model.unet_1d import UNet1DDecoder


def test_unet_output_shape():
    B, T, D = 2, 64, 100
    unet = UNet1DDecoder(mel_dim=D, cond_dim=D, channels=[256, 256, 512])
    x_t = torch.randn(B, T, D)
    cond = torch.randn(B, T, D)
    t = torch.rand(B)
    mask = torch.ones(B, T)
    out = unet(x_t, t, cond, mask)
    assert out.shape == (B, T, D)


def test_unet_different_lengths():
    B, T, D = 1, 37, 100
    unet = UNet1DDecoder(mel_dim=D, cond_dim=D, channels=[256, 256, 512])
    x_t = torch.randn(B, T, D)
    cond = torch.randn(B, T, D)
    t = torch.rand(B)
    mask = torch.ones(B, T)
    out = unet(x_t, t, cond, mask)
    assert out.shape == (B, T, D)
