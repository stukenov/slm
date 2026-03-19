import torch
from omniaudio.model import AudioEncoder, AudioProjector, OmniAudioModel


def test_audio_encoder_output_shape():
    enc = AudioEncoder(n_mels=80, d_model=384)
    mel = torch.randn(2, 80, 3000)
    out = enc(mel)
    assert out.dim() == 3
    assert out.size(0) == 2
    assert out.size(2) == 384


def test_projector():
    proj = AudioProjector(audio_dim=384, llm_dim=576)
    x = torch.randn(2, 100, 384)
    out = proj(x)
    assert out.shape == (2, 100, 576)


def test_omni_audio_forward():
    model = OmniAudioModel()
    mel = torch.randn(2, 80, 3000)
    text_ids = torch.randint(0, 32000, (2, 50))
    loss = model(mel, text_ids)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_param_count():
    model = OmniAudioModel()
    total = sum(p.numel() for p in model.parameters())
    assert 50_000_000 < total < 90_000_000, f"Param count {total} out of range"
