import torch
from omniaudio.model_v2 import RotaryEmbedding, AudioEncoderV2, AudioProjectorV2, OmniAudioV2Model


def test_rotary_embedding_shape():
    rope = RotaryEmbedding(dim=64)
    cos, sin = rope(seq_len=100)
    assert cos.shape == (100, 64)
    assert sin.shape == (100, 64)


def test_rotary_embedding_different_lengths():
    rope = RotaryEmbedding(dim=64)
    cos1, sin1 = rope(seq_len=50)
    cos2, sin2 = rope(seq_len=100)
    assert torch.allclose(cos1, cos2[:50])


def test_encoder_v2_small_shape():
    enc = AudioEncoderV2(n_mels=80, d_model=256, n_heads=4, n_layers=6, n_conv=2)
    mel = torch.randn(2, 80, 1000)
    out = enc(mel)
    assert out.dim() == 3
    assert out.size(0) == 2
    assert out.size(2) == 256
    assert out.size(1) == 250  # 1000 / 4x


def test_encoder_v2_medium_shape():
    enc = AudioEncoderV2(n_mels=80, d_model=384, n_heads=6, n_layers=8, n_conv=3)
    mel = torch.randn(2, 80, 1000)
    out = enc(mel)
    assert out.size(2) == 384
    assert out.size(1) == 125  # 1000 / 8x


def test_encoder_v2_param_count_small():
    enc = AudioEncoderV2(n_mels=80, d_model=256, n_heads=4, n_layers=6, n_conv=2)
    total = sum(p.numel() for p in enc.parameters())
    assert 5_000_000 < total < 15_000_000, f"Small encoder params {total}"


def test_encoder_v2_param_count_medium():
    enc = AudioEncoderV2(n_mels=80, d_model=384, n_heads=6, n_layers=8, n_conv=3)
    total = sum(p.numel() for p in enc.parameters())
    assert 15_000_000 < total < 35_000_000, f"Medium encoder params {total}"


def test_projector_v2():
    proj = AudioProjectorV2(audio_dim=256, llm_dim=768)
    x = torch.randn(2, 100, 256)
    out = proj(x)
    assert out.shape == (2, 100, 768)


def test_omniaudio_v2_forward_ctc():
    model = OmniAudioV2Model(
        encoder_config=dict(n_mels=80, d_model=256, n_heads=4, n_layers=2, n_conv=2),
        llm_name=None, vocab_size=100,
    )
    mel = torch.randn(2, 80, 500)
    targets = torch.randint(1, 100, (2, 20))
    target_lengths = torch.tensor([20, 15])
    loss = model.forward_ctc(mel, targets, target_lengths)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_omniaudio_v2_forward_e2e():
    from transformers import LlamaConfig, LlamaForCausalLM
    tiny_config = LlamaConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    tiny_llm = LlamaForCausalLM(tiny_config)
    model = OmniAudioV2Model(
        encoder_config=dict(n_mels=80, d_model=32, n_heads=2, n_layers=2, n_conv=2),
        llm_name=None, vocab_size=100, llm_dim=64,
    )
    model.llm = tiny_llm
    for p in model.llm.parameters():
        p.requires_grad = False
    mel = torch.randn(2, 80, 200)
    text_ids = torch.randint(0, 100, (2, 10))
    loss = model.forward_e2e(mel, text_ids)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_omniaudio_v2_generate():
    from transformers import LlamaConfig, LlamaForCausalLM
    tiny_config = LlamaConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    tiny_llm = LlamaForCausalLM(tiny_config)
    model = OmniAudioV2Model(
        encoder_config=dict(n_mels=80, d_model=32, n_heads=2, n_layers=2, n_conv=2),
        llm_name=None, vocab_size=100, llm_dim=64,
    )
    model.llm = tiny_llm
    mel = torch.randn(1, 80, 200)
    tokens = model.generate(mel, max_new_tokens=10, eos_token_id=0)
    assert isinstance(tokens, list)
    assert len(tokens) <= 10
