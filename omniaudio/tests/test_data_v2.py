import torch
from omniaudio.data_v2 import AudioCollatorV2


def _fake_sample(duration_s=1.0, sr=16000, text="test text"):
    return {
        "audio": {"array": torch.randn(int(duration_s * sr)).numpy(), "sampling_rate": sr},
        "sentence": text,
    }


def test_collator_v2_output_keys():
    collator = AudioCollatorV2(
        tokenizer_path="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        n_mels=80, sample_rate=16000, max_audio_len=15.0, max_text_len=256,
        augment=False,
    )
    batch = collator([_fake_sample()])
    assert "mel" in batch
    assert "text_ids" in batch
    assert "ctc_targets" in batch
    assert "ctc_target_lengths" in batch


def test_collator_v2_mel_shape():
    collator = AudioCollatorV2(
        tokenizer_path="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        n_mels=80, sample_rate=16000, max_audio_len=15.0, max_text_len=256,
        augment=False,
    )
    batch = collator([_fake_sample(), _fake_sample()])
    assert batch["mel"].dim() == 3
    assert batch["mel"].size(0) == 2
    assert batch["mel"].size(1) == 80


def test_collator_v2_ctc_targets_valid():
    collator = AudioCollatorV2(
        tokenizer_path="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        n_mels=80, sample_rate=16000, max_audio_len=15.0, max_text_len=256,
        augment=False,
    )
    batch = collator([_fake_sample()])
    assert (batch["ctc_targets"] >= 0).all()
