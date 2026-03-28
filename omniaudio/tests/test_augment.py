import torch
from omniaudio.augment import spec_augment, speed_perturb


def test_spec_augment_shape():
    mel = torch.randn(80, 500)
    augmented = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                             num_freq_masks=2, num_time_masks=2)
    assert augmented.shape == mel.shape


def test_spec_augment_has_zeros():
    torch.manual_seed(42)
    mel = torch.ones(80, 500)
    augmented = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                             num_freq_masks=2, num_time_masks=2)
    assert (augmented == 0).any()


def test_spec_augment_no_masks():
    mel = torch.randn(80, 500)
    augmented = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                             num_freq_masks=0, num_time_masks=0)
    assert torch.equal(mel, augmented)


def test_speed_perturb_shape():
    waveform = torch.randn(16000)
    perturbed = speed_perturb(waveform, sample_rate=16000, factor=0.9)
    assert perturbed.shape[0] > waveform.shape[0]


def test_speed_perturb_identity():
    waveform = torch.randn(16000)
    perturbed = speed_perturb(waveform, sample_rate=16000, factor=1.0)
    assert perturbed.shape[0] == waveform.shape[0]
