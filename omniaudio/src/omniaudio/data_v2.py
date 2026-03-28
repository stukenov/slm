"""Data loading and collation for OmniAudio v2."""

import random

import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from omniaudio.augment import spec_augment, speed_perturb


def load_speech_dataset(name="fleurs", split="train", max_samples=None):
    """Load a Kazakh speech dataset. Supports 'fleurs' and 'common_voice'."""
    if name == "fleurs":
        ds = load_dataset("google/fleurs", "kk_kz", split=split, trust_remote_code=True)
        # Normalize column names: FLEURS uses 'transcription', we need 'sentence'
        ds = ds.rename_column("transcription", "sentence")
    elif name == "common_voice":
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "kk",
                          split=split, trust_remote_code=True)
    else:
        ds = load_dataset(name, split=split, trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


# Backward compat alias
def load_commonvoice_kk(split="train", max_samples=None):
    return load_speech_dataset("common_voice", split=split, max_samples=max_samples)


class AudioCollatorV2:
    """Collate audio+text for OmniAudio v2. Produces mel, text_ids, and CTC targets."""

    SPEED_FACTORS = [0.9, 1.0, 1.1]

    def __init__(self, tokenizer_path: str, n_mels: int = 80, sample_rate: int = 16000,
                 max_audio_len: float = 15.0, max_text_len: int = 256, augment: bool = True):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except (ValueError, OSError):
            # Fallback for transformers 5.1 where AutoTokenizer fails on some configs
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.augment = augment
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160,
        )

    def __call__(self, batch):
        max_audio_samples = int(self.max_audio_len * self.sample_rate)
        mels, text_ids_list, ctc_targets_list = [], [], []

        for sample in batch:
            audio = sample["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)
            sr = audio["sampling_rate"]

            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            if self.augment:
                factor = random.choice(self.SPEED_FACTORS)
                waveform = speed_perturb(waveform, self.sample_rate, factor)

            waveform = waveform[:max_audio_samples]
            mel = self.mel_transform(waveform.unsqueeze(0))
            mel = torch.log(torch.clamp(mel, min=1e-10)).squeeze(0)

            if self.augment:
                mel = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                                   num_freq_masks=2, num_time_masks=2)
            mels.append(mel)

            tokens = self.tokenizer(sample["sentence"], max_length=self.max_text_len,
                                    truncation=True, return_tensors="pt")
            ids = tokens["input_ids"].squeeze(0)
            text_ids_list.append(ids)
            ctc_targets_list.append(ids.clone())

        max_t = max(m.shape[1] for m in mels)
        padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_t)
        for i, m in enumerate(mels):
            padded_mels[i, :, :m.shape[1]] = m

        padded_text = torch.nn.utils.rnn.pad_sequence(text_ids_list, batch_first=True, padding_value=-100)
        ctc_target_lengths = torch.tensor([len(t) for t in ctc_targets_list])
        padded_ctc = torch.nn.utils.rnn.pad_sequence(ctc_targets_list, batch_first=True, padding_value=0)

        return {
            "mel": padded_mels,
            "text_ids": padded_text,
            "ctc_targets": padded_ctc,
            "ctc_target_lengths": ctc_target_lengths,
        }
