"""Data loading and collation for OmniAudio Kazakh ASR."""

import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoTokenizer


def load_commonvoice_kk(split="train", max_samples=None, streaming=False):
    """Load Common Voice 17.0 Kazakh split."""
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "kk",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    if max_samples and not streaming:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


class AudioCollator:
    """Collate audio+text samples into padded batches for training."""

    def __init__(
        self,
        tokenizer_path: str,
        n_mels: int = 80,
        sample_rate: int = 16000,
        max_audio_len: float = 30.0,
        max_text_len: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160,
        )

    def __call__(self, batch):
        max_audio_samples = int(self.max_audio_len * self.sample_rate)
        mels = []
        text_ids_list = []

        for sample in batch:
            # Audio processing
            audio = sample["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)
            sr = audio["sampling_rate"]

            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            # Truncate
            waveform = waveform[:max_audio_samples]

            # Mel spectrogram + log
            mel = self.mel_transform(waveform.unsqueeze(0))  # (1, n_mels, T)
            mel = torch.log(torch.clamp(mel, min=1e-10))
            mels.append(mel.squeeze(0))  # (n_mels, T)

            # Tokenize text
            tokens = self.tokenizer(
                sample["sentence"],
                max_length=self.max_text_len,
                truncation=True,
                return_tensors="pt",
            )
            text_ids_list.append(tokens["input_ids"].squeeze(0))

        # Pad mels to max time length in batch
        max_t = max(m.shape[1] for m in mels)
        padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_t)
        for i, m in enumerate(mels):
            padded_mels[i, :, : m.shape[1]] = m

        # Pad text ids with -100
        padded_text = torch.nn.utils.rnn.pad_sequence(
            text_ids_list, batch_first=True, padding_value=-100
        )

        return {"mel": padded_mels, "text_ids": padded_text}
