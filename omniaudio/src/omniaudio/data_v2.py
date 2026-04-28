"""Data loading and collation for OmniAudio v2."""

import csv
import io
import random
import re
import tarfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from omniaudio.augment import spec_augment, speed_perturb


def _normalize_columns(ds):
    """Ensure dataset has 'sentence' column with text."""
    cols = ds.column_names
    if "sentence" in cols:
        return ds
    if "transcription" in cols:
        return ds.rename_column("transcription", "sentence")
    if "text" in cols:
        return ds.rename_column("text", "sentence")
    return ds


def _load_fleurs_without_script(split: str, max_samples=None):
    """Load FLEURS directly from Hub files when dataset scripts are unsupported."""
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    fleurs_split = split_map.get(split, split)
    repo_id = "google/fleurs"
    lang = "kk_kz"

    tsv_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f"data/{lang}/{fleurs_split}.tsv")
    archive_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f"data/{lang}/audio/{fleurs_split}.tar.gz")

    extract_dir = Path(archive_path).with_suffix("").with_suffix("")
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)

    audio_files = {p.name: str(p) for p in extract_dir.rglob("*.wav")}

    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 7:
                continue
            audio_path = audio_files.get(row[1])
            if not audio_path:
                continue
            rows.append({
                "id": int(row[0]),
                "audio": {"path": audio_path, "sampling_rate": 16000},
                "raw_transcription": row[2],
                "sentence": row[3],
                "num_samples": int(row[5]),
                "gender": row[6].lower(),
            })

    if max_samples:
        rows = rows[: min(max_samples, len(rows))]

    return Dataset.from_list(rows)


def _load_sozkz_mels_without_script(split: str, max_samples=None):
    """Load SozKZ mels directly from parquet files, bypassing dataset scripts."""
    repo_id = "stukenov/sozkz-asr-mels-kk-v1"
    cache_dir = Path("./datasets/sozkz-asr-mels-kk-v1")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    token = token_path.read_text().strip() if token_path.exists() else None

    if not cache_dir.exists():
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(cache_dir),
            token=token,
        )

    pattern = str(cache_dir / "data" / f"{split}-*.parquet")
    ds = load_dataset("parquet", data_files={split: pattern}, split=split)
    if "text" in ds.column_names and "sentence" not in ds.column_names:
        ds = ds.rename_column("text", "sentence")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _load_single_dataset(name, split="train", max_samples=None):
    """Load a single Kazakh speech dataset by name."""
    if name == "fleurs":
        try:
            ds = load_dataset("google/fleurs", "kk_kz", split=split, trust_remote_code=True)
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" not in str(e):
                raise
            ds = _load_fleurs_without_script(split, max_samples=max_samples)
            return ds
    elif name == "common_voice":
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "kk",
                          split=split, trust_remote_code=True)
    elif name == "kzcalm":
        ds = load_dataset("stukenov/kzcalm-tts-kk-v1", split="train", trust_remote_code=True)
        n = len(ds)
        if split == "train":
            ds = ds.select(range(int(n * 0.94)))
        elif split == "validation":
            ds = ds.select(range(int(n * 0.94), int(n * 0.99)))
        elif split == "test":
            ds = ds.select(range(int(n * 0.99), n))
    elif name == "openslr140":
        ds = load_dataset("voice-biomarkers/openslr-140-hq-Kazakh", split="train", trust_remote_code=True)
        n = len(ds)
        if split == "train":
            ds = ds.select(range(int(n * 0.94)))
        elif split == "validation":
            ds = ds.select(range(int(n * 0.94), int(n * 0.99)))
        elif split == "test":
            ds = ds.select(range(int(n * 0.99), n))
    elif name == "flamme":
        ds = load_dataset("Flamme-VRM/kazakh-speech-dataset", split="train", trust_remote_code=True)
        n = len(ds)
        if split == "train":
            ds = ds.select(range(int(n * 0.94)))
        elif split == "validation":
            ds = ds.select(range(int(n * 0.94), int(n * 0.99)))
        elif split == "test":
            ds = ds.select(range(int(n * 0.99), n))
    elif name == "sozkz_mels":
        return _load_sozkz_mels_without_script(split, max_samples=max_samples)
    else:
        ds = load_dataset(name, split=split, trust_remote_code=True)
    ds = _normalize_columns(ds)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def load_speech_dataset(name="fleurs", split="train", max_samples=None, **kwargs):
    """Load Kazakh speech dataset(s). Use '+' to combine: 'kzcalm+openslr140+flamme'."""
    if "+" in name:
        from datasets import concatenate_datasets
        parts = [p.strip() for p in name.split("+")]
        datasets = [_load_single_dataset(p, split=split) for p in parts]
        ds = concatenate_datasets(datasets)
        ds = ds.shuffle(seed=42)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds
    return _load_single_dataset(name, split=split, max_samples=max_samples)


# Backward compat alias
def load_commonvoice_kk(split="train", max_samples=None):
    return load_speech_dataset("common_voice", split=split, max_samples=max_samples)


def normalize_asr_text(
    text: str,
    *,
    lowercase: bool = False,
    strip_punctuation: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Normalize transcription text consistently across train and eval."""
    if lowercase:
        text = text.lower()
    if strip_punctuation:
        chars = []
        for ch in text:
            category = unicodedata.category(ch)
            if category.startswith("P") or category.startswith("S"):
                chars.append(" ")
            else:
                chars.append(ch)
        text = "".join(chars)
    if collapse_whitespace:
        text = " ".join(text.split())
    return text.strip()


class AudioCollatorV2:
    """Collate audio+text for OmniAudio v2. Produces mel, text_ids, and CTC targets."""

    SPEED_FACTORS = [0.9, 1.0, 1.1]

    def __init__(self, tokenizer_path: str, n_mels: int = 80, sample_rate: int = 16000,
                 max_audio_len: float = 15.0, max_text_len: int = 256, augment: bool = True,
                 text_lowercase: bool = False, text_strip_punctuation: bool = False,
                 text_collapse_whitespace: bool = True):
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
        self.text_lowercase = text_lowercase
        self.text_strip_punctuation = text_strip_punctuation
        self.text_collapse_whitespace = text_collapse_whitespace
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160,
        )

    def __call__(self, batch):
        max_audio_samples = int(self.max_audio_len * self.sample_rate)
        mels, text_ids_list, ctc_targets_list = [], [], []

        for sample in batch:
            audio = sample["audio"]
            if isinstance(audio, dict) and "array" in audio:
                waveform = torch.tensor(audio["array"], dtype=torch.float32)
                sr = audio["sampling_rate"]
            else:
                audio_path = audio["path"] if isinstance(audio, dict) else audio
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    if waveform.ndim == 2 and waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=False)
                    elif waveform.ndim == 2:
                        waveform = waveform.squeeze(0)
                except ImportError:
                    waveform_np, sr = sf.read(audio_path, dtype="float32")
                    if waveform_np.ndim == 2:
                        waveform_np = waveform_np.mean(axis=1)
                    waveform = torch.from_numpy(waveform_np)

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

            text = normalize_asr_text(
                sample["sentence"],
                lowercase=self.text_lowercase,
                strip_punctuation=self.text_strip_punctuation,
                collapse_whitespace=self.text_collapse_whitespace,
            )
            tokens = self.tokenizer(text, max_length=self.max_text_len - 1,
                                    truncation=True, return_tensors="pt")
            ids = tokens["input_ids"].squeeze(0)
            # Append EOS so model learns to stop generating
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and (len(ids) == 0 or ids[-1].item() != eos_id):
                ids = torch.cat([ids, torch.tensor([eos_id])])
            text_ids_list.append(ids)
            ctc_targets_list.append(ids[:-1].clone())  # CTC targets without EOS

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


class PrecomputedMelCollator:
    """Collate pre-computed mel spectrograms (stored as npy bytes in HF dataset).

    Mels are stored as numpy .npy format bytes (no pickle - just standard numpy arrays).
    """

    _TAG_RE = re.compile(r'<[^>]+>')

    def __init__(self, tokenizer_path: str, max_audio_len: float = 30.0,
                 max_text_len: int = 256, augment: bool = True, sample_rate: int = 16000,
                 text_lowercase: bool = False, text_strip_punctuation: bool = False,
                 text_collapse_whitespace: bool = True):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except (ValueError, OSError):
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_mel_frames = int(max_audio_len * sample_rate / 160)  # hop_length=160
        self.max_text_len = max_text_len
        self.augment = augment
        self.text_lowercase = text_lowercase
        self.text_strip_punctuation = text_strip_punctuation
        self.text_collapse_whitespace = text_collapse_whitespace

    def __call__(self, batch):
        mels, text_ids_list, ctc_targets_list = [], [], []

        for sample in batch:
            # Decode mel from npy bytes (standard numpy format, no pickle)
            mel_bytes = sample["mel"]
            mel = np.load(io.BytesIO(mel_bytes))
            mel = torch.from_numpy(mel.astype(np.float32))  # (n_mels, time)

            # Truncate if needed
            if mel.shape[1] > self.max_mel_frames:
                mel = mel[:, :self.max_mel_frames]

            # Augment
            if self.augment:
                mel = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                                   num_freq_masks=2, num_time_masks=2)
            mels.append(mel)

            # Clean tags (e.g. <marzhan> <fearful> from KazEmoTTS) and tokenize
            text = self._TAG_RE.sub('', sample["sentence"]).strip()
            text = normalize_asr_text(
                text,
                lowercase=self.text_lowercase,
                strip_punctuation=self.text_strip_punctuation,
                collapse_whitespace=self.text_collapse_whitespace,
            )
            tokens = self.tokenizer(text, max_length=self.max_text_len - 1,
                                    truncation=True, return_tensors="pt")
            ids = tokens["input_ids"].squeeze(0)
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and (len(ids) == 0 or ids[-1].item() != eos_id):
                ids = torch.cat([ids, torch.tensor([eos_id])])
            text_ids_list.append(ids)
            ctc_targets_list.append(ids[:-1].clone())

        # Pad mels
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
