"""TTS datasets: codes streaming, pre-extracted latents, and mel spectrograms."""

from __future__ import annotations

import glob
import random

import torch
import torchaudio
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset

from kzcalm.codec.mel import MelExtractor


class CodesDataset(IterableDataset):
    """Streams (text, codes) pairs from a HuggingFace codes dataset."""

    def __init__(
        self,
        hf_dataset: str,
        tokenizer,
        split: str = "train",
        max_text_len: int = 512,
        max_audio_frames: int = 1500,
        num_codebooks: int = 8,
        streaming: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_frames = max_audio_frames
        self.num_codebooks = num_codebooks
        self.ds = load_dataset(hf_dataset, split=split, streaming=streaming)

    def __iter__(self):
        for sample in self.ds:
            text_ids = self.tokenizer.encode(sample["text"])
            if len(text_ids) > self.max_text_len:
                text_ids = text_ids[: self.max_text_len]

            codes = sample["codes"]
            num_frames = sample["num_frames"]
            if num_frames > self.max_audio_frames:
                continue

            codes_tensor = torch.tensor(
                codes[: self.num_codebooks], dtype=torch.long
            )

            yield {
                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                "codes": codes_tensor,
            }


class LatentDataset(Dataset):
    """Loads pre-extracted 512-dim Mimi latents from .pt shard files."""

    def __init__(
        self,
        latent_dir: str,
        tokenizer,
        max_text_len: int = 512,
        max_latent_frames: int = 3000,
    ):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_latent_frames = max_latent_frames

        # Load all shards into memory
        shard_files = sorted(glob.glob(f"{latent_dir}/shard_*.pt"))
        self.items = []
        for sf in shard_files:
            self.items.extend(torch.load(sf, weights_only=False))

        # Filter by max length
        self.items = [
            it for it in self.items
            if it["num_latent_frames"] <= max_latent_frames
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        text_ids = self.tokenizer.encode(item["text"])
        if len(text_ids) > self.max_text_len:
            text_ids = text_ids[: self.max_text_len]

        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "latents": item["latents"].float(),  # (2T, 512)
        }


class MelDataset(IterableDataset):
    """Streams audio from HF dataset, extracts mel spectrograms on the fly."""

    def __init__(
        self,
        hf_dataset: str,
        tokenizer,
        mel_extractor: MelExtractor,
        split: str = "train",
        max_text_len: int = 512,
        max_mel_frames: int = 3000,
        min_mel_frames: int = 10,
        streaming: bool = True,
        max_samples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.mel_extractor = mel_extractor
        self.max_text_len = max_text_len
        self.max_mel_frames = max_mel_frames
        self.min_mel_frames = min_mel_frames
        self.max_samples = max_samples
        self.target_sr = mel_extractor.sample_rate
        self.ds = load_dataset(hf_dataset, split=split, streaming=streaming)
        self._cache: list[dict] | None = None

    def __iter__(self):
        if self._cache is not None:
            items = list(self._cache)
            random.shuffle(items)
            yield from items
            return

        cache = []
        count = 0
        for sample in self.ds:
            if self.max_samples > 0 and count >= self.max_samples:
                break

            text = sample.get("text") or sample.get("sentence", "")
            if not text:
                continue

            text_ids = self.tokenizer.encode(text)
            if len(text_ids) > self.max_text_len:
                text_ids = text_ids[: self.max_text_len]

            audio = sample["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)
            sr = audio["sampling_rate"]

            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

            mel = self.mel_extractor(waveform)  # (T, n_mels)

            if mel.shape[0] > self.max_mel_frames or mel.shape[0] < self.min_mel_frames:
                continue

            item = {
                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                "mel": mel,
            }
            cache.append(item)
            yield item
            count += 1

        if self.max_samples > 0:
            self._cache = cache


class PreExtractedMelDataset(Dataset):
    """Loads pre-extracted mel spectrograms from .pt shard files."""

    def __init__(self, mel_dir: str, max_mel_frames: int = 3000):
        shard_files = sorted(glob.glob(f"{mel_dir}/shard_*.pt"))
        self.items = []
        for sf in shard_files:
            self.items.extend(torch.load(sf, weights_only=False))
        self.items = [it for it in self.items if it["mel"].shape[0] <= max_mel_frames]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            "text_ids": item["text_ids"],
            "mel": item["mel"].float(),
        }


def mel_collate_fn(batch: list[dict]) -> dict:
    """Collate for MelDataset: pad text_ids and mel, create masks."""
    text_ids = [item["text_ids"] for item in batch]
    mels = [item["mel"] for item in batch]  # each (T_i, 80)

    max_text = max(t.shape[0] for t in text_ids)
    max_mel = max(m.shape[0] for m in mels)
    B = len(batch)
    D = mels[0].shape[1]  # 80

    text_padded = torch.zeros(B, max_text, dtype=torch.long)
    text_mask = torch.ones(B, max_text, dtype=torch.bool)
    for i, t in enumerate(text_ids):
        text_padded[i, : t.shape[0]] = t
        text_mask[i, : t.shape[0]] = False

    mel_padded = torch.zeros(B, max_mel, D)
    mel_mask = torch.zeros(B, max_mel)
    for i, m in enumerate(mels):
        T = m.shape[0]
        mel_padded[i, :T, :] = m
        mel_mask[i, :T] = 1.0

    return {
        "text_ids": text_padded,
        "text_mask": text_mask,
        "mel": mel_padded,
        "mel_mask": mel_mask,
    }


def collate_fn(batch: list[dict]) -> dict:
    """Pad text_ids and codes to batch max, create masks."""
    text_ids = [item["text_ids"] for item in batch]
    codes = [item["codes"] for item in batch]

    max_text = max(t.shape[0] for t in text_ids)
    B = len(batch)
    text_padded = torch.zeros(B, max_text, dtype=torch.long)
    text_mask = torch.ones(B, max_text, dtype=torch.bool)
    for i, t in enumerate(text_ids):
        text_padded[i, : t.shape[0]] = t
        text_mask[i, : t.shape[0]] = False

    K = codes[0].shape[0]
    max_frames = max(c.shape[1] for c in codes)
    codes_padded = torch.zeros(B, K, max_frames, dtype=torch.long)
    codes_mask = torch.zeros(B, max_frames)
    for i, c in enumerate(codes):
        T = c.shape[1]
        codes_padded[i, :, :T] = c
        codes_mask[i, :T] = 1.0

    return {
        "text_ids": text_padded,
        "text_mask": text_mask,
        "codes": codes_padded,
        "codes_mask": codes_mask,
    }


def latent_collate_fn(batch: list[dict]) -> dict:
    """Collate for LatentDataset: pad text_ids and latents."""
    text_ids = [item["text_ids"] for item in batch]
    latents = [item["latents"] for item in batch]  # each (2T_i, 512)

    max_text = max(t.shape[0] for t in text_ids)
    max_lat = max(l.shape[0] for l in latents)
    B = len(batch)

    text_padded = torch.zeros(B, max_text, dtype=torch.long)
    text_mask = torch.ones(B, max_text, dtype=torch.bool)
    for i, t in enumerate(text_ids):
        text_padded[i, : t.shape[0]] = t
        text_mask[i, : t.shape[0]] = False

    D = latents[0].shape[1]  # 512
    lat_padded = torch.zeros(B, max_lat, D)
    lat_mask = torch.zeros(B, max_lat)
    for i, l in enumerate(latents):
        T = l.shape[0]
        lat_padded[i, :T, :] = l
        lat_mask[i, :T] = 1.0

    return {
        "text_ids": text_padded,
        "text_mask": text_mask,
        "latents": lat_padded,
        "latent_mask": lat_mask,
    }
