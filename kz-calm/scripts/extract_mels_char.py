"""Extract mel spectrograms with character tokenization."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import torchaudio
from datasets import load_dataset
from kzcalm.codec.mel import MelExtractor
from kzcalm.tokenizer.char_tokenizer import KazakhCharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/root/slm/kz-calm/data/mels_char")
    parser.add_argument("--hf_dataset", default="stukenov/kzcalm-tts-kk-v1")
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--max_mel_frames", type=int, default=3000)
    parser.add_argument("--min_mel_frames", type=int, default=10)
    parser.add_argument("--max_text_len", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=100)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--sample_rate", type=int, default=24000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = KazakhCharTokenizer()
    print(f"Char tokenizer: vocab_size={tokenizer.vocab_size}")

    mel_extractor = MelExtractor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    ds = load_dataset(args.hf_dataset, split="train", streaming=True)

    shard_idx = 0
    shard_items = []
    total = 0
    skipped = 0

    for sample in ds:
        text = sample.get("text") or sample.get("sentence", "")
        if not text:
            skipped += 1
            continue

        text_ids = tokenizer.encode(text)
        if len(text_ids) > args.max_text_len:
            text_ids = text_ids[:args.max_text_len]

        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]

        if sr != args.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)

        mel = mel_extractor(waveform)

        if mel.shape[0] > args.max_mel_frames or mel.shape[0] < args.min_mel_frames:
            skipped += 1
            continue

        shard_items.append({
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "mel": mel.half(),
        })
        total += 1

        if len(shard_items) >= args.shard_size:
            path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
            torch.save(shard_items, path)
            print(f"Saved {path} ({len(shard_items)} items, total={total})")
            shard_items = []
            shard_idx += 1

    if shard_items:
        path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
        torch.save(shard_items, path)
        print(f"Saved {path} ({len(shard_items)} items, total={total})")

    print(f"Done! {total} samples in {shard_idx + 1} shards, {skipped} skipped")


if __name__ == "__main__":
    main()
