"""Prepare CosyVoice training data from stukenov/kzcalm-tts-kk-v1."""
import argparse
import os
import soundfile as sf
import numpy as np
from datasets import load_dataset
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", default="stukenov/kzcalm-tts-kk-v1")
    parser.add_argument("--output_dir", default="/root/slm/CosyVoice/data/kazakh-train")
    parser.add_argument("--wav_dir", default="/root/slm/CosyVoice/data/kazakh-wavs")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max audio duration in seconds")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Min audio duration in seconds")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.wav_dir, exist_ok=True)

    ds = load_dataset(args.hf_dataset, split="train", streaming=True)

    wav_scp = open(os.path.join(args.output_dir, "wav.scp"), "w")
    text_f = open(os.path.join(args.output_dir, "text"), "w")
    utt2spk = open(os.path.join(args.output_dir, "utt2spk"), "w")

    spk2utt = defaultdict(list)
    total = 0
    skipped = 0

    for sample in ds:
        text = sample.get("text") or sample.get("sentence", "")
        if not text or len(text.strip()) < 2:
            skipped += 1
            continue

        audio = sample["audio"]
        waveform = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        duration = len(waveform) / sr

        if duration > args.max_duration or duration < args.min_duration:
            skipped += 1
            continue

        # Speaker ID — use "speaker" field if available, else "spk_default"
        spk = str(sample.get("speaker", sample.get("speaker_id", "spk_default")))

        utt_id = f"kk_{total:07d}"
        wav_path = os.path.join(args.wav_dir, f"{utt_id}.wav")

        # Save wav at original sample rate
        sf.write(wav_path, waveform, sr)

        # Write Kaldi-style files
        wav_scp.write(f"{utt_id} {wav_path}\n")
        text_f.write(f"{utt_id} <|kk|>{text.strip()}\n")
        utt2spk.write(f"{utt_id} {spk}\n")
        spk2utt[spk].append(utt_id)

        total += 1
        if total % 5000 == 0:
            print(f"Processed {total} samples, {skipped} skipped")

        if args.max_samples > 0 and total >= args.max_samples:
            break

    wav_scp.close()
    text_f.close()
    utt2spk.close()

    # Write spk2utt
    with open(os.path.join(args.output_dir, "spk2utt"), "w") as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")

    print(f"Done! {total} samples, {skipped} skipped, {len(spk2utt)} speakers")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
