"""OmniAudio v2 — Local inference script for Kazakh ASR.
Usage:
    python transcribe.py audio.wav
    python transcribe.py audio.mp3
    python transcribe.py  # interactive mode
"""
import sys
import os

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

# Add omniaudio to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from omniaudio.model_v2 import OmniAudioScratchModel

REPO_ID = "stukenov/sozkz-core-omniaudio-70m-kk-asr-v1"
SAMPLE_RATE = 16000
N_MELS = 80


def load_model():
    """Load OmniAudio v2 model from HuggingFace."""
    print("Loading model...")
    model_path = hf_hub_download(REPO_ID, "model.pt")
    tok_path = hf_hub_download(REPO_ID, "tokenizer/tokenizer.json")
    tok_cfg = hf_hub_download(REPO_ID, "tokenizer/tokenizer_config.json")

    model = OmniAudioScratchModel(
        encoder_config={"n_mels": N_MELS, "d_model": 256, "n_heads": 4, "n_layers": 6, "n_conv": 2},
        decoder_config={"d_model": 512, "n_heads": 8, "n_layers": 8},
        vocab_size=50257,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        tokenizer_config=tok_cfg,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def transcribe(model, tokenizer, device, audio_path):
    """Transcribe a single audio file."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=400, hop_length=160,
    )
    mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-10))
    mel = mel.to(device)

    with torch.no_grad():
        tokens = model.generate(mel, max_new_tokens=256)

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def main():
    model, tokenizer, device = load_model()

    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            text = transcribe(model, tokenizer, device, path)
            print(f"\n{path}:")
            print(f"  {text}")
    else:
        print("\nInteractive mode. Enter audio file path (or 'q' to quit):")
        while True:
            try:
                path = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if path.lower() in ("q", "quit", "exit"):
                break
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            text = transcribe(model, tokenizer, device, path)
            print(f"  {text}")


if __name__ == "__main__":
    main()
