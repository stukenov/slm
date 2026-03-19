"""Inference for Matcha-TTS: text -> mel (ODE) -> Vocos -> waveform."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import soundfile as sf
import yaml
from pathlib import Path

from kzcalm.model.matcha import MatchaTTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt checkpoint")
    parser.add_argument("--texts", nargs="+", default=[
        "Сәлеметсіз бе",
        "Қазақстан Республикасы",
        "Бүгін ауа райы жақсы",
    ])
    parser.add_argument("--output_dir", default="/root/slm/kz-calm/outputs/infer_matcha")
    parser.add_argument("--steps", type=int, default=8, help="ODE sampling steps")
    parser.add_argument("--length_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model_cfg = config["model"]
    codec_cfg = config["codec"]
    tok_cfg = config["tokenizer"]

    # Tokenizer
    if tok_cfg.get("type") == "char":
        from kzcalm.tokenizer.char_tokenizer import KazakhCharTokenizer
        tokenizer = KazakhCharTokenizer()
    else:
        from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer
        from huggingface_hub import hf_hub_download
        if tok_cfg.get("model_path"):
            tok_path = tok_cfg["model_path"]
        elif tok_cfg.get("hf_repo"):
            tok_path = hf_hub_download(tok_cfg["hf_repo"], "tokenizer.model")
        tokenizer = KazakhTokenizer(tok_path)

    # Model
    model = MatchaTTS(
        vocab_size=tokenizer.vocab_size,
        mel_dim=codec_cfg.get("latent_dim", 100),
        encoder_dim=model_cfg.get("encoder_dim", 256),
        encoder_layers=model_cfg.get("encoder_layers", 4),
        encoder_heads=model_cfg.get("encoder_heads", 4),
        encoder_ff=model_cfg.get("encoder_ff", 1024),
        unet_channels=model_cfg.get("unet_channels", [256, 256, 512]),
        dropout=model_cfg.get("dropout", 0.1),
        max_text_len=model_cfg.get("max_text_len", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint step={ckpt['step']}")

    # Vocoder
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocos.eval()

    # Mel extractor for denorm stats
    from kzcalm.codec.mel import MelExtractor
    mel_ext = MelExtractor(
        n_mels=codec_cfg.get("n_mels", 100),
        hop_length=codec_cfg.get("hop_length", 256),
        n_fft=codec_cfg.get("n_fft", 1024),
        sample_rate=config.get("sample_rate", 24000),
    ).to(device)

    for i, text in enumerate(args.texts):
        text_ids = tokenizer.encode(text)
        text_tokens = torch.tensor([text_ids], dtype=torch.long, device=device)
        text_mask = torch.ones(1, len(text_ids), device=device)

        mel = model.synthesize(
            text_tokens, text_mask,
            num_steps=args.steps,
            length_scale=args.length_scale,
        )  # (1, T, mel_dim)

        # Denormalize
        mel_denorm = mel * mel_ext.mel_std + mel_ext.mel_mean

        # Vocos: (B, n_mels, T)
        mel_t = mel_denorm.transpose(1, 2)
        with torch.no_grad():
            wav = vocos.decode(mel_t)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        out_path = os.path.join(args.output_dir, f"sample_{i:02d}.wav")
        wav_np = wav.squeeze(0).squeeze(0).cpu().numpy()
        sf.write(out_path, wav_np, 24000)
        dur_s = len(wav_np) / 24000
        print(f"[{i}] '{text}' -> {out_path} ({dur_s:.2f}s)")


if __name__ == "__main__":
    main()
