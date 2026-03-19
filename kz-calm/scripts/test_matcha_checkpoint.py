"""Generate WAV files from a Matcha-TTS checkpoint."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import argparse
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from kzcalm.model.matcha import MatchaTTS
from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer

TEST_TEXTS = [
    "Сәлеметсіз бе, менің атым Айгүл.",
    "Бүгін ауа райы өте жақсы.",
    "Қазақстан — Орталық Азиядағы ең үлкен мемлекет.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./inference_test")
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--length_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    step = ckpt["step"]
    model_cfg = config["model"]
    codec_cfg = config["codec"]
    tok_cfg = config["tokenizer"]

    if tok_cfg.get("model_path"):
        tok_path = tok_cfg["model_path"]
    else:
        tok_path = hf_hub_download(tok_cfg["hf_repo"], "tokenizer.model")
    tokenizer = KazakhTokenizer(tok_path)

    model = MatchaTTS(
        vocab_size=tokenizer.vocab_size,
        mel_dim=codec_cfg.get("latent_dim", 100),
        encoder_dim=model_cfg.get("encoder_dim", 256),
        encoder_layers=model_cfg.get("encoder_layers", 4),
        encoder_heads=model_cfg.get("encoder_heads", 4),
        encoder_ff=model_cfg.get("encoder_ff", 1024),
        unet_channels=model_cfg.get("unet_channels", [256, 256, 512]),
        dropout=0.0,
        max_text_len=model_cfg.get("max_text_len", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded step={step}, {n_params:.1f}M params")

    from vocos import Vocos
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocoder.eval()

    mel_mean, mel_std = -1.42, 3.80

    for i, text in enumerate(TEST_TEXTS):
        text_ids = tokenizer.encode(text)[:256]
        text_tensor = torch.tensor([text_ids], dtype=torch.long, device=device)
        text_mask = torch.ones(1, len(text_ids), device=device)

        with torch.no_grad():
            mel = model.synthesize(
                text_tensor, text_mask,
                num_steps=args.num_steps,
                length_scale=args.length_scale,
            )
            mel = mel * mel_std + mel_mean
            wav = vocoder.decode(mel.transpose(1, 2))

        wav_np = wav.squeeze().cpu().numpy()
        path = os.path.join(args.output_dir, f"step{step}_sample{i}.wav")
        sf.write(path, wav_np, 24000)
        dur = len(wav_np) / 24000
        print(f"  [{i}] {text[:40]} -> {dur:.1f}s -> {path}")

    print("Done!")


if __name__ == "__main__":
    main()
