"""Test inference from a training checkpoint. Generates WAV files."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import argparse
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from kzcalm.model.backbone import TTSBackbone
from kzcalm.model.flow_head import sample_euler
from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer


TEST_TEXTS = [
    "Сәлеметсіз бе, менің атым Айгүл.",
    "Бүгін ауа райы өте жақсы.",
    "Қазақстан — Орталық Азиядағы ең үлкен мемлекет.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="/workspace/inference_test")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    step = ckpt["step"]
    print(f"Step: {step}")

    model_cfg = config["model"]
    codec_cfg = config["codec"]
    codec_type = codec_cfg.get("type", "mimi")
    latent_dim = codec_cfg["latent_dim"]

    # Tokenizer
    tok_path = hf_hub_download("stukenov/kzcalm-sp-tokenizer-4k-kk-v1", "tokenizer.model")
    tokenizer = KazakhTokenizer(tok_path)

    # Model
    max_latent_frames = model_cfg.get("max_audio_frames", 3000)
    model = TTSBackbone(
        vocab_size=tokenizer.vocab_size,
        latent_dim=latent_dim,
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=0.0,
        max_text_len=model_cfg.get("max_text_len", 512),
        max_audio_frames=max_latent_frames,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load decoder/vocoder
    if codec_type == "mel":
        print("Loading Vocos vocoder...")
        from vocos import Vocos
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        vocoder.eval()
    else:
        print("Loading Mimi decoder...")
        from kzcalm.model.code_embedding import MimiLatentExtractor
        vocoder = MimiLatentExtractor(device=device)

    # Generate
    for i, text in enumerate(TEST_TEXTS):
        text_ids = tokenizer.encode(text)[:256]
        text_tensor = torch.tensor([text_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            latents = sample_euler(
                model, text_tensor, args.num_frames,
                latent_dim=latent_dim, num_steps=args.num_steps, device=device,
            )

            if codec_type == "mel":
                # Denormalize and decode via Vocos
                mel = latents * 3.0 + (-6.0)
                mel_t = mel.transpose(1, 2)  # (B, 80, T)
                waveform = vocoder.decode(mel_t)
            else:
                waveform = vocoder.decode_latents(latents)

        wav = waveform.squeeze().cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        path = os.path.join(args.output_dir, f"step{step}_sample{i}.wav")
        torchaudio.save(path, wav, 24000)
        dur = wav.shape[-1] / 24000
        print(f"  [{i}] \"{text[:40]}\" -> {dur:.1f}s -> {path}")

    print("Done!")


if __name__ == "__main__":
    main()
