"""Inference: text -> flow-matching ODE -> mel/latents -> vocoder/Mimi -> waveform."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torchaudio
import yaml

from kzcalm.model.backbone import TTSBackbone
from kzcalm.model.flow_head import sample_euler
from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer
from kzcalm.train import load_config, _download_sp_tokenizer

logger = logging.getLogger(__name__)


def load_vocoder(codec_type: str = "mel", device: str = "cuda"):
    """Load the appropriate vocoder/decoder."""
    if codec_type == "mel":
        from vocos import Vocos
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        vocos.eval()
        return vocos
    else:
        from transformers import MimiModel
        mimi = MimiModel.from_pretrained("kyutai/mimi").to(device)
        for p in mimi.parameters():
            p.requires_grad = False
        return mimi


def decode_to_waveform(
    vocoder,
    output: torch.Tensor,
    codec_type: str = "mel",
    mel_extractor=None,
) -> torch.Tensor:
    """Decode model output to waveform.

    Args:
        vocoder: Vocos or MimiModel
        output: (B, T, D) mel or latents from flow matching
        codec_type: "mel" or "mimi"
        mel_extractor: MelExtractor (needed to denormalize mel)

    Returns:
        waveform: (B, 1, num_samples) at 24kHz
    """
    with torch.no_grad():
        if codec_type == "mel":
            # Denormalize mel if we have the extractor stats
            if mel_extractor is not None:
                output = output * mel_extractor.mel_std + mel_extractor.mel_mean
            else:
                output = output * 3.0 + (-6.0)

            # Vocos expects (B, n_mels, T)
            mel = output.transpose(1, 2)
            waveform = vocoder.decode(mel)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)
            return waveform
        else:
            latents_t = output.transpose(1, 2)  # (B, D, T)
            return vocoder.decode(latents_t, audio_codes=None)


def synthesize(
    text: str,
    model: TTSBackbone,
    tokenizer: KazakhTokenizer,
    vocoder,
    codec_type: str = "mel",
    mel_extractor=None,
    num_steps: int = 8,
    duration_frames: int | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate waveform from text."""
    model.eval()

    text_ids = tokenizer.encode(text)
    text_tokens = torch.tensor([text_ids], dtype=torch.long, device=device)

    if duration_frames is None:
        duration_frames = len(text_ids) * 10

    latent_dim = model.latent_dim

    latents = sample_euler(
        model=model,
        text_tokens=text_tokens,
        num_frames=duration_frames,
        latent_dim=latent_dim,
        num_steps=num_steps,
        device=device,
    )

    waveform = decode_to_waveform(vocoder, latents, codec_type, mel_extractor)
    return waveform.squeeze(0)  # (1, num_samples)


def main():
    parser = argparse.ArgumentParser(description="KZ-CALM TTS Inference")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint dir")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output wav path")
    parser.add_argument("--steps", type=int, default=8, help="ODE sampling steps")
    parser.add_argument("--frames", type=int, default=None, help="Duration in frames")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.model_path)

    # Load config from checkpoint
    config_path = model_path / "config.yaml"
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logger.warning("No config.yaml found, using defaults")
        config = {
            "codec": {"type": "mel", "latent_dim": 80},
            "tokenizer": {"vocab_size": 4096, "hf_repo": "stukenov/kzcalm-sp-tokenizer-4k-kk-v1"},
            "model": {},
        }

    codec_cfg = config["codec"]
    codec_type = codec_cfg.get("type", "mimi")

    # Load tokenizer
    tok_cfg = config["tokenizer"]
    if tok_cfg.get("model_path"):
        tok_path = tok_cfg["model_path"]
    else:
        tok_path = _download_sp_tokenizer(tok_cfg["hf_repo"])
    tokenizer = KazakhTokenizer(tok_path)

    # Load model
    model_cfg = config.get("model", {})
    model = TTSBackbone(
        vocab_size=tokenizer.vocab_size,
        latent_dim=codec_cfg["latent_dim"],
        d_model=model_cfg.get("d_model", 512),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 10),
        d_ff=model_cfg.get("d_ff", 2048),
    ).to(device)

    # Load weights
    ckpt = model_path / "model_final.pt"
    if not ckpt.exists():
        ckpts = sorted(model_path.glob("checkpoint-*/model.pt"))
        ckpt = ckpts[-1] if ckpts else None

    if ckpt:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state)
        logger.info(f"Loaded: {ckpt}")

    # Load vocoder
    vocoder = load_vocoder(codec_type, device=device)

    # Mel extractor for denormalization
    mel_extractor = None
    if codec_type == "mel":
        from kzcalm.codec.mel import MelExtractor
        mel_extractor = MelExtractor(
            n_mels=codec_cfg.get("n_mels", 80),
            hop_length=codec_cfg.get("hop_length", 256),
            n_fft=codec_cfg.get("n_fft", 1024),
        ).to(device)

    waveform = synthesize(
        text=args.text,
        model=model,
        tokenizer=tokenizer,
        vocoder=vocoder,
        codec_type=codec_type,
        mel_extractor=mel_extractor,
        num_steps=args.steps,
        duration_frames=args.frames,
        device=device,
    )

    torchaudio.save(args.output, waveform.cpu(), 24000)
    logger.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
