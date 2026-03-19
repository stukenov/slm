"""Evaluate OmniAudio model: WER/CER on Common Voice kk test split."""

import argparse
import logging
from pathlib import Path

import torch
from jiwer import cer, wer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from omniaudio.data import AudioCollator, load_commonvoice_kk
from omniaudio.model import OmniAudioModel
from omniaudio.train import load_config

logger = logging.getLogger(__name__)


def evaluate(config, model_path, max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    model = OmniAudioModel(
        n_mels=config["n_mels"],
        audio_d_model=config["audio_d_model"],
        audio_n_heads=config["audio_n_heads"],
        audio_n_layers=config["audio_n_layers"],
        llm_vocab_size=config["llm_vocab_size"],
        llm_d_model=config["llm_d_model"],
        llm_n_heads=config["llm_n_heads"],
        llm_n_layers=config["llm_n_layers"],
        llm_intermediate_size=config["llm_intermediate_size"],
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    test_ds = load_commonvoice_kk("test", max_samples=max_samples)
    collator = AudioCollator(
        tokenizer_path=config["tokenizer_path"],
        n_mels=config["n_mels"],
        sample_rate=config["sample_rate"],
        max_audio_len=config["max_audio_len"],
        max_text_len=config["max_text_len"],
    )

    all_refs = []
    all_hyps = []

    logger.info("Evaluating on %d samples...", len(test_ds))

    with torch.no_grad():
        for i, sample in enumerate(test_ds):
            # Process single sample through collator
            batch = collator([sample])
            mel = batch["mel"].to(device)

            # Generate
            tokens = model.generate(mel, tokenizer, max_new_tokens=config.get("max_text_len", 256))
            hyp = tokenizer.decode(tokens, skip_special_tokens=True)
            ref = sample["sentence"]

            all_refs.append(ref)
            all_hyps.append(hyp)

            if (i + 1) % 100 == 0:
                logger.info("Processed %d/%d samples", i + 1, len(test_ds))

    # Compute metrics
    results = {
        "wer": wer(all_refs, all_hyps),
        "cer": cer(all_refs, all_hyps),
        "num_samples": len(all_refs),
    }

    print(f"\n{'='*40}")
    print(f"OmniAudio Evaluation Results")
    print(f"{'='*40}")
    print(f"Samples: {results['num_samples']}")
    print(f"WER:     {results['wer']:.2%}")
    print(f"CER:     {results['cer']:.2%}")
    print(f"{'='*40}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="OmniAudio Evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--model-path", required=True, help="Path to model.pt")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config(args.config)
    evaluate(config, args.model_path, args.max_samples)


if __name__ == "__main__":
    main()
