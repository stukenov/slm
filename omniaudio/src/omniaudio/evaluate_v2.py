"""WER/CER measurement for OmniAudio v2 on Common Voice kk test."""

import argparse
import logging
import random

import torch
from jiwer import cer, wer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from omniaudio.data_v2 import AudioCollatorV2, load_commonvoice_kk
from omniaudio.model_v2 import OmniAudioV2Model
from omniaudio.train_v2 import load_config

logger = logging.getLogger(__name__)


def run_assessment(config, model_path, max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_config = {
        "n_mels": config["n_mels"], "d_model": config["audio_d_model"],
        "n_heads": config["audio_n_heads"], "n_layers": config["audio_n_layers"],
        "n_conv": config["audio_n_conv"],
    }

    model = OmniAudioV2Model(
        encoder_config=encoder_config, llm_name=config["llm_name"],
        vocab_size=config["vocab_size"], llm_dim=config.get("llm_dim", 768),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.train(False)

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    except (ValueError, OSError):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config["tokenizer_path"])
    test_ds = load_commonvoice_kk("test", max_samples=max_samples)
    collator = AudioCollatorV2(
        tokenizer_path=config["tokenizer_path"], n_mels=config["n_mels"],
        sample_rate=config["sample_rate"], max_audio_len=config["max_audio_len"],
        max_text_len=config["max_text_len"], augment=False,
    )

    all_refs, all_hyps = [], []
    logger.info("Running on %d samples...", len(test_ds))

    with torch.no_grad():
        for i, sample in enumerate(test_ds):
            batch = collator([sample])
            mel = batch["mel"].to(device)
            tokens = model.generate(mel, max_new_tokens=config.get("max_text_len", 256),
                                    eos_token_id=tokenizer.eos_token_id or 0)
            hyp = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            ref = sample["sentence"].strip()
            all_refs.append(ref)
            all_hyps.append(hyp)
            if (i + 1) % 100 == 0:
                logger.info("Processed %d/%d", i + 1, len(test_ds))

    results = {"wer": wer(all_refs, all_hyps), "cer": cer(all_refs, all_hyps), "n": len(all_refs)}

    print(f"\n{'='*40}")
    print("OmniAudio v2 Results")
    print(f"{'='*40}")
    print(f"Samples:  {results['n']}")
    print(f"WER:      {results['wer']:.2%}")
    print(f"CER:      {results['cer']:.2%}")
    print(f"{'='*40}\n")

    indices = random.sample(range(len(all_refs)), min(5, len(all_refs)))
    for idx in indices:
        print(f"REF: {all_refs[idx]}")
        print(f"HYP: {all_hyps[idx]}\n")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)
    run_assessment(config, args.model_path, args.max_samples)


if __name__ == "__main__":
    main()
