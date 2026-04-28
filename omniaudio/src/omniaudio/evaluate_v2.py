"""WER/CER measurement for OmniAudio v2 on Common Voice kk test."""

import argparse
import logging
import random

import numpy as np
import soundfile as sf
import torch
import torchaudio
from jiwer import cer, wer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from omniaudio.data_v2 import AudioCollatorV2, load_speech_dataset, normalize_asr_text
from omniaudio.model_v2 import OmniAudioV2Model, OmniAudioScratchModel
from omniaudio.train_v2 import load_config

logger = logging.getLogger(__name__)


def _dedupe_chunk_texts(texts: list[str]) -> str:
    merged = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if not merged or merged[-1] != text:
            merged.append(text)
    return " ".join(merged).strip()


def _decode_sample(
    sample,
    *,
    model,
    tokenizer,
    collator,
    device,
    config,
) -> str:
    batch = collator([sample])
    mel = batch["mel"].to(device)
    tokens = model.generate(
        mel,
        max_new_tokens=config.get("max_text_len", 256),
        eos_token_id=tokenizer.eos_token_id or 0,
        repetition_penalty=float(config.get("decode_repetition_penalty", 1.15)),
        no_repeat_ngram_size=int(config.get("decode_no_repeat_ngram_size", 3)),
    )
    return normalize_asr_text(
        tokenizer.decode(tokens, skip_special_tokens=True),
        lowercase=config.get("text_lowercase", False),
        strip_punctuation=config.get("text_strip_punctuation", False),
        collapse_whitespace=config.get("text_collapse_whitespace", True),
    )


def _decode_chunked_sample(
    sample,
    *,
    model,
    tokenizer,
    collator,
    device,
    config,
) -> str:
    audio = sample["audio"]
    if isinstance(audio, dict) and "array" in audio:
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]
    else:
        audio_path = audio["path"] if isinstance(audio, dict) else audio
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            elif waveform.ndim == 2:
                waveform = waveform.squeeze(0)
        except ImportError:
            waveform_np, sr = sf.read(audio_path, dtype="float32")
            if waveform_np.ndim == 2:
                waveform_np = waveform_np.mean(axis=1)
            waveform = torch.from_numpy(np.asarray(waveform_np, dtype=np.float32))

    target_sr = config["sample_rate"]
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    chunk_sec = float(config.get("chunk_audio_len", config["max_audio_len"]))
    overlap_sec = float(config.get("chunk_overlap_sec", 3.0))
    chunk_samples = int(chunk_sec * target_sr)
    stride_samples = max(1, int((chunk_sec - overlap_sec) * target_sr))

    if waveform.numel() <= chunk_samples:
        chunk_sample = dict(sample)
        chunk_sample["audio"] = {"array": waveform.cpu().numpy(), "sampling_rate": target_sr}
        return _decode_sample(
            chunk_sample, model=model, tokenizer=tokenizer, collator=collator, device=device, config=config
        )

    chunk_texts = []
    for start in range(0, waveform.numel(), stride_samples):
        chunk = waveform[start:start + chunk_samples]
        if chunk.numel() == 0:
            break
        if chunk.numel() < int(1.0 * target_sr):
            break
        chunk_sample = dict(sample)
        chunk_sample["audio"] = {"array": chunk.cpu().numpy(), "sampling_rate": target_sr}
        chunk_text = _decode_sample(
            chunk_sample, model=model, tokenizer=tokenizer, collator=collator, device=device, config=config
        )
        chunk_texts.append(chunk_text)
        if start + chunk_samples >= waveform.numel():
            break

    return _dedupe_chunk_texts(chunk_texts)


def run_assessment(config, model_path, max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_config = {
        "n_mels": config["n_mels"], "d_model": config["audio_d_model"],
        "n_heads": config["audio_n_heads"], "n_layers": config["audio_n_layers"],
        "n_conv": config["audio_n_conv"],
    }

    model_type = config.get("model_type", "pretrained")
    if model_type == "scratch":
        decoder_config = {
            "d_model": config["decoder_d_model"],
            "n_heads": config["decoder_n_heads"],
            "n_layers": config["decoder_n_layers"],
        }
        model = OmniAudioScratchModel(
            encoder_config=encoder_config, decoder_config=decoder_config,
            vocab_size=config["vocab_size"],
        )
    else:
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
    dataset_name = config.get("dataset_name", "fleurs")
    test_ds = load_speech_dataset(dataset_name, "test", max_samples=max_samples)
    collator = AudioCollatorV2(
        tokenizer_path=config["tokenizer_path"], n_mels=config["n_mels"],
        sample_rate=config["sample_rate"], max_audio_len=config["max_audio_len"],
        max_text_len=config["max_text_len"], augment=False,
        text_lowercase=config.get("text_lowercase", False),
        text_strip_punctuation=config.get("text_strip_punctuation", False),
        text_collapse_whitespace=config.get("text_collapse_whitespace", True),
    )

    all_refs, all_hyps = [], []
    logger.info("Running on %d samples...", len(test_ds))
    chunked = bool(config.get("chunked_inference", False))

    with torch.no_grad():
        for i, sample in enumerate(test_ds):
            if chunked:
                hyp = _decode_chunked_sample(
                    sample, model=model, tokenizer=tokenizer, collator=collator, device=device, config=config
                )
            else:
                hyp = _decode_sample(
                    sample, model=model, tokenizer=tokenizer, collator=collator, device=device, config=config
                )
            ref = normalize_asr_text(
                sample["sentence"],
                lowercase=config.get("text_lowercase", False),
                strip_punctuation=config.get("text_strip_punctuation", False),
                collapse_whitespace=config.get("text_collapse_whitespace", True),
            )
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
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset (e.g. kzcalm)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)
    if args.dataset:
        config["dataset_name"] = args.dataset
    run_assessment(config, args.model_path, args.max_samples)


if __name__ == "__main__":
    main()
