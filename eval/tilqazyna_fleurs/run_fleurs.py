"""FLEURS kk_kz WER/CER benchmark for TilQazyna Whisper models.

Usage:
    HF_TOKEN=hf_xxx python run_fleurs.py --model TilQazyna/tq-small-kaz-1 --out small.json
    HF_TOKEN=hf_xxx python run_fleurs.py --model TilQazyna/tq-large-kaz-1 --out large.json --batch 8
"""
import argparse
import json
import time

import torch
from datasets import load_dataset
from jiwer import cer, wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=225)
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[info] device={device} dtype={dtype} model={args.model}")

    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.train(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] loaded in {time.time()-t0:.1f}s | params={n_params/1e6:.1f}M")

    forced = processor.get_decoder_prompt_ids(language="kk", task="transcribe")

    print("[info] loading google/fleurs kk_kz ...")
    ds = load_dataset("google/fleurs", "kk_kz", split=args.split, trust_remote_code=True)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[info] samples={len(ds)}")

    normalizer = BasicTextNormalizer()
    preds_raw, refs_raw, preds_norm, refs_norm = [], [], [], []

    t0 = time.time()
    for i in range(0, len(ds), args.batch):
        batch = ds[i : i + args.batch]
        audios = [a["array"] for a in batch["audio"]]
        srs = [a["sampling_rate"] for a in batch["audio"]]
        assert all(sr == 16000 for sr in srs), f"non-16k sr: {set(srs)}"

        inputs = processor(
            audios, sampling_rate=16000, return_tensors="pt", padding=True
        )
        input_features = inputs.input_features.to(device, dtype=dtype)

        with torch.no_grad():
            gen = model.generate(
                input_features,
                forced_decoder_ids=forced,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
            )
        texts = processor.batch_decode(gen, skip_special_tokens=True)
        refs = batch["transcription"]

        for p, r in zip(texts, refs):
            preds_raw.append(p)
            refs_raw.append(r)
            pn, rn = normalizer(p), normalizer(r)
            if rn.strip():
                preds_norm.append(pn)
                refs_norm.append(rn)

        done = i + len(texts)
        elapsed = time.time() - t0
        print(f"[{done}/{len(ds)}] {elapsed:.1f}s ({done/max(elapsed,1e-6):.1f} utt/s)")

    total_t = time.time() - t0

    results = {
        "model": args.model,
        "split": args.split,
        "samples": len(ds),
        "params_M": round(n_params / 1e6, 2),
        "device": device,
        "dtype": str(dtype),
        "batch": args.batch,
        "decoding": "greedy",
        "language": "kk",
        "time_s": round(total_t, 2),
        "utts_per_s": round(len(ds) / max(total_t, 1e-6), 2),
        "wer_raw": round(wer(refs_raw, preds_raw) * 100, 3),
        "cer_raw": round(cer(refs_raw, preds_raw) * 100, 3),
        "wer_normalized": round(wer(refs_norm, preds_norm) * 100, 3),
        "cer_normalized": round(cer(refs_norm, preds_norm) * 100, 3),
        "examples": [
            {"ref": refs_raw[i], "hyp": preds_raw[i]} for i in range(min(10, len(refs_raw)))
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
