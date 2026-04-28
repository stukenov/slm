"""
exp034: Evaluate translation models on FLORES+ devtest.

Metrics: BLEU (sacrebleu), chrF (sacrebleu)
Benchmark: FLORES+ devtest (1012 sentences, kk↔ru)

Usage:
  python3 evaluate_flores.py --model stukenov/ekitil-core-qwen3-123m-kkru-translate-v1
  python3 evaluate_flores.py --model stukenov/ekitil-core-qwen3-123m-kkru-translate-v1 --direction ru-kk
"""

import os
import json
import argparse
import torch
from tqdm import tqdm


KK_TAG = "<|kk|>"
RU_TAG = "<|ru|>"
TRANSLATE_TAG = "<|translate|>"


def load_flores_devtest():
    """Load FLORES+ devtest for kk and ru."""
    from datasets import load_dataset

    print("Loading FLORES+ devtest...")
    ds = load_dataset("openlanguagedata/flores_plus", split="devtest")

    pairs = []
    for row in ds:
        kk = row.get("kk", row.get("kaz_Cyrl", "")).strip()
        ru = row.get("ru", row.get("rus_Cyrl", "")).strip()
        if kk and ru:
            pairs.append({"kk": kk, "ru": ru})

    print(f"  FLORES+ devtest: {len(pairs)} sentence pairs")
    return pairs


def translate_batch(model, tokenizer, texts, src_lang, tgt_lang, device, max_new_tokens=256):
    """Translate a batch of texts."""
    src_tag = KK_TAG if src_lang == "kk" else RU_TAG
    tgt_tag = RU_TAG if src_lang == "kk" else KK_TAG

    translations = []
    for text in texts:
        prompt = f"{src_tag} {text} {TRANSLATE_TAG} {tgt_tag}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for consistent eval
                num_beams=4,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = output[0][input_ids.shape[1]:]
        translation = tokenizer.decode(generated, skip_special_tokens=True).strip()
        translations.append(translation)

    return translations


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # Load FLORES+
    pairs = load_flores_devtest()
    if args.max_samples:
        pairs = pairs[:args.max_samples]

    # Determine direction
    if args.direction == "kk-ru":
        src_lang, tgt_lang = "kk", "ru"
        sources = [p["kk"] for p in pairs]
        references = [p["ru"] for p in pairs]
    else:
        src_lang, tgt_lang = "ru", "kk"
        sources = [p["ru"] for p in pairs]
        references = [p["kk"] for p in pairs]

    print(f"\nTranslating {len(sources)} sentences ({src_lang} -> {tgt_lang})...")

    # Translate
    hypotheses = []
    for i in tqdm(range(0, len(sources), args.batch_size)):
        batch = sources[i:i + args.batch_size]
        translations = translate_batch(model, tokenizer, batch, src_lang, tgt_lang, device)
        hypotheses.extend(translations)

    # Compute metrics
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Direction: {src_lang} -> {tgt_lang}")
    print(f"Samples: {len(hypotheses)}")
    print(f"{'='*60}")
    print(f"BLEU:  {bleu.score:.2f}")
    print(f"chrF:  {chrf.score:.2f}")
    print(f"{'='*60}")

    # Show examples
    print(f"\nExamples:")
    for i in range(min(5, len(sources))):
        print(f"\n  SRC: {sources[i][:100]}")
        print(f"  REF: {references[i][:100]}")
        print(f"  HYP: {hypotheses[i][:100]}")

    # Save results
    results = {
        "model": args.model,
        "direction": f"{src_lang}-{tgt_lang}",
        "n_samples": len(hypotheses),
        "bleu": bleu.score,
        "chrf": chrf.score,
        "bleu_str": str(bleu),
        "chrf_str": str(chrf),
    }

    output_file = args.output or f"flores_results_{src_lang}_{tgt_lang}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model repo or path")
    parser.add_argument("--direction", default="kk-ru", choices=["kk-ru", "ru-kk"])
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for translation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()
