"""
Large-scale evaluation of GEC pipeline.
Tests each tag independently + full pipeline on real data.
"""
import json
import os
import time
import random
import sys
from collections import defaultdict
from huggingface_hub import hf_hub_download


def load_data(max_per_type=100, max_clean=100):
    """Load balanced set from GEC dataset."""
    REPO = "stukenov/sozkz-corpus-synthetic-kk-gec-v1"
    FILES = [
        "data/grammar_balanced_v2/train.jsonl",
        "data/grammar_v2/train.jsonl",
        "data/processed_v2/train.jsonl",
    ]
    by_type = defaultdict(list)
    total = 0
    for fname in FILES:
        try:
            local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    total += 1
                    et = ex.get("error_type", "unknown")
                    inp = ex.get("input", "")
                    tgt = ex.get("target", "")
                    if not inp or not tgt:
                        continue
                    cat = "clean" if et == "clean" else et
                    limit = max_clean if cat == "clean" else max_per_type
                    if len(by_type[cat]) < limit:
                        by_type[cat].append({"input": inp, "target": tgt, "type": et})
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
    print(f"Loaded {sum(len(v) for v in by_type.values())} examples from {total} total")
    for et, exs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {et}: {len(exs)}")
    return dict(by_type)


def run_tag_eval(tag, examples, correct_fn):
    correct = wrong = no_change = corrupted = 0
    for ex in examples:
        inp, tgt = ex["input"], ex["target"]
        out = correct_fn(tag, inp)
        if out == tgt:
            correct += 1
        elif out == inp:
            no_change += 1
        else:
            wrong += 1
            cyr = sum(1 for c in out if '\u0400' <= c <= '\u04FF')
            if len(out) > 3 and cyr / max(1, len(out.replace(' ', ''))) < 0.5:
                corrupted += 1
    total = len(examples)
    return {"correct": correct, "wrong": wrong, "no_change": no_change, "corrupted": corrupted, "total": total}


def run_clean_eval(tag, clean_examples, correct_fn):
    preserved = fp = 0
    for ex in clean_examples:
        out = correct_fn(tag, ex["input"])
        if out == ex["input"]:
            preserved += 1
        else:
            fp += 1
    return {"preserved": preserved, "false_positive": fp, "total": len(clean_examples)}


if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast

    MODEL_ID = "stukenov/sozkz-core-llama-300m-kk-gec-v1"
    TOKEN = os.environ.get("HF_TOKEN", "")

    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
    except (ValueError, ImportError):
        tok_file = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.json", token=TOKEN)
        tokenizer = GPT2TokenizerFast(tokenizer_file=tok_file)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, token=TOKEN)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.requires_grad_(False)
    print(f"Model on {device}")

    def make_correct_fn(temperature=0.3, top_p=0.9, rep_pen=1.1):
        def correct(tag, text):
            prompt = f"<{tag}> {text}\n\u2192 "
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=len(inputs["input_ids"][0]) + 30,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    do_sample=temperature > 0.05,
                    repetition_penalty=rep_pen,
                    pad_token_id=tokenizer.eos_token_id,
                )
            result = tokenizer.decode(out[0], skip_special_tokens=True)
            if "\u2192 " in result:
                result = result.split("\u2192 ", 1)[1]
            for stop in ["\n<", "\n\n"]:
                if stop in result:
                    result = result[:result.index(stop)]
            result = result.strip()
            while result and not result[0].isalpha() and result[0] not in "({[\"'":
                result = result[1:]
            return result.strip() or text
        return correct

    correct_fn = make_correct_fn()

    print("\n=== Loading data ===")
    data = load_data(max_per_type=100, max_clean=100)

    TAG_MAP = {
        "3_vowel_harmony": "сингармонизм",
        "1_septik": "септік",
        "5_possessive": "тәуелдік",
        "4_personal_ending": "жіктік",
        "10_postposition": "шылау",
        "6_plural": "көптік",
        "8_negation": "болымсыз",
        "7_tense": "шақ",
        "grammar": "жалғау",
        "noise": "қате",
    }

    GENERAL_TAGS = ["грамматика", "жалғау", "қате"]

    # 1. Per-tag accuracy on matching error types
    print("\n" + "=" * 70)
    print("PER-TAG ACCURACY")
    print("=" * 70)

    tag_results = {}
    for error_type, examples in data.items():
        if error_type == "clean" or error_type not in TAG_MAP:
            continue
        tag = TAG_MAP[error_type]
        t0 = time.time()
        r = run_tag_eval(tag, examples, correct_fn)
        dt = time.time() - t0
        acc = r["correct"] / r["total"] * 100
        wr = r["wrong"] / r["total"] * 100
        nc = r["no_change"] / r["total"] * 100
        print(f"<{tag}> ({error_type}): correct={acc:.1f}% wrong={wr:.1f}% no_change={nc:.1f}% corrupted={r['corrupted']} [{dt:.1f}s]")
        tag_results[tag] = r

    # Also test грамматика on each error type
    print("\n<грамматика> as catch-all:")
    for error_type, examples in data.items():
        if error_type in ("clean", "unknown") or error_type not in TAG_MAP:
            continue
        r = run_tag_eval("грамматика", examples[:50], correct_fn)
        acc = r["correct"] / r["total"] * 100
        wr = r["wrong"] / r["total"] * 100
        print(f"  on {error_type}: correct={acc:.1f}% wrong={wr:.1f}%")

    # 2. False positive on clean
    print("\n" + "=" * 70)
    print("FALSE POSITIVE RATE (clean texts)")
    print("=" * 70)
    clean = data.get("clean", [])[:50]
    if clean:
        for tag in ["грамматика", "сингармонизм", "септік", "шылау", "көптік", "тәуелдік", "жіктік", "болымсыз"]:
            r = run_clean_eval(tag, clean, correct_fn)
            fp = r["false_positive"] / r["total"] * 100
            print(f"  <{tag}>: preserved={r['preserved']}/{r['total']} FP={fp:.1f}%")

    # 3. Temperature sweep
    print("\n" + "=" * 70)
    print("TEMPERATURE SWEEP (<грамматика> on grammar examples)")
    print("=" * 70)
    grammar_ex = data.get("grammar", [])[:50]
    if grammar_ex:
        for temp in [0.1, 0.2, 0.3, 0.5, 0.7]:
            fn = make_correct_fn(temperature=temp)
            r = run_tag_eval("грамматика", grammar_ex, fn)
            acc = r["correct"] / r["total"] * 100
            wr = r["wrong"] / r["total"] * 100
            print(f"  temp={temp}: correct={acc:.1f}% wrong={wr:.1f}% no_change={r['no_change']}")

    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Tag':<20} {'Correct':>8} {'Wrong':>8} {'NoChange':>8} {'Corrupt':>8}")
    for tag, r in sorted(tag_results.items(), key=lambda x: -x[1]["correct"]):
        print(f"<{tag}>{'':<{18-len(tag)}} {r['correct']:>7}/{r['total']:<3} {r['wrong']:>7} {r['no_change']:>8} {r['corrupted']:>8}")

    with open("/tmp/gec_eval.json", "w") as f:
        json.dump({"tags": {k: v for k, v in tag_results.items()}, "data_sizes": {k: len(v) for k, v in data.items()}}, f, indent=2, ensure_ascii=False)
    print("\nSaved to /tmp/gec_eval.json")
