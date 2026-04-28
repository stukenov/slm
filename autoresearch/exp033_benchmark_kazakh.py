#!/usr/bin/env python3
"""exp033: Benchmark small models on Kazakh. Measures PPL, fertility, generation."""

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path

import torch

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B"),
    ("LLaMA-3.2-1B", "meta-llama/Llama-3.2-1B"),
    ("LLaMA-3.2-3B", "meta-llama/Llama-3.2-3B"),
    ("Gemma-2-2B", "google/gemma-2-2b"),
    ("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M"),
    ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B"),
    ("mGPT-1.3B", "ai-forever/mGPT"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    ("SozKZ-TinyLlama-kk", "stukenov/sozkz-core-tinyllama-1b-kk-ru-v1"),
    ("StableLM-2-1.6B", "stabilityai/stablelm-2-1_6b"),
]

KAZ_TEXTS = [
    "Қазақстан Республикасы — Орталық Азиядағы мемлекет. Астанасы — Астана қаласы. Тұрғындар саны — 19 миллионнан астам.",
    "Қазақ тілі — түркі тілдерінің қыпшақ тобына жататын тіл. Қазақстанның мемлекеттік тілі болып табылады.",
    "Білім беру жүйесі — мемлекеттік саясаттың маңызды бағыттарының бірі. Қазақстанда жоғары білім беретін 130-дан астам университет бар.",
    "Алматы — Қазақстанның ең ірі қаласы және мәдени астанасы. Халық саны — 2 миллионнан астам адам.",
    "Қазақстан экономикасы мұнай мен газ өндірісіне негізделген. Ел әлемдегі ең ірі мұнай өндіруші елдердің бірі.",
    "Абай Құнанбайұлы — ұлы қазақ ақыны, ағартушы, ойшыл. Қазақ жазба әдебиетінің негізін қалаушы.",
    "Наурыз мейрамы — қазақ халқының көне мерекесі. Жыл сайын наурыздың 22-сінде тойланады.",
    "Қазақтың ұлттық тағамы — бешбармақ. Ол қой етінен және қамырдан жасалады.",
    "Байқоңыр ғарыш айлағы — әлемдегі ең ірі ғарыш айлағы. Ол Қызылорда облысында орналасқан.",
    "Қазақстан Республикасының Конституциясы 1995 жылы 30 тамызда қабылданды.",
]

RUS_TEXTS = [
    "Республика Казахстан — государство в Центральной Азии. Столица — город Астана.",
    "Казахский язык относится к кыпчакской группе тюркских языков.",
    "Система образования является одним из важнейших направлений государственной политики.",
    "Алматы — крупнейший город Казахстана и культурная столица страны.",
    "Экономика Казахстана основана на добыче нефти и газа.",
]

ENG_TEXTS = [
    "The Republic of Kazakhstan is a country in Central Asia. Its capital is Astana.",
    "The Kazakh language belongs to the Kipchak group of Turkic languages.",
    "The education system is one of the most important areas of state policy.",
    "Almaty is the largest city in Kazakhstan and the cultural capital of the country.",
]

GEN_PROMPTS = [
    ("kaz_1", "Қазақстан — бұл"),
    ("kaz_2", "Қазақ тілінің тарихы"),
    ("kaz_3", "Бүгін ауа райы"),
    ("rus_1", "Казахстан — это"),
    ("eng_1", "Kazakhstan is"),
]

KAZ_CHARS = list("әғқңөүұіһӘҒҚҢӨҮҰІҺ")


def compute_ppl(model, tokenizer, texts, max_length=512):
    total_loss, total_toks = 0, 0
    for text in texts:
        try:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            total_loss += out.loss.item() * enc["input_ids"].shape[1]
            total_toks += enc["input_ids"].shape[1]
        except Exception as e:
            log.warning("  PPL error: %s", str(e)[:80])
    return math.exp(total_loss / total_toks) if total_toks > 0 else float("inf")


def fertility(tokenizer, text):
    toks = tokenizer.encode(text, add_special_tokens=False)
    words = text.split()
    return len(toks) / len(words) if words else 0


def kaz_char_coverage(tokenizer):
    vocab = tokenizer.get_vocab()
    found = 0
    for ch in KAZ_CHARS:
        for prefix in ["", " ", "\u2581", "\u0120"]:
            if prefix + ch in vocab:
                found += 1
                break
    return found, len(KAZ_CHARS)


def gen(model, tokenizer, prompt, max_new=60):
    try:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.2,
                                 pad_token_id=pad)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"ERROR: {e}"


def bench(name, repo):
    log.info("=" * 70)
    log.info("MODEL: %s (%s)", name, repo)
    log.info("=" * 70)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    r = {"name": name, "repo": repo, "error": None}
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        r["vocab_size"] = len(tokenizer)

        cov, tot = kaz_char_coverage(tokenizer)
        r["kaz_chars"] = f"{cov}/{tot}"

        r["fert_kaz"] = round(fertility(tokenizer, "Қазақстан Республикасы — Орталық Азиядағы мемлекет"), 2)
        r["fert_rus"] = round(fertility(tokenizer, "Республика Казахстан — государство в Центральной Азии"), 2)
        r["fert_eng"] = round(fertility(tokenizer, "Republic of Kazakhstan — a state in Central Asia"), 2)
        log.info("  Vocab: %d, KazChars: %s, Fert: kaz=%.2f rus=%.2f eng=%.2f",
                 r["vocab_size"], r["kaz_chars"], r["fert_kaz"], r["fert_rus"], r["fert_eng"])

        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16,
                                                      trust_remote_code=True).to(DEVICE)
        model.eval()
        r["params_B"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
        log.info("  Params: %.3fB", r["params_B"])

        log.info("  PPL...")
        r["ppl_kaz"] = round(compute_ppl(model, tokenizer, KAZ_TEXTS), 2)
        r["ppl_rus"] = round(compute_ppl(model, tokenizer, RUS_TEXTS), 2)
        r["ppl_eng"] = round(compute_ppl(model, tokenizer, ENG_TEXTS), 2)
        log.info("  PPL: kaz=%.1f  rus=%.1f  eng=%.1f", r["ppl_kaz"], r["ppl_rus"], r["ppl_eng"])

        log.info("  Generating...")
        r["gen"] = {}
        for pname, prompt in GEN_PROMPTS:
            text = gen(model, tokenizer, prompt)
            r["gen"][pname] = text
            log.info("    [%s] %s", pname, text[:120])

        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        log.error("  FAILED: %s", str(e)[:200])
        r["error"] = str(e)[:300]
        gc.collect()
        torch.cuda.empty_cache()

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/exp033_results.json")
    args = parser.parse_args()

    log.info("exp033: Kazakh Benchmark, %d models, device=%s", len(MODELS), DEVICE)
    results = []

    for name, repo in MODELS:
        r = bench(name, repo)
        results.append(r)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    log.info("\n" + "=" * 110)
    log.info("%-25s %6s %8s %8s %8s %8s %7s %7s",
             "Model", "Params", "Vocab", "PPL_kk", "PPL_ru", "PPL_en", "Frt_kk", "KzChr")
    log.info("-" * 110)
    for r in sorted(results, key=lambda x: x.get("ppl_kaz", 99999)):
        if r.get("error"):
            log.info("%-25s FAILED", r["name"])
            continue
        log.info("%-25s %5.2fB %8d %8.1f %8.1f %8.1f %7.2f %7s",
                 r["name"], r.get("params_B", 0), r.get("vocab_size", 0),
                 r.get("ppl_kaz", 0), r.get("ppl_rus", 0), r.get("ppl_eng", 0),
                 r.get("fert_kaz", 0), r.get("kaz_chars", "?"))
    log.info("=" * 110)
    log.info("Results: %s", args.output)


if __name__ == "__main__":
    main()
