#!/usr/bin/env python3
"""exp033 extra: Benchmark sozkz-600m + compute tokenizer compression ratios."""

import gc
import json
import logging
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KAZ_TEXTS = [
    "Қазақстан Республикасы — Орталық Азиядағы мемлекет. Астанасы — Астана қаласы.",
    "Қазақ тілі — түркі тілдерінің қыпшақ тобына жататын тіл.",
    "Білім беру жүйесі — мемлекеттік саясаттың маңызды бағыттарының бірі.",
    "Алматы — Қазақстанның ең ірі қаласы және мәдени астанасы.",
    "Қазақстан экономикасы мұнай мен газ өндірісіне негізделген.",
    "Абай Құнанбайұлы — ұлы қазақ ақыны, ағартушы, ойшыл.",
    "Наурыз мейрамы — қазақ халқының көне мерекесі.",
    "Қазақтың ұлттық тағамы — бешбармақ. Ол қой етінен және қамырдан жасалады.",
    "Байқоңыр ғарыш айлағы — әлемдегі ең ірі ғарыш айлағы.",
    "Қазақстан Республикасының Конституциясы 1995 жылы 30 тамызда қабылданды.",
]
RUS_TEXTS = [
    "Республика Казахстан — государство в Центральной Азии.",
    "Казахский язык относится к кыпчакской группе тюркских языков.",
    "Система образования является одним из важнейших направлений государственной политики.",
    "Алматы — крупнейший город Казахстана и культурная столица страны.",
    "Экономика Казахстана основана на добыче нефти и газа.",
]
ENG_TEXTS = [
    "The Republic of Kazakhstan is a country in Central Asia.",
    "The Kazakh language belongs to the Kipchak group of Turkic languages.",
    "The education system is one of the most important areas of state policy.",
    "Almaty is the largest city in Kazakhstan and the cultural capital.",
]

KAZ_LONG = "Қазақстан Республикасы — Орталық Азиядағы мемлекет. Жер аумағы бойынша әлемде тоғызыншы орында. Халық саны 19 миллионнан астам. Мемлекеттік тілі — қазақ тілі. Ресми тілі — орыс тілі. Астанасы — Астана қаласы. Ең ірі қаласы — Алматы. Қазақстан егемендігін 1991 жылы жариялады. Экономикасы мұнай газ уран және басқа табиғи ресурстарға негізделген. Қазақ халқы көшпенді мал шаруашылығымен айналысқан."
RUS_LONG = "Республика Казахстан — государство в Центральной Азии. По площади территории занимает девятое место в мире. Население составляет более 19 миллионов человек. Государственный язык — казахский. Официальный язык — русский. Столица — город Астана. Крупнейший город — Алматы. Независимость была провозглашена в 1991 году. Экономика основана на нефти газе уране и других природных ресурсах."
ENG_LONG = "The Republic of Kazakhstan is a country in Central Asia. It is the ninth largest country in the world by area. The population is over 19 million people. The state language is Kazakh. The official language is Russian. The capital is Astana. The largest city is Almaty. Independence was declared in 1991. The economy is based on oil gas uranium and other natural resources."

KAZ_CHARS = list("әғқңөүұіһӘҒҚҢӨҮҰІҺ")

ALL_MODELS = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B"),
    ("LLaMA-3.2-1B", "meta-llama/Llama-3.2-1B"),
    ("LLaMA-3.2-3B", "meta-llama/Llama-3.2-3B"),
    ("Gemma-2-2B", "google/gemma-2-2b"),
    ("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M"),
    ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    ("SozKZ-TinyLlama-kk", "stukenov/sozkz-core-tinyllama-1b-kk-ru-v1"),
    ("StableLM-2-1.6B", "stabilityai/stablelm-2-1_6b"),
    ("SozKZ-Llama-600M", "stukenov/sozkz-core-llama-600m-kk-base-v1"),
]


def compute_ppl(model, tokenizer, texts):
    total_loss, total_toks = 0, 0
    for text in texts:
        try:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            total_loss += out.loss.item() * enc["input_ids"].shape[1]
            total_toks += enc["input_ids"].shape[1]
        except Exception as e:
            log.warning("  err: %s", str(e)[:80])
    return math.exp(total_loss / total_toks) if total_toks > 0 else float("inf")


def compression(tokenizer, text):
    toks = tokenizer.encode(text, add_special_tokens=False)
    chars = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    words = len(text.split())
    return {
        "tokens": len(toks), "chars": chars, "bytes": utf8_bytes, "words": words,
        "tok_per_word": round(len(toks)/words, 2) if words else 0,
        "bytes_per_tok": round(utf8_bytes/len(toks), 2) if toks else 0,
    }


def main():
    log.info("=== exp033 extra ===")

    # Part 1: 600M PPL
    log.info("--- 600M Benchmark ---")
    repo = "stukenov/sozkz-core-llama-600m-kk-base-v1"
    r600 = {"name": "SozKZ-Llama-600M", "repo": repo}
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16,
                                                    trust_remote_code=True).to(DEVICE)
        mdl.eval()
        r600["params_B"] = round(sum(p.numel() for p in mdl.parameters())/1e9, 3)
        r600["vocab_size"] = len(tok)
        r600["ppl_kaz"] = round(compute_ppl(mdl, tok, KAZ_TEXTS), 2)
        r600["ppl_rus"] = round(compute_ppl(mdl, tok, RUS_TEXTS), 2)
        r600["ppl_eng"] = round(compute_ppl(mdl, tok, ENG_TEXTS), 2)
        log.info("  600M: %.3fB params, vocab %d, PPL kaz=%.1f rus=%.1f eng=%.1f",
                 r600["params_B"], r600["vocab_size"], r600["ppl_kaz"], r600["ppl_rus"], r600["ppl_eng"])

        for prompt in ["Қазақстан — бұл", "Казахстан — это"]:
            ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = mdl.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.7,
                                   repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
            r600[f"gen_{prompt[:10]}"] = tok.decode(out[0], skip_special_tokens=True)[:200]
            log.info("  GEN: %s", r600[f"gen_{prompt[:10]}"][:150])
        del mdl; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log.error("  FAILED: %s", e)
        r600["error"] = str(e)[:200]

    # Part 2: Compression for all
    log.info("\n--- Tokenizer Compression ---")
    comp = []
    for name, repo in ALL_MODELS:
        try:
            t = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            kaz = compression(t, KAZ_LONG)
            rus = compression(t, RUS_LONG)
            eng = compression(t, ENG_LONG)
            cov = sum(1 for ch in KAZ_CHARS if any((p+ch) in t.get_vocab() for p in ["","\u2581","\u0120"," "]))
            ratio = round(kaz["tok_per_word"] / eng["tok_per_word"], 2) if eng["tok_per_word"] else 0
            entry = {"name": name, "vocab": len(t), "kaz_chars": f"{cov}/{len(KAZ_CHARS)}",
                     "kaz_tokens": kaz["tokens"], "eng_tokens": eng["tokens"],
                     "kaz_tok_per_word": kaz["tok_per_word"], "eng_tok_per_word": eng["tok_per_word"],
                     "kaz_bytes_per_tok": kaz["bytes_per_tok"], "eng_bytes_per_tok": eng["bytes_per_tok"],
                     "kaz_vs_eng": ratio}
            comp.append(entry)
            log.info("%-25s vocab=%6d  kaz=%3d tok (%.2f/w, %.1f B/t)  eng=%3d tok (%.2f/w)  ratio=%.2fx  kzch=%s",
                     name, len(t), kaz["tokens"], kaz["tok_per_word"], kaz["bytes_per_tok"],
                     eng["tokens"], eng["tok_per_word"], ratio, f"{cov}/{len(KAZ_CHARS)}")
        except Exception as e:
            log.error("  %s: %s", name, str(e)[:100])

    with open("/root/exp033_extra.json", "w") as f:
        json.dump({"sozkz_600m": r600, "compression": comp}, f, indent=2, ensure_ascii=False)
    log.info("DONE: /root/exp033_extra.json")


if __name__ == "__main__":
    main()
