#!/usr/bin/env python3
"""exp033: Benchmark models up to 10B (and MoE active <=10B) on Kazakh."""
import gc, json, logging, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    # New medium + MoE models only (small ones already done)
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
    ("LLaMA-3.1-8B", "meta-llama/Llama-3.1-8B"),
    ("Gemma-2-9B", "google/gemma-2-9b"),
    ("Mistral-7B-v0.3", "mistralai/Mistral-7B-v0.3"),
    ("Phi-3.5-mini-3.8B", "microsoft/Phi-3.5-mini-instruct"),
    ("Yi-1.5-6B", "01-ai/Yi-1.5-6B"),
    ("Yi-1.5-9B", "01-ai/Yi-1.5-9B"),
    ("OLMo-2-7B", "allenai/OLMo-2-1124-7B"),
    ("Falcon-7B", "tiiuae/falcon-7b"),
    ("OLMoE-1B-7B", "allenai/OLMoE-1B-7B-0924"),
    ("DeepSeek-MoE-16B", "deepseek-ai/deepseek-moe-16b-base"),
]

KAZ_TEXTS = [
    "Қазақстан Республикасы — Орталық Азиядағы мемлекет. Астанасы — Астана қаласы. Тұрғындар саны — 19 миллионнан астам.",
    "Қазақ тілі — түркі тілдерінің қыпшақ тобына жататын тіл. Қазақстанның мемлекеттік тілі болып табылады.",
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
    "Almaty is the largest city in Kazakhstan and the cultural capital.",
]
GEN_PROMPTS = [
    "Қазақстан — бұл",
    "Қазақ тілі — түркі тілдерінің",
    "Алматы қаласында бүгін",
    "Білім беру саласында",
    "Абай Құнанбайұлы — ұлы",
    "Наурыз мейрамы — қазақ халқының",
    "Қазақстан экономикасы",
    "Ұлттық тағамдар арасында",
    "Қазақ мәдениеті мен өнері",
    "Жастарға арналған",
]
KAZ_CHARS = list("әғқңөүұіһӘҒҚҢӨҮҰІҺ")
KAZ_LONG = "Қазақстан Республикасы — Орталық Азиядағы мемлекет. Жер аумағы бойынша әлемде тоғызыншы орында. Халық саны 19 миллионнан астам. Мемлекеттік тілі — қазақ тілі."
ENG_LONG = "The Republic of Kazakhstan is a country in Central Asia. It is the ninth largest country in the world by area. The population is over 19 million people. The state language is Kazakh."

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
            log.warning("  ppl err: %s", str(e)[:80])
    return math.exp(total_loss / total_toks) if total_toks > 0 else float("inf")

def gen(model, tokenizer, prompt, max_new=80):
    try:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.2,
                                 pad_token_id=pad)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return "[ERROR]"

def bench(name, repo):
    log.info("=" * 60)
    log.info("MODEL: %s (%s)", name, repo)
    r = {"name": name, "repo": repo, "error": None}
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        r["vocab_size"] = len(tok)
        vocab = tok.get_vocab()
        cov = sum(1 for ch in KAZ_CHARS if any((p+ch) in vocab for p in ["", "\u2581", "\u0120", " "]))
        r["kaz_chars"] = f"{cov}/{len(KAZ_CHARS)}"
        kt = tok.encode(KAZ_LONG, add_special_tokens=False)
        et = tok.encode(ENG_LONG, add_special_tokens=False)
        r["fert_kaz"] = round(len(kt)/len(KAZ_LONG.split()), 2)
        r["fert_eng"] = round(len(et)/len(ENG_LONG.split()), 2)
        r["kk_en_ratio"] = round(r["fert_kaz"]/r["fert_eng"], 2) if r["fert_eng"] else 0

        log.info("  Loading...")
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
        model.eval()
        r["params_B"] = round(sum(p.numel() for p in model.parameters())/1e9, 3)
        log.info("  %.3fB params, vocab %d, fert_kk %.2f", r["params_B"], r["vocab_size"], r["fert_kaz"])

        log.info("  PPL...")
        r["ppl_kaz"] = round(compute_ppl(model, tok, KAZ_TEXTS), 2)
        r["ppl_rus"] = round(compute_ppl(model, tok, RUS_TEXTS), 2)
        r["ppl_eng"] = round(compute_ppl(model, tok, ENG_TEXTS), 2)
        log.info("  PPL kaz=%.1f rus=%.1f eng=%.1f", r["ppl_kaz"], r["ppl_rus"], r["ppl_eng"])

        log.info("  Gen...")
        r["gen"] = []
        for prompt in GEN_PROMPTS:
            text = gen(model, tok, prompt)
            r["gen"].append({"prompt": prompt, "output": text})
            log.info("    %s", text[:120])

        del model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        log.error("  FAILED: %s", str(e)[:200])
        r["error"] = str(e)[:300]
        gc.collect(); torch.cuda.empty_cache()
    return r

def main():
    log.info("exp033 10B benchmark: %d models", len(MODELS))
    results = []
    for name, repo in MODELS:
        r = bench(name, repo)
        results.append(r)
        with open("/root/exp033_10b_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("\n" + "=" * 120)
    log.info("%-22s %6s %8s %8s %8s %8s %7s %6s %6s", "Model","Params","Vocab","PPL_kk","PPL_ru","PPL_en","Frt_kk","kk/en","KzCh")
    log.info("-" * 120)
    for r in sorted(results, key=lambda x: x.get("ppl_kaz", 99999)):
        if r.get("error"):
            log.info("%-22s FAILED", r["name"])
            continue
        log.info("%-22s %5.2fB %8d %8.1f %8.1f %8.1f %7.2f %5.2fx %6s",
                 r["name"], r.get("params_B",0), r.get("vocab_size",0),
                 r.get("ppl_kaz",0), r.get("ppl_rus",0), r.get("ppl_eng",0),
                 r.get("fert_kaz",0), r.get("kk_en_ratio",0), r.get("kaz_chars","?"))
    log.info("DONE")

if __name__ == "__main__":
    main()
