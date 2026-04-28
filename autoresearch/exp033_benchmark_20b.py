#!/usr/bin/env python3
"""exp033: Extended benchmark — 20B+ models, diverse providers, MoE."""
import gc, json, logging, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    ("InternLM2.5-7B", "internlm/internlm2_5-7b"),
    ("InternLM2-20B", "internlm/internlm2-20b"),
    ("Baichuan2-7B", "baichuan-inc/Baichuan2-7B-Base"),
    ("Baichuan2-13B", "baichuan-inc/Baichuan2-13B-Base"),
    ("XGLM-7.5B", "facebook/xglm-7.5B"),
    ("mGPT-13B", "ai-forever/mGPT-13B"),
    ("Bloom-7B", "bigscience/bloom-7b1"),
    ("CroissantLLM-1.3B", "croissantllm/CroissantLLMBase"),
    ("Aya-23-8B", "CohereForAI/aya-23-8B"),
    ("Jais-13B", "inceptionai/jais-13b"),
    ("SEA-LION-v3-7B", "aisingapore/llama3-8b-cpt-sea-lionv3-base"),
    ("Mixtral-8x7B", "mistralai/Mixtral-8x7B-v0.1"),
    ("Qwen1.5-MoE-A2.7B", "Qwen/Qwen1.5-MoE-A2.7B"),
    ("DBRX-base", "databricks/dbrx-base"),
]
KAZ_TEXTS = [
    "Қазақстан Республикасы — Орталық Азиядағы мемлекет. Астанасы — Астана қаласы.",
    "Қазақ тілі — түркі тілдерінің қыпшақ тобына жататын тіл.",
    "Білім беру жүйесі — мемлекеттік саясаттың маңызды бағыттарының бірі.",
    "Алматы — Қазақстанның ең ірі қаласы және мәдени астанасы.",
    "Қазақстан экономикасы мұнай мен газ өндірісіне негізделген.",
    "Абай Құнанбайұлы — ұлы қазақ ақыны, ағартушы, ойшыл.",
    "Наурыз мейрамы — қазақ халқының көне мерекесі.",
    "Қазақтың ұлттық тағамы — бешбармақ.",
    "Байқоңыр ғарыш айлағы — әлемдегі ең ірі ғарыш айлағы.",
    "Қазақстан Конституциясы 1995 жылы 30 тамызда қабылданды.",
]
RUS_TEXTS = [
    "Республика Казахстан — государство в Центральной Азии.",
    "Казахский язык относится к кыпчакской группе тюркских языков.",
    "Система образования является важнейшим направлением политики.",
    "Алматы — крупнейший город Казахстана.",
    "Экономика Казахстана основана на добыче нефти и газа.",
]
ENG_TEXTS = [
    "The Republic of Kazakhstan is a country in Central Asia.",
    "The Kazakh language belongs to the Kipchak group of Turkic languages.",
    "The education system is one of the most important areas of state policy.",
    "Almaty is the largest city in Kazakhstan.",
]
GEN_PROMPTS = [
    "Қазақстан — бұл", "Қазақ тілі — түркі тілдерінің",
    "Алматы қаласында бүгін", "Білім беру саласында",
    "Абай Құнанбайұлы — ұлы", "Наурыз мейрамы — қазақ халқының",
    "Қазақстан экономикасы", "Ұлттық тағамдар арасында",
    "Қазақ мәдениеті мен өнері", "Жастарға арналған",
]
KAZ_CHARS = list("әғқңөүұіһӘҒҚҢӨҮҰІҺ")
KAZ_LONG = "Қазақстан Республикасы Орталық Азиядағы мемлекет. Жер аумағы бойынша әлемде тоғызыншы орында."
ENG_LONG = "The Republic of Kazakhstan is a country in Central Asia. It is the ninth largest country in the world."

def compute_ppl(model, tokenizer, texts):
    tl, tt = 0, 0
    for text in texts:
        try:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad(): out = model(**enc, labels=enc["input_ids"])
            tl += out.loss.item() * enc["input_ids"].shape[1]
            tt += enc["input_ids"].shape[1]
        except Exception as e:
            log.warning("  ppl: %s", str(e)[:60])
    return math.exp(tl/tt) if tt > 0 else float("inf")

def gen_text(model, tokenizer, prompt):
    try:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.2, pad_token_id=pad)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except: return "[ERROR]"

def bench(name, repo):
    log.info("=" * 60)
    log.info("MODEL: %s (%s)", name, repo)
    r = {"name": name, "repo": repo, "error": None}
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        r["vocab_size"] = len(tok)
        v = tok.get_vocab()
        r["kaz_chars"] = f"{sum(1 for c in KAZ_CHARS if any((p+c) in v for p in ['',chr(9601),chr(288),' ']))}/{len(KAZ_CHARS)}"
        kt = tok.encode(KAZ_LONG, add_special_tokens=False)
        et = tok.encode(ENG_LONG, add_special_tokens=False)
        kw, ew = len(KAZ_LONG.split()), len(ENG_LONG.split())
        r["fert_kaz"] = round(len(kt)/kw, 2)
        r["fert_eng"] = round(len(et)/ew, 2)
        r["kk_en_ratio"] = round(r["fert_kaz"]/r["fert_eng"], 2) if r["fert_eng"] else 0
        log.info("  Tok: vocab=%d fert_kk=%.2f kk/en=%.2fx kzch=%s", r["vocab_size"], r["fert_kaz"], r["kk_en_ratio"], r["kaz_chars"])

        log.info("  Loading...")
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
        model.eval()
        r["params_B"] = round(sum(p.numel() for p in model.parameters())/1e9, 3)
        r["vram_gb"] = round(torch.cuda.max_memory_allocated()/1e9, 1)
        log.info("  %.2fB params, %.1fGB VRAM", r["params_B"], r["vram_gb"])

        log.info("  PPL...")
        r["ppl_kaz"] = round(compute_ppl(model, tok, KAZ_TEXTS), 2)
        r["ppl_rus"] = round(compute_ppl(model, tok, RUS_TEXTS), 2)
        r["ppl_eng"] = round(compute_ppl(model, tok, ENG_TEXTS), 2)
        log.info("  PPL kaz=%.1f rus=%.1f eng=%.1f", r["ppl_kaz"], r["ppl_rus"], r["ppl_eng"])

        log.info("  Gen...")
        r["gen"] = []
        for prompt in GEN_PROMPTS:
            text = gen_text(model, tok, prompt)
            r["gen"].append({"prompt": prompt, "output": text})
            log.info("    %s", text[:120])
        del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        log.error("  FAILED: %s", str(e)[:200])
        r["error"] = str(e)[:300]
        gc.collect(); torch.cuda.empty_cache()
    return r

def main():
    log.info("exp033 20B+: %d models", len(MODELS))
    results = []
    for name, repo in MODELS:
        r = bench(name, repo)
        results.append(r)
        with open("/root/exp033_20b_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("\n" + "=" * 130)
    log.info("%-22s %5s %8s %7s %7s %7s %6s %5s %5s %5s",
             "Model","Param","Vocab","PPLkk","PPLru","PPLen","Frtkk","kk/en","VRAM","KzCh")
    log.info("-" * 130)
    for r in sorted(results, key=lambda x: x.get("ppl_kaz", 99999)):
        if r.get("error"): log.info("%-22s FAIL: %s", r["name"], r.get("error","")[:50]); continue
        log.info("%-22s %4.1fB %8d %7.1f %7.1f %7.1f %6.2f %4.2fx %4.0fG %5s",
                 r["name"], r.get("params_B",0), r.get("vocab_size",0),
                 r.get("ppl_kaz",0), r.get("ppl_rus",0), r.get("ppl_eng",0),
                 r.get("fert_kaz",0), r.get("kk_en_ratio",0), r.get("vram_gb",0), r.get("kaz_chars","?"))
    log.info("DONE")

if __name__ == "__main__":
    main()
