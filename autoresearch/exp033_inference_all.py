#!/usr/bin/env python3
"""exp033: Run 10 Kazakh continuation prompts on all models."""

import gc
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    ("LLaMA-3.2-3B", "meta-llama/Llama-3.2-3B"),
    ("LLaMA-3.2-1B", "meta-llama/Llama-3.2-1B"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("Gemma-2-2B", "google/gemma-2-2b"),
    ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B"),
    ("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M"),
    ("StableLM-2-1.6B", "stabilityai/stablelm-2-1_6b"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"),
    ("SozKZ-TinyLlama-kk", "stukenov/sozkz-core-tinyllama-1b-kk-ru-v1"),
    ("SozKZ-Llama-600M", "stukenov/sozkz-core-llama-600m-kk-base-v1"),
]

PROMPTS = [
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


def main():
    log.info("exp033 inference: %d models x %d prompts", len(MODELS), len(PROMPTS))
    results = {}

    for name, repo in MODELS:
        log.info("=" * 60)
        log.info("MODEL: %s", name)
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(DEVICE)
            mdl.eval()
            gens = []
            for i, prompt in enumerate(PROMPTS):
                text = gen(mdl, tok, prompt)
                gens.append({"prompt": prompt, "output": text})
                log.info("  [%d] %s", i+1, text[:150])
            results[name] = {"repo": repo, "generations": gens}
            del mdl
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            log.error("  FAILED: %s", str(e)[:200])
            results[name] = {"repo": repo, "error": str(e)[:200]}

    with open("/root/exp033_inference.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("DONE")


if __name__ == "__main__":
    main()
