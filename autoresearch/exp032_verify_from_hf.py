#!/usr/bin/env python3
"""exp032: Verify model loads and works from HuggingFace on a fresh environment."""

import logging
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

REPO_ID = "stukenov/sozkz-core-tinyllama-1b-kk-ru-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TESTS = [
    ("kaz", "Қазақстан — бұл"),
    ("kaz", "Бүгін ауа райы"),
    ("rus", "Казахстан — это"),
    ("rus", "Сегодня погода"),
    ("eng", "Hello, my name is"),
]


def main():
    log.info("=== FRESH VERIFICATION FROM HUGGINGFACE ===")
    log.info("Repo: %s, Device: %s", REPO_ID, DEVICE)

    # 1. Tokenizer
    log.info("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    log.info("  Vocab: %d", len(tokenizer))
    assert len(tokenizer) > 40000, "Vocab too small"
    log.info("  PASS")

    # 2. Model
    log.info("Step 2: Loading model...")
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("  Params: %.1fB", total_params / 1e9)
    assert total_params > 1_000_000_000, "Model too small"
    log.info("  PASS")

    # 3. Fertility
    log.info("Step 3: Fertility check...")
    kaz_text = "Қазақстан Республикасы — Орталық Азиядағы мемлекет"
    kaz_toks = tokenizer.encode(kaz_text, add_special_tokens=False)
    fertility = len(kaz_toks) / len(kaz_text.split())
    log.info("  Kazakh: %.2f tok/word", fertility)
    assert fertility < 3.0, "Fertility too high — tokenizer not extended?"
    log.info("  PASS")

    # 4. Generation
    log.info("Step 4: Generation...")
    model.eval()
    all_ok = True
    for lang, prompt in TESTS:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.2)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        ok = len(text) > len(prompt) + 10
        log.info("  [%s] %s: %s", "PASS" if ok else "FAIL", lang, text[:150])
        if not ok:
            all_ok = False

    # 5. PPL
    log.info("Step 5: Perplexity...")
    enc = tokenizer("Қазақстан Республикасы — Орталық Азиядағы мемлекет.", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    ppl = math.exp(loss.item())
    log.info("  Kazakh PPL: %.2f", ppl)
    assert ppl < 100, "PPL too high"
    log.info("  PASS")

    log.info("=" * 60)
    if all_ok:
        log.info("ALL CHECKS PASSED!")
    else:
        log.info("SOME CHECKS FAILED")
        sys.exit(1)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
