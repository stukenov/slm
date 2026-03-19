#!/usr/bin/env python3
"""Debug: compare HF generate vs vLLM to understand output format difference."""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Stop 600M server first to free GPU 0
# Or just use CPU for this small test

MODEL = "stukenov/sozkz-core-llama-600m-kk-gec-v1"
print("Loading %s on cuda..." % MODEL)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model = model.to("cuda")
model.requires_grad_(False)

prompts = [
    "<грамматика> Бүгін ауа райы жақсы\n",
    "<грамматика> Бүгін ауа рйы жақсы\n",
    "<грамматика> Мен мектепке бардым кеше\n",
    "<грамматика> Ол мектепке барған кеше\n",
    "<грамматика> Менің атым Сакен\n",
    "<қате> Бүгін ауа рйы жақсы\n",
]

for prompt in prompts:
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    inputs.pop("token_type_ids", None)
    input_len = len(inputs["input_ids"][0])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )
    full_decoded = tok.decode(out[0], skip_special_tokens=True)
    new_only = tok.decode(out[0][input_len:], skip_special_tokens=True)
    print("PROMPT: %r" % prompt)
    print("  FULL OUTPUT:     %r" % full_decoded)
    print("  NEW TOKENS ONLY: %r" % new_only)
    print()
