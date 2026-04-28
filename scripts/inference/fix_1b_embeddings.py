#!/usr/bin/env python3
"""Fix exp028 1.08B model: bake embedding scaling into weights.

Training used: x = self.emb(idx) * sqrt(2048) in forward()
HF Llama does NOT do this. Fix: multiply embed_tokens.weight by sqrt(2048).
"""
import torch
import math
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

MODEL = "stukenov/sozkz-core-llama-1b-kk-base-v1"
TOKEN = open("/root/.cache/huggingface/token").read().strip()
SCALE = math.sqrt(2048)

print("Loading model from HF...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
params = sum(p.numel() for p in model.parameters()) / 1e6
print("Loaded: %.1fM params" % params)

print("Scaling embed_tokens.weight by %.2f..." % SCALE)
with torch.no_grad():
    model.model.embed_tokens.weight.mul_(SCALE)
    if model.config.tie_word_embeddings:
        print("  Tied embeddings - lm_head auto-updated")

w = model.model.embed_tokens.weight.float()
print("  Stats after: mean=%.4f std=%.4f" % (w.mean().item(), w.std().item()))

print("\nInference test...")
tok_file = hf_hub_download(MODEL, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
tokenizer.pad_token_id = 1

model = model.to("cuda")

test_prompts = [
    "Қазақстан Президенті",
    "Алматы қаласында бүгін",
    "Білім беру жүйесі",
]

all_ok = True
for prompt in test_prompts:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=80, do_sample=True,
            temperature=0.7, top_p=0.9, repetition_penalty=1.1,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    print("\n[%s] (cyrillic: %d)" % (prompt, cyrillic))
    print("  %s" % text[:300])
    if cyrillic < 10:
        all_ok = False

if all_ok:
    print("\n\nOutput looks OK! Uploading fixed model...")
    model = model.to("cpu")
    model.push_to_hub(MODEL, token=TOKEN)
    tokenizer.push_to_hub(MODEL, token=TOKEN)
    print("FIX COMPLETE - model re-uploaded to HF")
else:
    print("\n\nWARNING: Output still bad. NOT uploading.")
