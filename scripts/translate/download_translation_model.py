#!/usr/bin/env python3
"""Download and convert HPLT EN→KK translation model for CTranslate2."""

import os
from huggingface_hub import hf_hub_download
from ctranslate2.converters.marian import MarianConverter

MODEL_ID = "HPLT/translate-en-kk-v2.0-hplt_opus"
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "translation-demo")
CACHE = os.path.join(BASE, "model_cache")
CT2 = os.path.join(BASE, "model_ct2")

os.makedirs(CACHE, exist_ok=True)
print(f"Downloading {MODEL_ID}...")
npz = hf_hub_download(MODEL_ID, "model.npz.best-chrf.npz", local_dir=CACHE)
hf_hub_download(MODEL_ID, "model.en-kk.spm", local_dir=CACHE)
vocab = hf_hub_download(MODEL_ID, "model.en-kk.vocab", local_dir=CACHE)

fixed = os.path.join(CACHE, "vocab_fixed.txt")
with open(vocab) as f, open(fixed, "w") as out:
    for i, line in enumerate(f):
        token = line.strip().split("\t")[0]
        out.write(f'"{token}":{i}\n')

if not os.path.exists(os.path.join(CT2, "model.bin")):
    print("Converting to CTranslate2...")
    MarianConverter(npz, [fixed]).convert(CT2)

print("Model ready.")
