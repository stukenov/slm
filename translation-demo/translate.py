"""
Local inference for HPLT/translate-en-kk-v2.0-hplt_opus
English → Kazakh translation via CTranslate2
"""

import os
import shutil
import ctranslate2
import sentencepiece as spm
from ctranslate2.converters.marian import MarianConverter
from huggingface_hub import hf_hub_download

MODEL_ID = "HPLT/translate-en-kk-v2.0-hplt_opus"
BASE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE, "model_cache")
CT2_DIR = os.path.join(BASE, "model_ct2")

# Download
print("Downloading model files...")
os.makedirs(CACHE_DIR, exist_ok=True)
npz_path = hf_hub_download(MODEL_ID, "model.npz.best-chrf.npz", local_dir=CACHE_DIR)
spm_path = hf_hub_download(MODEL_ID, "model.en-kk.spm", local_dir=CACHE_DIR)
vocab_path = hf_hub_download(MODEL_ID, "model.en-kk.vocab", local_dir=CACHE_DIR)

# Convert vocab from "token\tvalue" to "token:idx" (ct2 marian format)
fixed_vocab = os.path.join(CACHE_DIR, "vocab_fixed.txt")
with open(vocab_path) as fin, open(fixed_vocab, "w") as fout:
    for idx, line in enumerate(fin):
        token = line.strip().split("\t")[0]
        # Escape token for ct2 format
        fout.write(f'"{token}":{idx}\n')

# Convert to CT2
if not os.path.exists(os.path.join(CT2_DIR, "model.bin")):
    print("Converting to CTranslate2...")
    converter = MarianConverter(npz_path, [fixed_vocab])
    converter.convert(CT2_DIR)
    print("Done!")

# Load
print("Loading model...")
translator = ctranslate2.Translator(CT2_DIR, device="cpu")
sp = spm.SentencePieceProcessor(spm_path)

texts = [
    "The weather is beautiful today, and I want to go for a walk in the park.",
    "Artificial intelligence is transforming the way we live and work.",
    "Kazakhstan is the largest landlocked country in the world.",
    "My grandmother makes the best beshbarmak in the whole village.",
    "Students should read more books to expand their knowledge.",
    "The space program has achieved remarkable progress in recent years.",
    "Clean water is essential for the health of every person on the planet.",
    "Learning a new language opens doors to different cultures and opportunities.",
    "The economy of Central Asia is growing rapidly due to new investments.",
    "Children love playing football in the schoolyard after classes.",
]

print("=" * 80)
print("English → Kazakh Translation (HPLT v2.0)")
print("=" * 80)

for i, text in enumerate(texts, 1):
    tokens = sp.encode(text, out_type=str)
    results = translator.translate_batch([tokens], beam_size=4)
    result = sp.decode(results[0].hypotheses[0])
    print(f"\n[{i}] EN: {text}")
    print(f"    KK: {result}")

print("\n" + "=" * 80)
