#!/bin/bash
# ============================================================================
# exp023a + exp023b: GEC v2 — two variants, 1 epoch each
# a) multi-tag (8 tags + таза), no unknown
# b) single <грамматика> + таза, no unknown
# ============================================================================

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXPERIMENT="exp023_gec_v2"
BASE_MODEL="stukenov/sozkz-core-llama-300m-kk-base-v1"

tg() {
    local msg="$1"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
        TG_MSG="[$EXPERIMENT] $msg" python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ['TG_CHAT_ID'],
    'text': os.environ['TG_MSG'],
}).encode()
try: urllib.request.urlopen('https://api.telegram.org/bot' + os.environ['TG_TOKEN'] + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    fi
    echo "[TG] $msg"
}

tg "Started GEC v2 on ${NUM_GPUS}x GPU"
pip install -q torch transformers datasets huggingface-hub safetensors accelerate 2>&1 | tail -3

mkdir -p /root/gec_v2

cat > /root/gec_v2/train.py << 'PYEOF'
import os, json, time, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast, Trainer, TrainingArguments
from huggingface_hub import hf_hub_download
from datasets import Dataset
from collections import defaultdict

BASE_MODEL = os.environ.get("BASE_MODEL", "stukenov/sozkz-core-llama-300m-kk-base-v1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
VARIANT = os.environ.get("VARIANT", "a")  # "a" = multi-tag, "b" = single грамматика
OUTPUT_DIR = f"/root/gec_v2/output_{VARIANT}"

TAG_MAP = {
    "1_septik": "септік", "3_vowel_harmony": "сингармонизм",
    "4_personal_ending": "жіктік", "5_possessive": "тәуелдік",
    "6_plural": "көптік", "7_tense": "шақ",
    "8_negation": "болымсыз", "10_postposition": "шылау",
    "grammar": "жалғау", "noise": "қате",
    "2_word_order": "сөз_тәртібі", "9_complex_sentence": "құрмалас",
}

# Load data
print("Loading GEC data...")
REPO = "stukenov/sozkz-corpus-synthetic-kk-gec-v1"
FILES = [
    "data/grammar_balanced_v2/train.jsonl", "data/grammar_v2/train.jsonl",
    "data/grammar_combined/train.jsonl", "data/grammar_focused/train.jsonl",
    "data/processed/train.jsonl", "data/processed_v2/train.jsonl",
    "data/processed_v3/train.jsonl",
]

by_type = defaultdict(list)
total = 0
for fname in FILES:
    try:
        local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset", token=HF_TOKEN)
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                total += 1
                et = ex.get("error_type", "unknown")
                inp = ex.get("input", "")
                tgt = ex.get("target", "")
                if not inp or not tgt: continue
                # SKIP unknown — main source of noise
                if et == "unknown": continue
                # SKIP clean (will add separately)
                if et == "clean": continue
                by_type[et].append({"input": inp, "target": tgt, "error_type": et})
        print(f"  {fname}: loaded")
    except Exception as e:
        print(f"  {fname}: SKIP ({e})")

print(f"Total scanned: {total}")
for et, exs in sorted(by_type.items(), key=lambda x: -len(x[1])):
    print(f"  {et}: {len(exs)}")

# Build training examples
examples = []
for et, exs in by_type.items():
    tag = TAG_MAP.get(et, "грамматика")
    for ex in exs:
        if VARIANT == "b":
            # Single tag: everything → <грамматика>
            tag_to_use = "грамматика"
        else:
            # Multi-tag: use specific tag
            tag_to_use = tag
        examples.append({"input": ex["input"], "target": ex["target"], "tag": tag_to_use})

# Add clean examples (таза)
clean_target = 50000 if VARIANT == "b" else 30000
# Use clean examples from dataset + generate from targets
clean_added = 0
for et, exs in by_type.items():
    for ex in exs:
        if clean_added >= clean_target: break
        # Use target as both input and target (clean pair)
        examples.append({"input": ex["target"], "target": ex["target"], "tag": "таза"})
        clean_added += 1
    if clean_added >= clean_target: break

random.seed(42)
random.shuffle(examples)

error_count = sum(1 for e in examples if e["tag"] != "таза")
clean_count = sum(1 for e in examples if e["tag"] == "таза")
print(f"\nVariant {VARIANT}: {len(examples)} total ({error_count} errors + {clean_count} clean, {clean_count*100//len(examples)}% clean)")

# Load model
print(f"Loading model: {BASE_MODEL}")
MODEL_ID = "stukenov/sozkz-core-llama-300m-kk-gec-v1"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
except (ValueError, ImportError):
    tok_file = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.json", token=HF_TOKEN)
    tokenizer = GPT2TokenizerFast(tokenizer_file=tok_file)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, token=HF_TOKEN)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Collator
class GECCollator:
    def __init__(self, tok, ml=512):
        self.tok, self.ml = tok, ml
    def __call__(self, features):
        bids, bmask, blabels = [], [], []
        for f in features:
            prompt = f"<{f['tag']}> {f['input']}\n\u2192 "
            full = prompt + f["target"] + self.tok.eos_token
            enc = self.tok(full, truncation=True, max_length=self.ml, add_special_tokens=False)
            ids, mask = enc["input_ids"], enc["attention_mask"]
            penc = self.tok(prompt, truncation=True, max_length=self.ml, add_special_tokens=False)
            pl = len(penc["input_ids"])
            labels = [-100]*pl + ids[pl:]
            bids.append(ids); bmask.append(mask); blabels.append(labels)
        mx = max(len(x) for x in bids)
        pid = self.tok.pad_token_id or 0
        for i in range(len(bids)):
            p = mx - len(bids[i])
            bids[i] += [pid]*p; bmask[i] += [0]*p; blabels[i] += [-100]*p
        return {"input_ids": torch.tensor(bids), "attention_mask": torch.tensor(bmask), "labels": torch.tensor(blabels)}

train_ds = Dataset.from_list(examples)
num_gpus = torch.cuda.device_count()
pbs, ga = 8, max(1, 64 // (8 * num_gpus))

args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=1,
    per_device_train_batch_size=pbs, gradient_accumulation_steps=ga,
    learning_rate=5e-6, warmup_ratio=0.03, weight_decay=0.01,
    lr_scheduler_type="cosine", max_grad_norm=1.0, bf16=True,
    logging_steps=20, save_strategy="no", report_to="none",
    dataloader_num_workers=4, remove_unused_columns=False,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  data_collator=GECCollator(tokenizer), processing_class=tokenizer)

print(f"Training variant {VARIANT}: 1ep, BS={pbs}x{num_gpus}x{ga}={pbs*num_gpus*ga}")
t0 = time.time()
trainer.train()
tt = time.time() - t0
print(f"Done variant {VARIANT}: {tt/60:.1f}min")

fd = f"{OUTPUT_DIR}/final"
trainer.save_model(fd); tokenizer.save_pretrained(fd)
with open(f"{fd}/results.json", "w") as f:
    json.dump({"variant": VARIANT, "train_min": round(tt/60,1), "examples": len(examples),
               "errors": error_count, "clean": clean_count}, f, indent=2)
print(f"Saved to {fd}")
PYEOF

# --- Run variant A (multi-tag) ---
tg "Training variant A (multi-tag, no unknown)"
T0=$(date +%s)
PYTHONUNBUFFERED=1 VARIANT=a BASE_MODEL=$BASE_MODEL HF_TOKEN=$HF_TOKEN \
    torchrun --nproc_per_node=$NUM_GPUS /root/gec_v2/train.py 2>&1 | tee /root/gec_v2/train_a.log
T1=$(date +%s)
tg "Variant A done in $(( (T1-T0)/60 ))min"

# --- Run variant B (single грамматика) ---
tg "Training variant B (single грамматика, no unknown)"
PYTHONUNBUFFERED=1 VARIANT=b BASE_MODEL=$BASE_MODEL HF_TOKEN=$HF_TOKEN \
    torchrun --nproc_per_node=$NUM_GPUS /root/gec_v2/train.py 2>&1 | tee /root/gec_v2/train_b.log
T2=$(date +%s)
tg "Variant B done in $(( (T2-T1)/60 ))min"

# --- Upload both ---
tg "Uploading both models..."
python3 << UPLOADEOF
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast
from huggingface_hub import HfApi, create_repo, hf_hub_download
token = os.environ.get("HF_TOKEN")

for variant, repo in [("a", "stukenov/sozkz-core-llama-300m-kk-gec-v2a"), ("b", "stukenov/sozkz-core-llama-300m-kk-gec-v2b")]:
    fd = f"/root/gec_v2/output_{variant}/final"
    if not os.path.exists(fd):
        print(f"Variant {variant}: output missing, skip")
        continue
    print(f"Uploading variant {variant} to {repo}...")
    create_repo(repo, token=token, exist_ok=True)
    m = AutoModelForCausalLM.from_pretrained(fd)
    m.push_to_hub(repo, token=token)
    try:
        t = AutoTokenizer.from_pretrained(fd)
    except:
        tok_file = os.path.join(fd, "tokenizer.json")
        t = GPT2TokenizerFast(tokenizer_file=tok_file)
    t.push_to_hub(repo, token=token)
    HfApi().upload_file(path_or_fileobj=f"{fd}/results.json", path_in_repo="results.json", repo_id=repo, token=token)
    print(f"  {variant} uploaded!")

UPLOADEOF
tg "Both models uploaded!"

T_END=$(date +%s)
TOTAL_MIN=$(( (T_END-T0)/60 ))
tg "Total: ${TOTAL_MIN}min"

# Self-destruct
if [ -n "$VAST_API_KEY" ] && [ -n "$VAST_INSTANCE_ID" ]; then
    python3 -c "
import urllib.request, os
req = urllib.request.Request(
    'https://console.vast.ai/api/v0/instances/' + os.environ['VAST_INSTANCE_ID'] + '/',
    method='DELETE', headers={'Authorization': 'Bearer ' + os.environ['VAST_API_KEY']})
try: urllib.request.urlopen(req, timeout=30)
except: pass
"
    tg "Instance destroyed"
fi
