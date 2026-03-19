#!/bin/bash
# ============================================================================
# exp025: GEC v4 — diff-format with thinking, 10 epochs, 80% clean
# Format: <тег> text\n💭 explanation\n→ corrected
# ============================================================================

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXPERIMENT="exp025_gec_v4"
BASE_MODEL="stukenov/sozkz-core-llama-300m-kk-base-v1"
HF_REPO="stukenov/sozkz-core-llama-300m-kk-gec-v4"

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

tg "Started GEC v4 (diff-format, 10ep, 80% clean) on ${NUM_GPUS}x GPU"
pip install -q torch transformers datasets huggingface-hub safetensors accelerate 2>&1 | tail -3

mkdir -p /root/gec_v4

cat > /root/gec_v4/train.py << 'PYEOF'
import os, json, time, random, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast, Trainer, TrainingArguments
from huggingface_hub import hf_hub_download
from datasets import Dataset
from collections import defaultdict

BASE_MODEL = os.environ.get("BASE_MODEL", "stukenov/sozkz-core-llama-300m-kk-base-v1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = "/root/gec_v4/output"

TAG_MAP = {
    "1_septik": "септік", "3_vowel_harmony": "сингармонизм",
    "4_personal_ending": "жіктік", "5_possessive": "тәуелдік",
    "6_plural": "көптік", "7_tense": "шақ",
    "8_negation": "болымсыз", "10_postposition": "шылау",
    "grammar": "жалғау", "noise": "қате",
    "2_word_order": "сөз_тәртібі", "9_complex_sentence": "құрмалас",
}

TAG_DESCRIPTIONS = {
    "септік": "септік жалғау", "сингармонизм": "дауысты дыбыс үндесімі",
    "жіктік": "жіктік жалғау", "тәуелдік": "тәуелдік жалғау",
    "көптік": "көптік жалғау", "шақ": "шақ жалғау",
    "болымсыз": "болымсыз етістік", "шылау": "шылау сөз",
    "жалғау": "жалғау қатесі", "қате": "емле қатесі",
    "сөз_тәртібі": "сөз тәртібі", "құрмалас": "құрмалас сөйлем",
}

def find_diff(original, target):
    """Find the specific word-level diff between original and target."""
    orig_words = original.split()
    tgt_words = target.split()
    diffs = []
    # Simple word-level diff
    max_len = max(len(orig_words), len(tgt_words))
    for i in range(min(len(orig_words), len(tgt_words))):
        if orig_words[i] != tgt_words[i]:
            diffs.append(f"{orig_words[i]}→{tgt_words[i]}")
    if len(orig_words) > len(tgt_words):
        for i in range(len(tgt_words), len(orig_words)):
            diffs.append(f"{orig_words[i]}→(жою)")
    elif len(tgt_words) > len(orig_words):
        for i in range(len(orig_words), len(tgt_words)):
            diffs.append(f"(қосу)→{tgt_words[i]}")
    return ", ".join(diffs[:3]) if diffs else ""

def word_edit_distance(a, b):
    """Edit distance at word level."""
    wa, wb = a.split(), b.split()
    if len(wa) < len(wb): return word_edit_distance(b, a)
    if not wb: return len(wa)
    prev = list(range(len(wb) + 1))
    for i, ca in enumerate(wa):
        curr = [i + 1]
        for j, cb in enumerate(wb):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca != cb)))
        prev = curr
    return prev[-1]

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
skipped_long = 0
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
                if et == "unknown" or et == "clean": continue
                # FILTER: only pairs where word edit distance <= 2
                wed = word_edit_distance(inp, tgt)
                if wed > 2:
                    skipped_long += 1
                    continue
                by_type[et].append({"input": inp, "target": tgt, "error_type": et})
        print(f"  {fname}: loaded")
    except Exception as e:
        print(f"  {fname}: SKIP ({e})")

print(f"Total scanned: {total}, skipped (edit>2): {skipped_long}")
for et, exs in sorted(by_type.items(), key=lambda x: -len(x[1])):
    print(f"  {et}: {len(exs)}")

# Build training examples with thinking format
examples = []
for et, exs in by_type.items():
    tag = TAG_MAP.get(et, "грамматика")
    tag_desc = TAG_DESCRIPTIONS.get(tag, "грамматика қатесі")
    for ex in exs:
        diff = find_diff(ex["input"], ex["target"])
        if not diff: continue
        thinking = f"{diff} ({tag_desc})"
        examples.append({
            "input": ex["input"],
            "target": ex["target"],
            "tag": tag,
            "thinking": thinking,
        })

error_count = len(examples)
print(f"\nError examples with thinking: {error_count}")

# Add clean examples — 80% of total
# clean_target = error_count * 4 means 80% clean (4:1 ratio)
clean_target = error_count * 4
clean_added = 0
for et, exs in by_type.items():
    for ex in exs:
        if clean_added >= clean_target: break
        examples.append({
            "input": ex["target"],
            "target": ex["target"],
            "tag": "таза",
            "thinking": "қате жоқ",
        })
        clean_added += 1
    if clean_added >= clean_target: break

# Also add clean with error tags (teach model to say "қате жоқ" for any tag on clean text)
for et, exs in by_type.items():
    tag = TAG_MAP.get(et, "грамматика")
    for ex in exs:
        if clean_added >= clean_target: break
        examples.append({
            "input": ex["target"],
            "target": ex["target"],
            "tag": tag,
            "thinking": "қате жоқ",
        })
        clean_added += 1
    if clean_added >= clean_target: break

random.seed(42)
random.shuffle(examples)

clean_count = sum(1 for e in examples if e["thinking"] == "қате жоқ")
total_ex = len(examples)
print(f"GEC v4: {total_ex} total ({error_count} errors + {clean_count} clean, {clean_count*100//total_ex}% clean)")

# Load model + tokenizer
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

# Collator with thinking format
class GECThinkingCollator:
    def __init__(self, tok, ml=512):
        self.tok, self.ml = tok, ml
    def __call__(self, features):
        bids, bmask, blabels = [], [], []
        for f in features:
            prompt = f"<{f['tag']}> {f['input']}\n"
            full = prompt + f"\U0001f4ad {f['thinking']}\n\u2192 " + f["target"] + self.tok.eos_token
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
pbs, ga = 32, max(1, 128 // (32 * num_gpus))

args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=10,
    per_device_train_batch_size=pbs, gradient_accumulation_steps=ga,
    learning_rate=2e-5, warmup_ratio=0.05, weight_decay=0.01,
    lr_scheduler_type="cosine", max_grad_norm=1.0, bf16=True,
    logging_steps=50, save_strategy="epoch", save_total_limit=2,
    report_to="none", dataloader_num_workers=4, remove_unused_columns=False,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  data_collator=GECThinkingCollator(tokenizer), processing_class=tokenizer)

eff_bs = pbs * num_gpus * ga
print(f"Training GEC v4: 10ep, LR=2e-5, BS={pbs}x{num_gpus}x{ga}={eff_bs}")
t0 = time.time()
trainer.train()
tt = time.time() - t0
print(f"Done GEC v4: {tt/60:.1f}min")

fd = f"{OUTPUT_DIR}/final"
trainer.save_model(fd); tokenizer.save_pretrained(fd)
with open(f"{fd}/results.json", "w") as f:
    json.dump({"experiment": "exp025_gec_v4", "format": "thinking",
               "epochs": 10, "lr": 2e-5, "train_min": round(tt/60,1),
               "examples": total_ex, "errors": error_count, "clean": clean_count,
               "clean_pct": round(clean_count*100/total_ex,1),
               "filter": "word_edit_distance<=2"}, f, indent=2)
print(f"Saved to {fd}")
PYEOF

# --- Train ---
tg "Training GEC v4 (10ep, 80% clean, thinking format)"
T0=$(date +%s)
PYTHONUNBUFFERED=1 BASE_MODEL=$BASE_MODEL HF_TOKEN=$HF_TOKEN \
    torchrun --nproc_per_node=$NUM_GPUS /root/gec_v4/train.py 2>&1 | tee /root/gec_v4/train.log
T1=$(date +%s)
tg "Training done in $(( (T1-T0)/60 ))min"

# --- Upload ---
tg "Uploading to HF..."
python3 << UPLOADEOF
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast
from huggingface_hub import HfApi, create_repo
token = os.environ.get("HF_TOKEN")
repo = "$HF_REPO"
fd = "/root/gec_v4/output/final"
if not os.path.exists(fd):
    print("Output missing!")
    exit(1)
print(f"Uploading to {repo}...")
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
print("Uploaded!")
UPLOADEOF

# Verify upload
UPLOAD_OK=0
python3 -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('https://huggingface.co/api/models/$HF_REPO', timeout=30)
    if r.status == 200: print('HF verify OK'); sys.exit(0)
except: pass
print('HF verify FAILED'); sys.exit(1)
" && UPLOAD_OK=1

if [ "$UPLOAD_OK" = "1" ]; then
    tg "Upload verified: $HF_REPO"
else
    tg "WARNING: Upload verification failed!"
fi

T_END=$(date +%s)
TOTAL_MIN=$(( (T_END-T0)/60 ))
tg "Total: ${TOTAL_MIN}min"

# Self-destruct only if upload OK
if [ "$UPLOAD_OK" = "1" ] && [ -n "$VAST_API_KEY" ] && [ -n "$VAST_INSTANCE_ID" ]; then
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
