#!/bin/bash
# ============================================================================
# exp026: GEC 600M — single грамматика, thinking format, 3 epochs
# ============================================================================

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXPERIMENT="exp026_gec_600m"
BASE_MODEL="stukenov/sozkz-core-llama-600m-kk-base-v1"
HF_REPO="stukenov/sozkz-core-llama-600m-kk-gec-v1"

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

tg "Started GEC 600M (single-tag, 3ep) on ${NUM_GPUS}x GPU"
pip install -q torch transformers datasets huggingface-hub safetensors accelerate 2>&1 | tail -3

mkdir -p /root/gec_600m

cat > /root/gec_600m/train.py << 'PYEOF'
import os, json, time, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast, Trainer, TrainingArguments
from huggingface_hub import hf_hub_download
from datasets import Dataset
from collections import defaultdict

BASE_MODEL = os.environ.get("BASE_MODEL", "stukenov/sozkz-core-llama-600m-kk-base-v1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = "/root/gec_600m/output"

TAG_DESCRIPTIONS = {
    "1_septik": "септік жалғау", "3_vowel_harmony": "дауысты дыбыс үндесімі",
    "4_personal_ending": "жіктік жалғау", "5_possessive": "тәуелдік жалғау",
    "6_plural": "көптік жалғау", "7_tense": "шақ жалғау",
    "8_negation": "болымсыз етістік", "10_postposition": "шылау сөз",
    "grammar": "жалғау қатесі", "noise": "емле қатесі",
}

def find_diff(original, target):
    orig_words = original.split()
    tgt_words = target.split()
    diffs = []
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
total = skipped = 0
for fname in FILES:
    try:
        local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset", token=HF_TOKEN)
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                total += 1
                et = ex.get("error_type", "unknown")
                inp, tgt = ex.get("input", ""), ex.get("target", "")
                if not inp or not tgt: continue
                if et in ("unknown", "clean"): continue
                wed = word_edit_distance(inp, tgt)
                if wed > 2:
                    skipped += 1
                    continue
                by_type[et].append({"input": inp, "target": tgt, "error_type": et})
        print(f"  {fname}: loaded")
    except Exception as e:
        print(f"  {fname}: SKIP ({e})")

print(f"Total: {total}, skipped (edit>2): {skipped}")
for et, exs in sorted(by_type.items(), key=lambda x: -len(x[1])):
    print(f"  {et}: {len(exs)}")

# Build examples — SINGLE TAG <грамматика> for everything
examples = []
for et, exs in by_type.items():
    desc = TAG_DESCRIPTIONS.get(et, "грамматика қатесі")
    for ex in exs:
        diff = find_diff(ex["input"], ex["target"])
        if not diff: continue
        examples.append({
            "input": ex["input"], "target": ex["target"],
            "tag": "грамматика", "thinking": f"{diff} ({desc})",
        })

error_count = len(examples)

# 80% clean
clean_target = error_count * 4
clean_added = 0
for et, exs in by_type.items():
    for ex in exs:
        if clean_added >= clean_target: break
        examples.append({
            "input": ex["target"], "target": ex["target"],
            "tag": "грамматика", "thinking": "қате жоқ",
        })
        clean_added += 1
    if clean_added >= clean_target: break

random.seed(42)
random.shuffle(examples)
total_ex = len(examples)
print(f"\nGEC 600M: {total_ex} total ({error_count} errors + {clean_added} clean, {clean_added*100//total_ex}% clean)")

# Load model
print(f"Loading model: {BASE_MODEL}")
TOK_MODEL = "stukenov/sozkz-core-llama-300m-kk-gec-v1"
try:
    tokenizer = AutoTokenizer.from_pretrained(TOK_MODEL, token=HF_TOKEN)
except (ValueError, ImportError):
    tok_file = hf_hub_download(repo_id=TOK_MODEL, filename="tokenizer.json", token=HF_TOKEN)
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
pbs = 16  # 600M needs more VRAM per sample
ga = max(1, 128 // (pbs * num_gpus))

args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=3,
    per_device_train_batch_size=pbs, gradient_accumulation_steps=ga,
    learning_rate=1.5e-5, warmup_ratio=0.05, weight_decay=0.01,
    lr_scheduler_type="cosine", max_grad_norm=1.0, bf16=True,
    logging_steps=50, save_strategy="no",
    report_to="none", dataloader_num_workers=4, remove_unused_columns=False,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  data_collator=GECCollator(tokenizer), processing_class=tokenizer)

eff_bs = pbs * num_gpus * ga
print(f"Training: 3ep, LR=1.5e-5, BS={pbs}x{num_gpus}x{ga}={eff_bs}")
t0 = time.time()
trainer.train()
tt = time.time() - t0
print(f"Done: {tt/60:.1f}min")

fd = f"{OUTPUT_DIR}/final"
trainer.save_model(fd); tokenizer.save_pretrained(fd)
with open(f"{fd}/results.json", "w") as f:
    json.dump({"experiment": "exp026_gec_600m", "format": "thinking_single_tag",
               "epochs": 3, "lr": 1.5e-5, "train_min": round(tt/60,1),
               "examples": total_ex, "errors": error_count, "clean": clean_added,
               "params": "600M"}, f, indent=2)
print(f"Saved to {fd}")
PYEOF

# Train
tg "Training GEC 600M"
T0=$(date +%s)
PYTHONUNBUFFERED=1 BASE_MODEL=$BASE_MODEL HF_TOKEN=$HF_TOKEN \
    torchrun --nproc_per_node=$NUM_GPUS /root/gec_600m/train.py 2>&1 | tee /root/gec_600m/train.log
T1=$(date +%s)
tg "Training done in $(( (T1-T0)/60 ))min"

# Upload
tg "Uploading..."
python3 << UPLOADEOF
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast
from huggingface_hub import HfApi, create_repo
token = os.environ.get("HF_TOKEN")
repo = "$HF_REPO"
fd = "/root/gec_600m/output/final"
if not os.path.exists(fd):
    print("Output missing!"); exit(1)
create_repo(repo, token=token, exist_ok=True)
m = AutoModelForCausalLM.from_pretrained(fd)
m.push_to_hub(repo, token=token)
try:
    t = AutoTokenizer.from_pretrained(fd)
except:
    t = GPT2TokenizerFast(tokenizer_file=os.path.join(fd, "tokenizer.json"))
t.push_to_hub(repo, token=token)
HfApi().upload_file(path_or_fileobj=f"{fd}/results.json", path_in_repo="results.json", repo_id=repo, token=token)
print("Uploaded!")
UPLOADEOF

UPLOAD_OK=0
python3 -c "
import urllib.request, sys
try:
    r = urllib.request.urlopen('https://huggingface.co/api/models/$HF_REPO', timeout=30)
    if r.status == 200: print('HF OK'); sys.exit(0)
except: pass
sys.exit(1)
" && UPLOAD_OK=1

[ "$UPLOAD_OK" = "1" ] && tg "Uploaded: $HF_REPO" || tg "Upload FAILED"

T_END=$(date +%s)
tg "Total: $(( (T_END-T0)/60 ))min"

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
