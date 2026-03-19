#!/bin/bash
# ============================================================================
# exp021: SFT Pipeline — 300M Llama instruct on Kazakh instructions
# Downloads data -> Fine-tunes -> Uploads to HF -> Telegram -> Self-destruct
#
# Usage: bash run_sft.sh
# Required env vars: TG_TOKEN, TG_CHAT_ID, HF_TOKEN, VAST_API_KEY, VAST_INSTANCE_ID
# ============================================================================

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXPERIMENT="exp021_sft_300m"
HF_REPO="stukenov/sozkz-core-llama-300m-kk-instruct-v1"
BASE_MODEL="stukenov/sozkz-core-llama-300m-kk-base-v1"
TOKENIZER="stukenov/sozkz-core-gpt2-50k-kk-base-v1"
DATASET="AmanMussa/kazakh-instruction-v2"
WORKDIR="/root/sft"

# --- Telegram helper (injection-safe) ---
tg() {
    local msg="$1"
    if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
        TG_MSG="[$EXPERIMENT] $msg" python3 -c "
import os, urllib.request, urllib.parse
data = urllib.parse.urlencode({
    'chat_id': os.environ['TG_CHAT_ID'],
    'text': os.environ['TG_MSG'],
    'parse_mode': 'HTML'
}).encode()
try:
    urllib.request.urlopen('https://api.telegram.org/bot' + os.environ['TG_TOKEN'] + '/sendMessage', data, timeout=10)
except: pass
" 2>/dev/null || true
    fi
    echo "[TG] $msg"
}

# --- Step 0: Verify ---
tg "Started SFT on ${NUM_GPUS}x GPU"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
T0=$(date +%s)

# --- Step 1: Install deps ---
tg "Installing dependencies..."
pip install -q torch transformers datasets huggingface-hub safetensors accelerate 2>&1 | tail -3
tg "Dependencies installed"

# --- Step 2: SFT Training ---
tg "Starting SFT: 300M on ${DATASET}, ${NUM_GPUS} GPUs"
mkdir -p $WORKDIR

cat > $WORKDIR/train_sft.py << 'PYEOF'
"""SFT training for 300M Kazakh Llama — Alpaca format."""
import os, json, math, time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
)

# Config
BASE_MODEL = os.environ.get("BASE_MODEL", "stukenov/sozkz-core-llama-300m-kk-base-v1")
TOKENIZER = os.environ.get("TOKENIZER", "stukenov/sozkz-core-gpt2-50k-kk-base-v1")
DATASET = os.environ.get("DATASET", "AmanMussa/kazakh-instruction-v2")
OUTPUT_DIR = "/root/sft/output"
MAX_LENGTH = 1024
NUM_EPOCHS = 3
LR = 2e-5

PROMPT_TEMPLATE = """\
### Нұсқаулық:
{instruction}

### Жауап:
"""

PROMPT_TEMPLATE_INPUT = """\
### Нұсқаулық:
{instruction}

### Кіріс:
{input}

### Жауап:
"""

# Load model + tokenizer
print(f"Loading model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Load dataset
print(f"Loading dataset: {DATASET}")
ds = load_dataset(DATASET)
train_ds = ds["train"]
print(f"Train: {len(train_ds)} examples")

# Data collator with prompt masking
class AlpacaCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            instruction = f.get("instruction", "")
            inp = f.get("input", "")
            output = f.get("output", "")
            if inp:
                prompt = PROMPT_TEMPLATE_INPUT.format(instruction=instruction, input=inp)
            else:
                prompt = PROMPT_TEMPLATE.format(instruction=instruction)
            full = prompt + output + self.tokenizer.eos_token

            enc = self.tokenizer(full, truncation=True, max_length=self.max_length, add_special_tokens=False)
            ids = enc["input_ids"]
            mask = enc["attention_mask"]

            prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, add_special_tokens=False)
            prompt_len = len(prompt_enc["input_ids"])
            labels = [-100] * prompt_len + ids[prompt_len:]

            batch_ids.append(ids)
            batch_mask.append(mask)
            batch_labels.append(labels)

        max_len = max(len(x) for x in batch_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        for i in range(len(batch_ids)):
            pad = max_len - len(batch_ids[i])
            batch_ids[i] += [pad_id] * pad
            batch_mask[i] += [0] * pad
            batch_labels[i] += [-100] * pad

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

collator = AlpacaCollator(tokenizer, MAX_LENGTH)

num_gpus = torch.cuda.device_count()
# Effective batch = per_device * num_gpus * grad_accum = 8 * num_gpus * (64 // (8 * num_gpus))
per_device_bs = 8
target_batch = 64
grad_accum = max(1, target_batch // (per_device_bs * num_gpus))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    learning_rate=LR,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=collator,
    processing_class=tokenizer,
)

print(f"Training: {NUM_EPOCHS} epochs, BS={per_device_bs}x{num_gpus}GPUx{grad_accum}accum={per_device_bs*num_gpus*grad_accum}")
t0 = time.time()
trainer.train()
train_time = time.time() - t0
print(f"Training done: {train_time/60:.1f}min")

# Save final
final_dir = f"{OUTPUT_DIR}/final"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

# Save results
results = {
    "train_time_min": round(train_time / 60, 1),
    "num_epochs": NUM_EPOCHS,
    "dataset": DATASET,
    "dataset_size": len(train_ds),
    "base_model": BASE_MODEL,
}
with open(f"{final_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {final_dir}")
PYEOF

PYTHONUNBUFFERED=1 BASE_MODEL=$BASE_MODEL TOKENIZER=$TOKENIZER DATASET=$DATASET \
    torchrun --nproc_per_node=$NUM_GPUS $WORKDIR/train_sft.py 2>&1 | tee $WORKDIR/train.log
TRAIN_EXIT=${PIPESTATUS[0]}

if [ "$TRAIN_EXIT" -ne 0 ]; then
    tg "ERROR: SFT training FAILED (exit $TRAIN_EXIT). Instance kept alive!"
    exit 1
fi

if [ ! -d "/root/sft/output/final" ]; then
    tg "ERROR: Training exited OK but output missing! Instance kept alive!"
    exit 1
fi

T1=$(date +%s)
TRAIN_MIN=$(( (T1 - T0) / 60 ))
tg "SFT done in ${TRAIN_MIN}min"

# --- Step 3: Upload to HuggingFace ---
tg "Uploading to HF: ${HF_REPO}"
UPLOAD_OK=0

python3 << UPLOADEOF
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo

repo = "${HF_REPO}"
token = os.environ.get("HF_TOKEN")
final_dir = "/root/sft/output/final"

print(f"Loading model from {final_dir}...")
model = AutoModelForCausalLM.from_pretrained(final_dir)
tokenizer = AutoTokenizer.from_pretrained(final_dir)

print(f"Uploading to {repo}...")
create_repo(repo, token=token, exist_ok=True)
model.push_to_hub(repo, token=token)
tokenizer.push_to_hub(repo, token=token)

# Upload results
import json
api = HfApi()
api.upload_file(
    path_or_fileobj=f"{final_dir}/results.json",
    path_in_repo="results.json",
    repo_id=repo,
    token=token,
)
print(f"Done! https://huggingface.co/{repo}")
UPLOADEOF

if [ $? -eq 0 ]; then
    # Verify repo exists
    HTTP_CODE=$(python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('https://huggingface.co/api/models/${HF_REPO}', timeout=15)
    print(r.getcode())
except Exception as e:
    print(getattr(e, 'code', 0))
")
    if [ "$HTTP_CODE" = "200" ]; then
        UPLOAD_OK=1
        tg "Model uploaded: https://huggingface.co/${HF_REPO}"
    else
        tg "ERROR: Upload OK but repo not found (HTTP $HTTP_CODE). Instance kept alive!"
    fi
else
    tg "ERROR: Upload failed. Instance kept alive!"
fi

# --- Step 4: Self-destruct (ONLY if upload confirmed) ---
if [ "$UPLOAD_OK" = "1" ]; then
    T_END=$(date +%s)
    TOTAL_MIN=$(( (T_END - T0) / 60 ))
    HOURLY_RATE=${HOURLY_RATE:-1.34}
    TOTAL_COST=$(python3 -c "print(f'{($T_END-$T0)/3600*$HOURLY_RATE:.2f}')")
    tg "Total: ${TOTAL_MIN}min, ~USD${TOTAL_COST}"
    tg "Self-destructing instance..."
    if [ -n "$VAST_API_KEY" ] && [ -n "$VAST_INSTANCE_ID" ]; then
        python3 -c "
import urllib.request
req = urllib.request.Request(
    'https://console.vast.ai/api/v0/instances/$VAST_INSTANCE_ID/',
    method='DELETE',
    headers={'Authorization': 'Bearer $VAST_API_KEY'}
)
try:
    urllib.request.urlopen(req, timeout=30)
    print('Instance destroyed')
except Exception as e:
    print(f'Destroy failed: {e}')
"
        tg "Instance destroyed. SFT pipeline complete!"
    fi
else
    tg "Training OK but upload FAILED. Instance kept alive!"
fi
