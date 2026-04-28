#!/usr/bin/env python3
"""exp030: GEC 1B — unified training script (LoRA / full fine-tune).

Format: {input}\n{target}<eos>  — loss masked on input line.

All config via environment variables:
  EXP_ID        — experiment id (e.g. "030a")
  BASE_MODEL    — HF model id (default: stukenov/sozkz-core-llama-1b-kk-base-v1)
  METHOD        — "lora" or "full"
  LORA_RANK     — LoRA rank (default: 16)
  LORA_MODULES  — "qv" (q_proj,v_proj) or "all" (all linear)
  LR            — learning rate (default: 2e-4 for LoRA, 1e-5 for full)
  CLEAN_RATIO   — ratio of clean examples (default: 0.8)
  EPOCHS        — num epochs (default: 3)
  MAX_STEPS     — override max steps (-1 = use epochs)
  BATCH_SIZE    — per device batch size (default: 8)
  MAX_SEQ_LEN   — max sequence length (default: 512)
  OUTPUT_DIR    — output directory (default: /root/exp030_{EXP_ID})
  HF_TOKEN      — HuggingFace token
  TG_TOKEN      — Telegram bot token (optional)
  TG_CHAT_ID    — Telegram chat id (optional)
"""

import os
import json
import time
import random
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from huggingface_hub import hf_hub_download
from datasets import Dataset

# ── Config from env ──────────────────────────────────────────────────────────

EXP_ID = os.environ.get("EXP_ID", "030x")
BASE_MODEL = os.environ.get("BASE_MODEL", "stukenov/sozkz-core-llama-1b-kk-base-v1")
METHOD = os.environ.get("METHOD", "lora")
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
LORA_MODULES = os.environ.get("LORA_MODULES", "qv")
LR = float(os.environ.get("LR", "2e-4" if METHOD == "lora" else "1e-5"))
CLEAN_RATIO = float(os.environ.get("CLEAN_RATIO", "0.8"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"/root/exp030_{EXP_ID}")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Telegram helper ──────────────────────────────────────────────────────────

def tg(msg):
    tg_token = os.environ.get("TG_TOKEN", "")
    tg_chat = os.environ.get("TG_CHAT_ID", "")
    full = f"[exp030/{EXP_ID}] {msg}"
    print(f"[TG] {full}")
    if tg_token and tg_chat:
        import urllib.request, urllib.parse
        try:
            data = urllib.parse.urlencode({"chat_id": tg_chat, "text": full}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{tg_token}/sendMessage", data, timeout=10)
        except Exception:
            pass

# ── Data loading ─────────────────────────────────────────────────────────────

def word_edit_distance(a, b):
    wa, wb = a.split(), b.split()
    if len(wa) < len(wb):
        return word_edit_distance(b, a)
    if not wb:
        return len(wa)
    prev = list(range(len(wb) + 1))
    for i, ca in enumerate(wa):
        curr = [i + 1]
        for j, cb in enumerate(wb):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def load_gec_data():
    """Load GEC dataset, filter, add clean examples."""
    REPO = "stukenov/sozkz-corpus-synthetic-kk-gec-v1"
    FILES = [
        "data/grammar_balanced_v2/train.jsonl",
        "data/grammar_v2/train.jsonl",
        "data/grammar_combined/train.jsonl",
        "data/grammar_focused/train.jsonl",
        "data/processed/train.jsonl",
        "data/processed_v2/train.jsonl",
        "data/processed_v3/train.jsonl",
    ]

    errors = []
    total = skipped = 0

    for fname in FILES:
        try:
            local = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset", token=HF_TOKEN)
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ex = json.loads(line)
                    total += 1
                    et = ex.get("error_type", "unknown")
                    inp, tgt = ex.get("input", ""), ex.get("target", "")
                    if not inp or not tgt:
                        continue
                    if et in ("unknown", "clean"):
                        continue
                    if word_edit_distance(inp, tgt) > 2:
                        skipped += 1
                        continue
                    if inp.strip() == tgt.strip():
                        continue
                    errors.append({"input": inp, "target": tgt})
            print(f"  {fname}: loaded")
        except Exception as e:
            print(f"  {fname}: SKIP ({e})")

    print(f"Total raw: {total}, skipped (edit>2): {skipped}, errors: {len(errors)}")

    # Build dataset: errors + clean examples
    examples = [{"input": e["input"], "target": e["target"]} for e in errors]
    error_count = len(examples)

    # Add clean examples (target == target)
    if CLEAN_RATIO > 0:
        clean_count = int(error_count * CLEAN_RATIO / (1 - CLEAN_RATIO))
        clean_added = 0
        for e in errors:
            if clean_added >= clean_count:
                break
            examples.append({"input": e["target"], "target": e["target"]})
            clean_added += 1
        print(f"Added {clean_added} clean examples ({clean_added * 100 // len(examples)}%)")

    random.seed(42)
    random.shuffle(examples)

    # Split: 99% train, 1% eval
    split_idx = max(1, int(len(examples) * 0.99))
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]

    print(f"Dataset: {len(train_examples)} train, {len(eval_examples)} eval "
          f"({error_count} errors, {len(examples) - error_count} clean)")

    return train_examples, eval_examples


# ── Collator ─────────────────────────────────────────────────────────────────

class TwoLineCollator:
    """Format: {input}\n{target}<eos>. Loss masked before \n (inclusive)."""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-encode \n to find its token id
        nl_enc = tokenizer.encode("\n", add_special_tokens=False)
        self.nl_token_id = nl_enc[-1] if nl_enc else None

    def __call__(self, features):
        batch_ids, batch_mask, batch_labels = [], [], []

        for f in features:
            full_text = f["input"] + "\n" + f["target"] + self.tokenizer.eos_token
            enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length,
                                 add_special_tokens=False)
            ids = enc["input_ids"]

            # Find first \n token — mask everything up to and including it
            nl_pos = -1
            if self.nl_token_id is not None:
                for i, tid in enumerate(ids):
                    if tid == self.nl_token_id:
                        nl_pos = i
                        break

            labels = list(ids)
            # Mask input portion (up to and including \n)
            mask_end = nl_pos + 1 if nl_pos >= 0 else 0
            for i in range(mask_end):
                labels[i] = -100

            batch_ids.append(ids)
            batch_mask.append(enc["attention_mask"])
            batch_labels.append(labels)

        # Pad to max in batch
        mx = max(len(x) for x in batch_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        for i in range(len(batch_ids)):
            p = mx - len(batch_ids[i])
            batch_ids[i] += [pad_id] * p
            batch_mask[i] += [0] * p
            batch_labels[i] += [-100] * p

        return {
            "input_ids": torch.tensor(batch_ids),
            "attention_mask": torch.tensor(batch_mask),
            "labels": torch.tensor(batch_labels),
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = {
        "exp_id": EXP_ID, "base_model": BASE_MODEL, "method": METHOD,
        "lora_rank": LORA_RANK, "lora_modules": LORA_MODULES, "lr": LR,
        "clean_ratio": CLEAN_RATIO, "epochs": EPOCHS, "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE, "max_seq_len": MAX_SEQ_LEN,
    }
    print(f"\n{'='*60}")
    print(f"exp030/{EXP_ID}: GEC 1B — {METHOD.upper()}")
    print(json.dumps(config, indent=2))
    print(f"{'='*60}\n")

    tg(f"Started: {METHOD.upper()}, LR={LR}, clean={CLEAN_RATIO}, epochs={EPOCHS}"
       + (f", rank={LORA_RANK}, modules={LORA_MODULES}" if METHOD == "lora" else ""))

    # Load data
    print("Loading GEC data...")
    train_data, eval_data = load_gec_data()

    # Load tokenizer & model
    print(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

    # Ensure EOS and PAD tokens exist
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, token=HF_TOKEN
    )

    # Resize embeddings if we added tokens
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings to {len(tokenizer)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params / 1e6:.1f}M")

    # Apply LoRA if needed
    if METHOD == "lora":
        from peft import LoraConfig, get_peft_model, TaskType

        if LORA_MODULES == "all":
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        else:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA: rank={LORA_RANK}, modules={target_modules}")
        print(f"Trainable: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M "
              f"({trainable * 100 / total_params:.2f}%)")

    # Build datasets
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)
    collator = TwoLineCollator(tokenizer, max_length=MAX_SEQ_LEN)

    # Gradient accumulation to hit effective BS ~128
    num_gpus = max(1, torch.cuda.device_count()) if METHOD == "full" else 1
    ga = max(1, 128 // (BATCH_SIZE * num_gpus))

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ga,
        learning_rate=LR,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, processing_class=tokenizer,
    )

    eff_bs = BATCH_SIZE * num_gpus * ga
    print(f"Training: {EPOCHS}ep, LR={LR}, BS={BATCH_SIZE}x{num_gpus}x{ga}={eff_bs}")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"Training done: {train_time / 60:.1f}min")

    # Save
    final_dir = f"{OUTPUT_DIR}/final"
    if METHOD == "lora":
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    else:
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

    # Eval loss
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", -1)
    print(f"Eval loss: {eval_loss:.4f}")

    # Save results
    results = {
        **config,
        "train_time_min": round(train_time / 60, 1),
        "eval_loss": round(eval_loss, 4),
        "train_examples": len(train_data),
        "eval_examples": len(eval_data),
    }
    with open(f"{final_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    tg(f"Done in {train_time / 60:.0f}min. Eval loss: {eval_loss:.4f}")
    print(f"\nSaved to {final_dir}")


if __name__ == "__main__":
    main()
