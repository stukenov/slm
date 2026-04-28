#!/usr/bin/env python3
"""exp032: Stages 5-6-7 in one script. No dataset reload between stages.

Optimizations vs per-stage script:
- Dataset loaded and tokenized ONCE, reused across 3 stages
- Batch size doubled (8 vs 4) — H100 80GB has headroom
- No Python restart between stages
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import torch
import wandb

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

STAGES = {
    5: {"desc": "Unfreeze top 50pct", "lr": 5e-5, "max_tokens": 500_000_000, "warmup": 500},
    6: {"desc": "Full fine-tuning", "lr": 3e-5, "max_tokens": 500_000_000, "warmup": 500},
    7: {"desc": "Cooldown freeze embeddings", "lr": 2e-5, "max_tokens": 200_000_000, "warmup": 200},
}

PROMPTS = {
    "kaz_1": "Қазақстан — бұл", "kaz_2": "Бүгін ауа райы", "kaz_3": "Білім — ол",
    "rus_1": "Казахстан — это", "rus_2": "Сегодня погода", "rus_3": "Образование — это",
}


def freeze_for_stage(model, stage, num_layers):
    if stage == 5:
        half = num_layers // 2
        for name, p in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                p.requires_grad = True
            elif "layers." in name:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                p.requires_grad = layer_idx >= half
            else:
                p.requires_grad = True
    elif stage == 6:
        for p in model.parameters():
            p.requires_grad = True
    elif stage == 7:
        for name, p in model.named_parameters():
            p.requires_grad = "embed_tokens" not in name and "lm_head" not in name


def log_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Params: %d total, %d trainable (%.1f%%)", total, train, 100 * train / total)
    return total, train


def gen_samples(model, tokenizer, device):
    model.eval()
    results = {}
    for name, prompt in PROMPTS.items():
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.2)
        results[name] = tokenizer.decode(out[0], skip_special_tokens=True)
        log.info("  [%s] %s", name, results[name][:200])
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", default="kz-transformers/multidomain-kazakh-dataset")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="exp032-kazakh-adapt")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    )

    # === Load tokenizer ONCE ===
    log.info("Loading tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Load + tokenize dataset ONCE ===
    log.info("Loading dataset ONCE for all stages...")
    t0 = time.time()
    ds = load_dataset(args.dataset, split="train")
    ds = ds.shuffle(seed=args.seed)
    split = ds.train_test_split(test_size=10000, seed=args.seed)

    def tok_fn(examples):
        texts = [t if isinstance(t, str) else "" for t in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=args.block_size, padding=False)

    train_tok = split["train"].map(tok_fn, batched=True, num_proc=8,
                                    remove_columns=split["train"].column_names)
    val_tok = split["test"].map(tok_fn, batched=True, num_proc=4,
                                 remove_columns=split["test"].column_names)
    log.info("Dataset ready in %.0fs. Train: %d, Val: %d", time.time() - t0, len(train_tok), len(val_tok))

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load model ONCE ===
    log.info("Loading model from %s", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    num_layers = model.config.num_hidden_layers
    model = model.to(device)
    log.info("Model on %s, %d layers", device, num_layers)

    # === Run stages 5, 6, 7 ===
    for stage in [5, 6, 7]:
        cfg = STAGES[stage]
        out_dir = f"/root/exp032_stage{stage}_out"

        log.info("=" * 60)
        log.info("STAGE %d: %s (LR=%s, %dM tokens)", stage, cfg["desc"], cfg["lr"], cfg["max_tokens"] // 1_000_000)
        log.info("=" * 60)

        freeze_for_stage(model, stage, num_layers)
        total_p, train_p = log_params(model)

        wandb.init(project=args.wandb_project,
                   name=f"stage{stage}-{cfg['desc'].replace(' ', '_')}",
                   config={"stage": stage, "lr": cfg["lr"], "max_tokens": cfg["max_tokens"],
                           "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                           "total_params": total_p, "trainable_params": train_p})

        max_steps = cfg["max_tokens"] // (args.block_size * args.batch_size * args.grad_accum)
        log.info("Max steps: %d", max_steps)

        log.info("Baseline gen:")
        bl = gen_samples(model, tokenizer, device)
        wandb.log({"baseline_gen": wandb.Table(columns=["prompt", "text"],
                   data=[[k, v] for k, v in bl.items()])})

        ta = TrainingArguments(
            output_dir=out_dir, max_steps=max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=float(cfg["lr"]), lr_scheduler_type="cosine",
            warmup_steps=cfg["warmup"], weight_decay=0.1, max_grad_norm=1.0,
            bf16=True, logging_steps=10, save_steps=1000,
            eval_strategy="steps", eval_steps=1000,
            save_total_limit=2, report_to="wandb",
            seed=args.seed, dataloader_num_workers=4,
            remove_unused_columns=False)

        trainer = Trainer(model=model, args=ta, train_dataset=train_tok,
                         eval_dataset=val_tok, data_collator=collator)

        trainer.train()

        log.info("Final gen stage %d:", stage)
        fg = gen_samples(model, tokenizer, device)
        wandb.log({"final_gen": wandb.Table(columns=["prompt", "text"],
                   data=[[k, v] for k, v in fg.items()])})

        log.info("Saving stage %d...", stage)
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        with open(Path(out_dir) / "stage_meta.json", "w") as f:
            json.dump({"stage": stage, "desc": cfg["desc"], "max_steps": max_steps,
                        "lr": cfg["lr"], "total_params": total_p, "trainable_params": train_p}, f, indent=2)

        wandb.finish()
        log.info("STAGE %d COMPLETE.", stage)

        for ckpt in Path(out_dir).glob("checkpoint-*"):
            shutil.rmtree(ckpt)

    log.info("=" * 60)
    log.info("ALL STAGES 5-6-7 COMPLETE! Final model: /root/exp032_stage7_out")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
