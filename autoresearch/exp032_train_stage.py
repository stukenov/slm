#!/usr/bin/env python3
"""exp032: 7-Stage continual pretraining for Kazakh language adaptation.

Usage:
    python exp032_train_stage.py --stage 1 --model-dir /root/exp032_extended_model_v2
    python exp032_train_stage.py --stage 2 --model-dir /root/exp032_stage1_out
    ...
    python exp032_train_stage.py --stage 7 --model-dir /root/exp032_stage6_out

Each stage freezes/unfreezes different parameters per the EEVE-inspired schedule.
All stages log to W&B project "exp032-kazakh-adapt".
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import wandb

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Stage configs
STAGE_CONFIGS = {
    1: {
        "desc": "Input embeddings only",
        "trainable": ["embed_tokens"],
        "frozen": "all except embed_tokens",
        "lr": 2e-4,
        "max_tokens": 100_000_000,
        "warmup_steps": 500,
        "use_lora": False,
    },
    2: {
        "desc": "Input + output embeddings",
        "trainable": ["embed_tokens", "lm_head"],
        "frozen": "transformer layers",
        "lr": 2e-4,
        "max_tokens": 200_000_000,
        "warmup_steps": 500,
        "use_lora": False,
    },
    3: {
        "desc": "Embeddings + LoRA QKV r=16",
        "trainable": ["embed_tokens", "lm_head"],
        "frozen": "MLP, LayerNorm (LoRA on Q,K,V)",
        "lr": 1e-4,
        "max_tokens": 500_000_000,
        "warmup_steps": 1000,
        "use_lora": True,
        "lora_target": ["q_proj", "k_proj", "v_proj"],
        "lora_r": 16,
    },
    4: {
        "desc": "Embeddings + LoRA QKVO + MLP",
        "trainable": ["embed_tokens", "lm_head"],
        "frozen": "LayerNorm only",
        "lr": 1e-4,
        "max_tokens": 500_000_000,
        "warmup_steps": 1000,
        "use_lora": True,
        "lora_target": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 16,
    },
    5: {
        "desc": "Merge LoRA, unfreeze top 50pct",
        "trainable": "top_half",
        "frozen": "bottom 50% layers",
        "lr": 5e-5,
        "max_tokens": 500_000_000,
        "warmup_steps": 500,
        "use_lora": False,
    },
    6: {
        "desc": "Full fine-tuning",
        "trainable": "all",
        "frozen": "nothing",
        "lr": 3e-5,
        "max_tokens": 500_000_000,
        "warmup_steps": 500,
        "use_lora": False,
    },
    7: {
        "desc": "Cooldown freeze embeddings",
        "trainable": "transformer_only",
        "frozen": "embed_tokens + lm_head",
        "lr": 2e-5,
        "max_tokens": 200_000_000,
        "warmup_steps": 200,
        "use_lora": False,
    },
}

PROMPTS = {
    "kaz_1": "Қазақстан — бұл",
    "kaz_2": "Бүгін ауа райы",
    "kaz_3": "Білім — ол",
    "rus_1": "Казахстан — это",
    "rus_2": "Сегодня погода",
    "rus_3": "Образование — это",
}


def compute_max_steps(max_tokens, block_size, batch_size, grad_accum):
    tokens_per_step = block_size * batch_size * grad_accum
    return max_tokens // tokens_per_step


def freeze_params(model, stage_config, num_layers):
    trainable = stage_config["trainable"]

    if trainable == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    if trainable == "transformer_only":
        for name, p in model.named_parameters():
            p.requires_grad = "embed_tokens" not in name and "lm_head" not in name
        return

    if trainable == "top_half":
        half = num_layers // 2
        for name, p in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                p.requires_grad = True
            elif "layers." in name:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                p.requires_grad = layer_idx >= half
            else:
                p.requires_grad = True
        return

    # List of substrings to keep trainable
    for name, p in model.named_parameters():
        p.requires_grad = any(t in name for t in trainable)


def log_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Params: %d total, %d trainable (%.2f%%)",
             total, trainable, 100 * trainable / total)
    return total, trainable


def generate_samples(model, tokenizer, device):
    model_for_gen = model
    if hasattr(model, 'base_model') and hasattr(model, 'peft_config'):
        model_for_gen = model
    was_training = model_for_gen.training
    model_for_gen.eval()
    results = {}
    for name, prompt in PROMPTS.items():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model_for_gen.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results[name] = text
        log.info("  [%s] %s", name, text[:200])
    if was_training:
        model_for_gen.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset", default="kz-transformers/multidomain-kazakh-dataset")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--wandb-project", default="exp032-kazakh-adapt")
    args = parser.parse_args()

    stage = args.stage
    cfg = STAGE_CONFIGS[stage]
    output_dir = args.output_dir or f"/root/exp032_stage{stage}_out"

    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    log.info("=" * 60)
    log.info("STAGE %d: %s", stage, cfg["desc"])
    log.info("Model: %s", args.model_dir)
    log.info("LR: %s, Max tokens: %dM, Warmup: %d",
             cfg["lr"], cfg["max_tokens"] // 1_000_000, cfg["warmup_steps"])
    log.info("=" * 60)

    # W&B
    wandb.init(
        project=args.wandb_project,
        name=f"stage{stage}-{cfg['desc'].replace(' ', '_')}",
        config={
            "stage": stage,
            "description": cfg["desc"],
            "lr": cfg["lr"],
            "max_tokens": cfg["max_tokens"],
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "use_lora": cfg["use_lora"],
        },
    )

    # Load tokenizer + model
    log.info("Loading tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("Tokenizer vocab: %d", len(tokenizer))

    log.info("Loading model from %s", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16
    )
    num_layers = model.config.num_hidden_layers
    log.info("Model layers: %d", num_layers)

    # LoRA or freeze
    if cfg["use_lora"]:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_r"] * 2,
            lora_dropout=0.05,
            target_modules=cfg["lora_target"],
        )
        model = get_peft_model(model, lora_config)
        log.info("LoRA: target=%s, r=%d", cfg["lora_target"], cfg["lora_r"])
        for name, p in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                p.requires_grad = True
    elif stage == 5:
        # Merge LoRA if present
        try:
            from peft import PeftModel
            if hasattr(model, 'merge_and_unload'):
                model = model.merge_and_unload()
                log.info("LoRA merged into base weights")
        except Exception:
            log.info("No LoRA to merge")
        freeze_params(model, cfg, num_layers)
    else:
        freeze_params(model, cfg, num_layers)

    total_params, trainable_params = log_trainable_params(model)
    wandb.config.update({"total_params": total_params, "trainable_params": trainable_params})

    # Load dataset
    log.info("Loading dataset: %s", args.dataset)
    ds = load_dataset(args.dataset, split="train")
    ds = ds.shuffle(seed=args.seed)

    split = ds.train_test_split(test_size=10000, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    def tokenize_fn(examples):
        texts = [t if isinstance(t, str) else "" for t in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=args.block_size, padding=False)

    log.info("Tokenizing train set...")
    train_tok = train_ds.map(tokenize_fn, batched=True, num_proc=8,
                              remove_columns=train_ds.column_names)
    log.info("Tokenizing val set...")
    val_tok = val_ds.map(tokenize_fn, batched=True, num_proc=4,
                          remove_columns=val_ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    max_steps = compute_max_steps(
        cfg["max_tokens"], args.block_size, args.batch_size, args.grad_accum
    )
    log.info("Max steps: %d (for %dM tokens)", max_steps, cfg["max_tokens"] // 1_000_000)

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=float(cfg["lr"]),
        lr_scheduler_type="cosine",
        warmup_steps=cfg["warmup_steps"],
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=3,
        report_to="wandb",
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )

    # Baseline generation
    log.info("Generating baseline samples...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline = generate_samples(model, tokenizer, device)
    wandb.log({"baseline_generations": wandb.Table(
        columns=["prompt", "text"],
        data=[[k, v] for k, v in baseline.items()]
    )})

    log.info("Starting training...")
    trainer.train()

    # Final generation
    log.info("Generating final samples...")
    final_gen = generate_samples(model, tokenizer, device)
    wandb.log({"final_generations": wandb.Table(
        columns=["prompt", "text"],
        data=[[k, v] for k, v in final_gen.items()]
    )})

    # Save
    log.info("Saving to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "stage": stage,
        "description": cfg["desc"],
        "model_dir_input": args.model_dir,
        "output_dir": output_dir,
        "max_steps": max_steps,
        "lr": cfg["lr"],
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    with open(Path(output_dir) / "stage_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    wandb.finish()
    log.info("STAGE %d COMPLETE.", stage)


if __name__ == "__main__":
    main()
