from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)

LANG_CODE = "kaz_Cyrl"


def load_gec_jsonl(path: str | Path) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("input") and item.get("target"):
                rows.append(item)
    return Dataset.from_list(rows)


@dataclass
class NLLBGECCollator:
    tokenizer: AutoTokenizer
    max_source_length: int = 256
    max_target_length: int = 256

    def __call__(self, features: list[dict]) -> dict:
        sources = [f["input"] for f in features]
        targets = [f["target"] for f in features]

        self.tokenizer.src_lang = LANG_CODE
        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        self.tokenizer.tgt_lang = LANG_CODE
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        label_ids = labels["input_ids"]
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = label_ids
        return model_inputs


def train_nllb_gec(config: dict) -> Path:
    model_name = config.get("model_name", "facebook/nllb-200-distilled-600M")
    data_path = config["data_path"]
    output_dir = config.get("output_dir", "outputs/round1_nllb_baseline")

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    forced_bos_id = tokenizer.convert_tokens_to_ids(LANG_CODE)
    model.config.forced_bos_token_id = forced_bos_id

    logger.info("Loading data from %s", data_path)
    ds = load_gec_jsonl(data_path)
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    logger.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    collator = NLLBGECCollator(
        tokenizer=tokenizer,
        max_source_length=config.get("max_source_length", 256),
        max_target_length=config.get("max_target_length", 256),
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=float(config.get("learning_rate", 3e-5)),
        warmup_ratio=float(config.get("warmup_ratio", 0.05)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 50),
        save_steps=config.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=config.get("save_steps", 500),
        predict_with_generate=True,
        generation_max_length=config.get("max_target_length", 256),
        report_to=config.get("report_to", "none"),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to %s", output_dir)
    return Path(output_dir)


def generate_correction(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    text: str,
    num_beams: int = 5,
    max_length: int = 256,
) -> str:
    tokenizer.src_lang = LANG_CODE
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    forced_bos_id = tokenizer.convert_tokens_to_ids(LANG_CODE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            num_beams=num_beams,
            max_length=max_length,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
