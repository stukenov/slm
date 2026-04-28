"""LoRA SFT for Qwen 500M -> Kazakh GEC v3.
Expanded dataset: 14,597 examples (original 9599 + 3000 emle + 2000 morph/punct).
"""
import os, json, time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import hf_hub_download, HfApi, create_repo

BASE_REPO = "stukenov/sozkz-core-qwen-500m-kk-base-v1"
OUTPUT_REPO = "stukenov/sozkz-fix-qwen-500m-kk-gec-v3"
DATA_PATH = "/root/gec_combined_v3.jsonl"

INSTRUCTION = (
    "\u041c\u04d9\u0442\u0456\u043d\u0434\u0435\u0433\u0456 \u0433\u0440\u0430\u043c\u043c\u0430\u0442\u0438\u043a\u0430\u043b\u044b\u049b, \u043e\u0440\u0444\u043e\u0433\u0440\u0430\u0444\u0438\u044f\u043b\u044b\u049b, \u043f\u0443\u043d\u043a\u0442\u0443\u0430\u0446\u0438\u044f\u043b\u044b\u049b \u0436\u04d9\u043d\u0435 \u0441\u04e9\u0437 \u049b\u043e\u043b\u0434\u0430\u043d\u044b\u0441\u044b\u043d\u0434\u0430\u0493\u044b "
    "\u049b\u0430\u0442\u0435\u043b\u0435\u0440\u0434\u0456 \u0442\u04af\u0437\u0435\u0442. \u041c\u0430\u0493\u044b\u043d\u0430\u043d\u044b \u04e9\u0437\u0433\u0435\u0440\u0442\u043f\u0435. \u0415\u0433\u0435\u0440 \u043c\u04d9\u0442\u0456\u043d \u0434\u04b1\u0440\u044b\u0441 \u0431\u043e\u043b\u0441\u0430, \u043e\u043d\u044b \u04e9\u0437\u0433\u0435\u0440\u0456\u0441\u0441\u0456\u0437 \u049b\u0430\u0439\u0442\u0430\u0440. "
    "\u0422\u0435\u043a \u0442\u04af\u0437\u0435\u0442\u0456\u043b\u0433\u0435\u043d \u043c\u04d9\u0442\u0456\u043d\u0434\u0456 \u049b\u0430\u0439\u0442\u0430\u0440."
)


def send_tg(msg):
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({"chat_id": "47474471", "text": msg}).encode()
        urllib.request.urlopen(
            "https://api.telegram.org/bot8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g/sendMessage",
            data, timeout=10,
        )
    except Exception:
        pass


def format_gec(example):
    inp = example.get("input", "")
    out = example.get("output", "")
    return {"text": f"### \u041d\u04b1\u0441\u049b\u0430\u0443:\n{INSTRUCTION}\n\n### \u041c\u04d9\u0442\u0456\u043d:\n{inp}\n\n### \u0422\u04af\u0437\u0435\u0442\u0456\u043b\u0433\u0435\u043d:\n{out}"}


def main():
    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    send_tg("[GEC v3] Starting training: 14.6K examples, LoRA r=64")

    print(f"Loading {BASE_REPO}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO, dtype=torch.bfloat16, device_map="auto", token=token,
    )
    tok_file = hf_hub_download(BASE_REPO, "tokenizer.json", token=token)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1
    tokenizer.padding_side = "right"
    tokenizer.eos_token = "<pad>"
    tokenizer.eos_token_id = 1

    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading {DATA_PATH}...")
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    ds = Dataset.from_list(rows)
    ds = ds.filter(lambda x: x.get("input") and x.get("output"))
    ds = ds.map(format_gec, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    t0 = time.time()
    batch_size, grad_accum = 8, 4
    effective_batch = batch_size * grad_accum
    total_steps = (len(train_ds) // effective_batch) * 3

    training_args = SFTConfig(
        output_dir="/tmp/gec_qwen500m_v3",
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(total_steps // 4, 50),
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=max(total_steps // 8, 50),
        dataset_text_field="text",
        max_length=512,
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    print(f"Training: {total_steps} steps, effective_batch={effective_batch}")
    trainer.train()
    train_time = time.time() - t0
    final_metrics = trainer.evaluate()
    eval_loss = final_metrics.get("eval_loss", "?")
    print(f"Done in {train_time/60:.1f}min, eval_loss={eval_loss}")

    print("Merging LoRA...")
    model = model.merge_and_unload()

    # Smoke test
    model.train(False)
    test_cases = [
        ("\u041c\u0435\u043d \u0431\u0443\u0433\u0456\u043d \u043c\u0435\u043a\u0442\u0435\u043f\u043a\u0435 \u0431\u0430\u0440\u0434\u044b\u043c",
         "\u041c\u0435\u043d \u0431\u04af\u0433\u0456\u043d \u043c\u0435\u043a\u0442\u0435\u043f\u043a\u0435 \u0431\u0430\u0440\u0434\u044b\u043c."),
        ("\u041e\u043b \u043a\u0456\u0442\u0430\u043f\u0442\u044b \u043e\u049b\u044b\u0434\u044b \u0436\u0430\u043d\u0435 \u043c\u0430\u0493\u0430\u043d \u0430\u0439\u0442\u0442\u044b",
         "\u041e\u043b \u043a\u0456\u0442\u0430\u043f\u0442\u044b \u043e\u049b\u044b\u0434\u044b \u0436\u04d9\u043d\u0435 \u043c\u0430\u0493\u0430\u043d \u0430\u0439\u0442\u0442\u044b."),
        ("\u0411\u0430\u043b\u0430 \u0431\u0430\u043a\u0448\u0430\u0433\u0430 \u0431\u0430\u0440\u0430\u0434\u044b",
         "\u0411\u0430\u043b\u0430 \u0431\u0430\u049b\u0448\u0430\u0493\u0430 \u0431\u0430\u0440\u0430\u0434\u044b."),
        ("\u049a\u0430\u0437\u0430\u049b\u0441\u0442\u0430\u043d \u041e\u0440\u0442\u0430\u043b\u044b\u049b \u0410\u0437\u0438\u044f\u0434\u0430\u0493\u044b \u0435\u04a3 \u0456\u0440\u0456 \u043c\u0435\u043c\u043b\u0435\u043a\u0435\u0442 \u0431\u043e\u043b\u0430\u043f \u0442\u0430\u0431\u044b\u043b\u0430\u0434\u044b.",
         "\u049a\u0430\u0437\u0430\u049b\u0441\u0442\u0430\u043d \u041e\u0440\u0442\u0430\u043b\u044b\u049b \u0410\u0437\u0438\u044f\u0434\u0430\u0493\u044b \u0435\u04a3 \u0456\u0440\u0456 \u043c\u0435\u043c\u043b\u0435\u043a\u0435\u0442 \u0431\u043e\u043b\u044b\u043f \u0442\u0430\u0431\u044b\u043b\u0430\u0434\u044b."),
        ("\u041c\u0435\u043d\u0456\u04a3 \u0434\u043e\u0441\u044b\u043c \u043a\u0435\u043b\u0434\u0456",
         "\u041c\u0435\u043d\u0456\u04a3 \u0434\u043e\u0441\u044b\u043c \u043a\u0435\u043b\u0434\u0456."),
    ]
    correct_count = 0
    for inp, exp in test_cases:
        prompt = f"### \u041d\u04b1\u0441\u049b\u0430\u0443:\n{INSTRUCTION}\n\n### \u041c\u04d9\u0442\u0456\u043d:\n{inp}\n\n### \u0422\u04af\u0437\u0435\u0442\u0456\u043b\u0433\u0435\u043d:\n"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=min(ids.shape[1], 512),
                do_sample=False, repetition_penalty=1.0, pad_token_id=1,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        corrected = decoded.split("### \u0422\u04af\u0437\u0435\u0442\u0456\u043b\u0433\u0435\u043d:\n")[-1].split("###")[0].strip()
        ok = corrected.strip().rstrip(".") == exp.strip().rstrip(".")
        if ok:
            correct_count += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}: {inp[:40]}... -> {corrected}")
    score = f"{correct_count}/{len(test_cases)}"
    print(f"Smoke test: {score}")

    print(f"Publishing to {OUTPUT_REPO}...")
    create_repo(OUTPUT_REPO, token=token, exist_ok=True)
    model.push_to_hub(OUTPUT_REPO, token=token)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=tok_file, path_in_repo="tokenizer.json",
        repo_id=OUTPUT_REPO, token=token,
    )

    results = {
        "base_model": BASE_REPO, "version": "v3", "task": "kazakh-gec",
        "dataset_size": len(train_ds) + len(eval_ds),
        "train_size": len(train_ds), "eval_size": len(eval_ds),
        "eval_loss": final_metrics.get("eval_loss"),
        "training_minutes": round(train_time / 60, 1),
        "smoke_test_score": score,
    }
    with open("/tmp/gec_results_v3.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    api.upload_file(
        path_or_fileobj="/tmp/gec_results_v3.json", path_in_repo="gec_results.json",
        repo_id=OUTPUT_REPO, token=token,
    )

    print(f"\nDone! https://huggingface.co/{OUTPUT_REPO}")
    send_tg(
        f"[GEC v3] DONE! https://huggingface.co/{OUTPUT_REPO}\n"
        f"Time: {train_time/60:.1f}min, data: {len(train_ds)+len(eval_ds)}\n"
        f"Eval loss: {eval_loss}\nSmoke: {score}"
    )


if __name__ == "__main__":
    main()
