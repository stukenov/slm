"""LoRA SFT for 447M Kazakh Qwen2.5 -> instruct version.
Base: stukenov/sozkz-core-qwen-500m-kk-base-v1
Data: stukenov/sozkz-corpus-instruct-kk-alpaca-qwen35-v1 (~4882 pairs)
Output: stukenov/sozkz-core-qwen-500m-kk-instruct-v1 (merged full model)
"""
import os, json, time, argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import hf_hub_download, HfApi, create_repo


BASE_REPO = "stukenov/sozkz-core-qwen-500m-kk-base-v1"
DATASET_REPO = "stukenov/sozkz-corpus-instruct-kk-alpaca-qwen35-v1"


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


def format_instruction(example):
    """Convert instruct_kk example to simple prompt format (compat with sft_1b sibling)."""
    instruction = example.get("instruction_kk", "") or ""
    inp = example.get("input_kk", "") or ""
    output = example.get("output_kk", "") or ""
    user_msg = instruction
    if inp:
        user_msg = f"{instruction}\n{inp}"
    return {"text": f"### Сұрақ:\n{user_msg}\n\n### Жауап:\n{output}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--repo", default="stukenov/sozkz-core-qwen-500m-kk-instruct-v1")
    parser.add_argument("--dry-run", action="store_true", help="Micro-train 5 steps and exit")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

    send_tg(f"[SFT-LoRA Qwen500M] Starting: LoRA r={args.lora_r} alpha={args.lora_alpha}, lr={args.lr}, {args.epochs} epochs")

    # --- Load base Qwen2 model ---
    print(f"Loading base model {BASE_REPO}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params, dtype={next(model.parameters()).dtype}")
    print(f"  Config: hidden={model.config.hidden_size} layers={model.config.num_hidden_layers} "
          f"heads={model.config.num_attention_heads} kv={model.config.num_key_value_heads} "
          f"vocab={model.config.vocab_size}")

    # --- Load tokenizer directly from tokenizer.json ---
    tok_file = hf_hub_download(BASE_REPO, "tokenizer.json", token=token)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1
    tokenizer.padding_side = "right"
    # SFTTrainer requires eos_token to append to training examples
    tokenizer.eos_token = "<pad>"
    tokenizer.eos_token_id = 1
    print(f"  Tokenizer vocab: {tokenizer.vocab_size}")

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    print(f"Loading dataset {DATASET_REPO}...")
    ds = load_dataset(DATASET_REPO, split="train")
    print(f"  Raw: {len(ds)} examples, columns={ds.column_names}")
    ds = ds.filter(lambda x: x.get("instruction_kk") and x.get("output_kk"))
    ds = ds.map(format_instruction, remove_columns=ds.column_names)
    print(f"  Formatted: {len(ds)} examples")
    print(f"  Sample text (first 400 chars):\n  {ds[0]['text'][:400]}")

    # --- Training ---
    t0 = time.time()

    total_steps = (len(ds) // (args.batch_size * args.grad_accum)) * args.epochs
    max_steps = 5 if args.dry_run else -1

    training_args = SFTConfig(
        output_dir="/tmp/sft_lora_qwen500m",
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        dataset_text_field="text",
        max_length=args.max_len,
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"Training: {total_steps} optim steps, effective_batch={args.batch_size * args.grad_accum}")
    trainer.train()
    total_time = time.time() - t0
    print(f"SFT-LoRA done in {total_time/60:.1f}min")

    if args.dry_run:
        print("DRY-RUN complete - exiting without merge/upload.")
        return

    # --- Merge LoRA weights back ---
    print("Merging LoRA into base...")
    model = model.merge_and_unload()

    # --- Inference smoke test ---
    print("\n=== Inference test ===")
    model.train(False)
    test_questions = [
        "Қазақстанның астанасы қай қала?",
        "Абай Құнанбаев туралы қысқаша айтып бер",
        "Жасанды интеллект дегеніміз не?",
        "Денсаулықты сақтау үшін не істеу керек?",
        "Python тілінде тізімді қалай сұрыптауға болады?",
    ]
    inference_results = []
    for q in test_questions:
        prompt = f"### Сұрақ:\n{q}\n\n### Жауап:\n"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=1,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        resp = text.split("### Жауап:\n")[-1].split("### Сұрақ:")[0].strip()
        print(f"\nQ: {q}")
        print(f"A: {resp[:400]}")
        inference_results.append(f"Q: {q}\nA: {resp[:400]}")

    # --- Publish merged model ---
    print(f"\nPublishing to {args.repo}...")
    create_repo(args.repo, token=token, exist_ok=True)
    model.push_to_hub(args.repo, token=token)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=tok_file,
        path_in_repo="tokenizer.json",
        repo_id=args.repo,
        token=token,
    )

    # --- Upload results metadata ---
    results = {
        "base_model": BASE_REPO,
        "method": "LoRA SFT (merged back)",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "sft_dataset": DATASET_REPO,
        "sft_examples": len(ds),
        "epochs": args.epochs,
        "lr": args.lr,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_len": args.max_len,
        "training_minutes": round(total_time / 60, 1),
        "inference_test": inference_results,
    }
    with open("/tmp/sft_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    api.upload_file(
        path_or_fileobj="/tmp/sft_results.json",
        path_in_repo="sft_results.json",
        repo_id=args.repo,
        token=token,
    )

    print(f"\nDone! https://huggingface.co/{args.repo}")
    send_tg(
        f"[SFT-LoRA Qwen500M] DONE! https://huggingface.co/{args.repo}\n"
        f"Time: {total_time/60:.1f}min\n\n" + "\n\n".join(inference_results[:3])
    )


if __name__ == "__main__":
    main()
