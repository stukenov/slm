"""LoRA SFT for 1.08B Kazakh Llama → instruct version.
Uses PEFT LoRA to avoid catastrophic forgetting.
"""
import os, json, time, argparse
import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import hf_hub_download, HfApi, create_repo

def send_tg(msg):
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({"chat_id": "47474471", "text": msg}).encode()
        urllib.request.urlopen("https://api.telegram.org/bot8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g/sendMessage", data, timeout=10)
    except Exception:
        pass

def format_instruction(example):
    """Convert AmanMussa format to simple prompt."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    user_msg = instruction
    if inp:
        user_msg += f"\n{inp}"
    return {"text": f"### Сұрақ:\n{user_msg}\n\n### Жауап:\n{output}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", default="step-16000")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--repo", default="stukenov/sozkz-core-llama-1b-kk-instruct-v1")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

    send_tg(f"[SFT-LoRA] Starting: {args.base_revision}, LoRA r={args.lora_r}, lr={args.lr}, {args.epochs} epochs")

    # Load base model
    print(f"Loading base model revision={args.base_revision}...")
    model = LlamaForCausalLM.from_pretrained(
        "stukenov/sozkz-core-llama-1b-kk-base-v1",
        revision=args.base_revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Load tokenizer
    tok_file = hf_hub_download("stukenov/sozkz-core-gpt2-50k-kk-base-v1", "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1
    tokenizer.padding_side = "right"

    # LoRA config
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

    # Dataset
    print("Loading dataset...")
    ds = load_dataset("AmanMussa/kazakh-instruction-v2", split="train")
    ds = ds.filter(lambda x: x.get("instruction") and x.get("output"))
    ds = ds.map(format_instruction, remove_columns=ds.column_names)
    print(f"  {len(ds)} examples")

    # Training
    t0 = time.time()

    training_args = SFTConfig(
        output_dir="/tmp/sft_lora_output",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        max_seq_length=args.max_len,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.train()
    total_time = time.time() - t0
    print(f"SFT-LoRA done in {total_time/60:.1f}min")

    # Merge LoRA back into base model
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Inference test
    print("\n=== Inference test ===")
    model.eval()
    test_questions = [
        "Қазақстанның астанасы қай қала?",
        "Абай Құнанбаев туралы қысқаша айтып бер",
        "Жасанды интеллект дегеніміз не?",
        "Денсаулықты сақтау үшін не істеу керек?",
        "Каспий теңізі туралы айтып бер",
    ]
    inference_results = []
    for q in test_questions:
        prompt = f"### Сұрақ:\n{q}\n\n### Жауап:\n"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        resp = text.split("### Жауап:\n")[-1].split("### Сұрақ:")[0].strip()
        print(f"\nQ: {q}")
        print(f"A: {resp[:300]}")
        inference_results.append(f"Q: {q}\nA: {resp[:300]}")

    # Publish
    print(f"\nPublishing to {args.repo}...")
    create_repo(args.repo, token=token, exist_ok=True)
    model.push_to_hub(args.repo, token=token)
    tokenizer.push_to_hub(args.repo, token=token)

    # Upload results
    api = HfApi(token=token)
    results = {
        "base_model": "stukenov/sozkz-core-llama-1b-kk-base-v1",
        "base_revision": args.base_revision,
        "method": "LoRA SFT (merged)",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "sft_dataset": "AmanMussa/kazakh-instruction-v2",
        "sft_examples": len(ds),
        "epochs": args.epochs,
        "lr": args.lr,
        "training_minutes": round(total_time / 60, 1),
        "inference_test": inference_results,
    }
    with open("/tmp/sft_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    api.upload_file(path_or_fileobj="/tmp/sft_results.json", path_in_repo="sft_results.json", repo_id=args.repo, token=token)

    print(f"\nDone! https://huggingface.co/{args.repo}")
    send_tg(f"[SFT-LoRA] DONE! https://huggingface.co/{args.repo}\nTime: {total_time/60:.1f}min\n\n" + "\n\n".join(inference_results))

if __name__ == "__main__":
    main()
