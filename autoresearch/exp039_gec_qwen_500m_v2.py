"""LoRA SFT for Qwen 500M -> Kazakh GEC v2.
Base: stukenov/sozkz-core-qwen-500m-kk-base-v1
Data: stukenov/sozkz-gec-synthetic-gpt4o-v1 (9,599 examples: 7,204 synthetic + 2,395 identity)
Output: stukenov/sozkz-fix-qwen-500m-kk-gec-v2 (merged full model)

Changes from v1:
  - Full dataset (9,599 vs ~3,740)
  - Clean data/train.jsonl (no more stage files)
  - Eval split (5%) for validation

Usage:
    python3 exp039_gec_qwen_500m_v2.py --dry-run
    python3 exp039_gec_qwen_500m_v2.py --epochs 3 --lr 2e-4
"""
import os, json, time, argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import hf_hub_download, HfApi, create_repo


BASE_REPO = "stukenov/sozkz-core-qwen-500m-kk-base-v1"
DATASET_REPO = "stukenov/sozkz-gec-synthetic-gpt4o-v1"
OUTPUT_REPO = "stukenov/sozkz-fix-qwen-500m-kk-gec-v2"

INSTRUCTION = (
    "Мәтіндегі грамматикалық, орфографиялық, пунктуациялық және сөз қолданысындағы "
    "қателерді түзет. Мағынаны өзгертпе. Егер мәтін дұрыс болса, оны өзгеріссіз қайтар. "
    "Тек түзетілген мәтінді қайтар."
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
    return {"text": f"### Нұсқау:\n{INSTRUCTION}\n\n### Мәтін:\n{inp}\n\n### Түзетілген:\n{out}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--repo", default=OUTPUT_REPO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

    send_tg(f"[GEC Qwen500M v2] Starting: LoRA r={args.lora_r}, lr={args.lr}, {args.epochs} epochs, full dataset")

    # --- Load model ---
    print(f"Loading base model {BASE_REPO}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO, dtype=torch.bfloat16, device_map="auto", token=token,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params, hidden={model.config.hidden_size}, "
          f"layers={model.config.num_hidden_layers}, heads={model.config.num_attention_heads}")

    # --- Load tokenizer ---
    tok_file = hf_hub_download(BASE_REPO, "tokenizer.json", token=token)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1
    tokenizer.padding_side = "right"
    tokenizer.eos_token = "<pad>"
    tokenizer.eos_token_id = 1
    print(f"  Tokenizer vocab: {tokenizer.vocab_size}")

    # --- LoRA ---
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
    ds = load_dataset(DATASET_REPO, split="train", token=token)
    print(f"  Raw: {len(ds)} examples, columns={ds.column_names}")
    ds = ds.filter(lambda x: x.get("input") and x.get("output"))
    ds = ds.map(format_gec, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=42)

    # Train/eval split
    split = ds.train_test_split(test_size=args.eval_ratio, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    print(f"  Sample:\n  {train_ds[0]['text'][:300]}")

    # --- Train ---
    t0 = time.time()
    effective_batch = args.batch_size * args.grad_accum
    total_steps = (len(train_ds) // effective_batch) * args.epochs
    max_steps = 5 if args.dry_run else -1

    training_args = SFTConfig(
        output_dir="/tmp/gec_qwen500m_v2",
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
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
        max_length=args.max_len,
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"Training: {total_steps} steps, effective_batch={effective_batch}, "
          f"train={len(train_ds)}, eval={len(eval_ds)}")
    trainer.train()
    train_time = time.time() - t0
    final_metrics = trainer.evaluate()
    print(f"Training done in {train_time/60:.1f} min, eval_loss={final_metrics.get('eval_loss', '?')}")

    if args.dry_run:
        print("DRY-RUN complete.")
        send_tg(f"[GEC Qwen500M v2] Dry-run OK, {train_time:.0f}s")
        return

    # --- Merge LoRA ---
    print("Merging LoRA into base...")
    model = model.merge_and_unload()

    # --- GEC smoke test ---
    print("\n=== GEC Smoke Test ===")
    model.train(False)
    test_cases = [
        ("Қазақстан Орталық Азиядағы ең ірі мемлекет болап табылады.", "Қазақстан Орталық Азиядағы ең ірі мемлекет болып табылады."),
        ("Мұғалім балаларға жаңа тақрыпты түсіндіріп берді.", "Мұғалім балаларға жаңа тақырыпты түсіндіріп берді."),
        ("Мен бугін мектепке бардым", "Мен бүгін мектепке бардым."),
        ("Ол кітапты оқыды жане маған айтты", "Ол кітапты оқыды және маған айтты."),
        ("Казакстанда тумыс деңгейі жоғары", "Қазақстанда тұрмыс деңгейі жоғары."),
        ("Бала бакшага барады", "Бала бақшаға барады."),
        ("Мугалім сабақты тусіндірді", "Мұғалім сабақты түсіндірді."),
        ("Менің досым келді", "Менің досым келді."),
        ("Ол кеше келіп, бүгін кетті", "Ол кеше келіп, бүгін кетті."),
        ("Біздің мектеп ен жаксы мектеп", "Біздің мектеп ең жақсы мектеп."),
        ("Мен кешке дукенге барамын", "Мен кешке дүкенге барамын."),
        ("Анам тамак пісірді", "Анам тамақ пісірді."),
        ("Мен бугін жумыска бардым", "Мен бүгін жұмысқа бардым."),
        ("Қала орталығында заманауи сауда орталығы ашылды.", None),
        ("Бүгінгі таңда технология өмірінде маңызды роль атқарады.", "Бүгінгі таңда технология адам өмірінде маңызды рөл атқарады."),
    ]
    results_log = []
    correct = 0
    total_tests = 0
    for inp, exp in test_cases:
        prompt = f"### Нұсқау:\n{INSTRUCTION}\n\n### Мәтін:\n{inp}\n\n### Түзетілген:\n"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=min(ids.shape[1], 512), do_sample=False,
                repetition_penalty=1.1, pad_token_id=1,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        corrected = decoded.split("### Түзетілген:\n")[-1].split("###")[0].strip()
        status = ""
        if exp:
            total_tests += 1
            if corrected.strip().rstrip(".") == exp.strip().rstrip("."):
                status = " PASS"
                correct += 1
            else:
                status = " FAIL"
        print(f"\n  IN:  {inp}")
        print(f"  OUT: {corrected}{status}")
        if exp:
            print(f"  EXP: {exp}")
        results_log.append({"input": inp, "output": corrected, "expected": exp, "status": status.strip()})

    score = f"{correct}/{total_tests}" if total_tests > 0 else "N/A"
    print(f"\n  Score: {score}")

    # --- Publish ---
    print(f"\nPublishing to {args.repo}...")
    create_repo(args.repo, token=token, exist_ok=True)
    model.push_to_hub(args.repo, token=token)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=tok_file, path_in_repo="tokenizer.json",
        repo_id=args.repo, token=token,
    )

    results = {
        "base_model": BASE_REPO,
        "version": "v2",
        "task": "kazakh-gec",
        "method": "LoRA SFT (merged)",
        "dataset": DATASET_REPO,
        "dataset_size": len(train_ds) + len(eval_ds),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "eval_loss": final_metrics.get("eval_loss"),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "effective_batch": effective_batch,
        "max_len": args.max_len,
        "training_minutes": round(train_time / 60, 1),
        "smoke_test_score": score,
        "smoke_test": results_log,
    }
    with open("/tmp/gec_results_v2.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    api.upload_file(
        path_or_fileobj="/tmp/gec_results_v2.json", path_in_repo="gec_results.json",
        repo_id=args.repo, token=token,
    )

    print(f"\nDone! https://huggingface.co/{args.repo}")
    send_tg(
        f"[GEC Qwen500M v2] DONE! https://huggingface.co/{args.repo}\n"
        f"Time: {train_time/60:.1f}min, data: {len(train_ds)+len(eval_ds)} examples\n"
        f"Eval loss: {final_metrics.get('eval_loss', '?')}\n"
        f"Smoke test: {score}\n\n"
        + "\n".join(f"IN: {r['input'][:50]}... -> {r['status']}" for r in results_log[:5] if r.get("status"))
    )


if __name__ == "__main__":
    main()
