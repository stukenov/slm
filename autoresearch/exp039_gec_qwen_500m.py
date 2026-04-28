"""LoRA SFT for Qwen 500M -> Kazakh GEC model.
Base: stukenov/sozkz-core-qwen-500m-kk-base-v1
Data: stukenov/sozkz-gec-synthetic-gpt4o-v1 (cumulative stage JSONL)
Output: stukenov/sozkz-fix-qwen-500m-kk-gec-v1 (merged full model)

Usage:
    # Dry run (5 steps)
    python3 exp039_gec_qwen_500m.py --dry-run

    # Full training on RunPod
    python3 exp039_gec_qwen_500m.py --epochs 3 --lr 2e-4 --stage 7
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


def find_latest_stage(repo: str, token: str) -> str:
    api = HfApi(token=token)
    files = api.list_repo_files(repo, repo_type="dataset")
    stages = sorted([f for f in files if f.startswith("data/stage_") and f.endswith(".jsonl")])
    if not stages:
        raise RuntimeError(f"No stage files in {repo}")
    return stages[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--stage", type=int, default=0, help="Stage number (0=latest)")
    parser.add_argument("--repo", default="stukenov/sozkz-fix-qwen-500m-kk-gec-v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

    send_tg(f"[GEC Qwen500M] Starting: LoRA r={args.lora_r}, lr={args.lr}, {args.epochs} epochs")

    # --- Find dataset stage ---
    if args.stage > 0:
        api = HfApi(token=token)
        files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
        stage_files = sorted([f for f in files if f.startswith(f"data/stage_{args.stage:02d}_")])
        if not stage_files:
            raise RuntimeError(f"Stage {args.stage} not found in {DATASET_REPO}")
        data_file = stage_files[-1]
    else:
        data_file = find_latest_stage(DATASET_REPO, token)
    print(f"Using dataset: {DATASET_REPO}/{data_file}")

    # --- Load model ---
    print(f"Loading base model {BASE_REPO}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO, dtype=torch.bfloat16, device_map="auto",
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
    print(f"Loading dataset {DATASET_REPO} / {data_file}...")
    ds = load_dataset(DATASET_REPO, data_files=data_file, split="train")
    print(f"  Raw: {len(ds)} examples, columns={ds.column_names}")
    ds = ds.filter(lambda x: x.get("input") and x.get("output"))
    ds = ds.map(format_gec, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=42)
    print(f"  Formatted: {len(ds)} examples")
    print(f"  Sample:\n  {ds[0]['text'][:300]}")

    # --- Train ---
    t0 = time.time()
    effective_batch = args.batch_size * args.grad_accum
    total_steps = (len(ds) // effective_batch) * args.epochs
    max_steps = 5 if args.dry_run else -1

    training_args = SFTConfig(
        output_dir="/tmp/gec_qwen500m",
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

    print(f"Training: {total_steps} steps, effective_batch={effective_batch}")
    trainer.train()
    train_time = time.time() - t0
    print(f"Training done in {train_time/60:.1f} min")

    if args.dry_run:
        print("DRY-RUN complete.")
        send_tg(f"[GEC Qwen500M] Dry-run OK, {train_time:.0f}s")
        return

    # --- Merge LoRA ---
    print("Merging LoRA into base...")
    model = model.merge_and_unload()

    # --- GEC smoke test ---
    print("\n=== GEC Smoke Test ===")
    model.train(False)
    test_inputs = [
        "Қазақстан Орталық Азиядағы ең ірі мемлекет болап табылады.",
        "Мұғалім балаларға жаңа тақрыпты түсіндіріп берді.",
        "Бүгінгі таңда технология өмірінде маңызды роль атқарады.",
        "Ол кеде біз мектепке күн сайын жаяу барушы едек.",
        "Қала орталығында заманауи сауда орталығы ашылды.",
    ]
    expected = [
        "Қазақстан Орталық Азиядағы ең ірі мемлекет болып табылады.",
        "Мұғалім балаларға жаңа тақырыпты түсіндіріп берді.",
        "Бүгінгі таңда технология адам өмірінде маңызды рөл атқарады.",
        "Ол кезде біз мектепке күн сайын жаяу барушы едік.",
        None,
    ]
    results_log = []
    for inp, exp in zip(test_inputs, expected):
        prompt = f"### Нұсқау:\n{INSTRUCTION}\n\n### Мәтін:\n{inp}\n\n### Түзетілген:\n"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=150, do_sample=False,
                repetition_penalty=1.1, pad_token_id=1,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        corrected = decoded.split("### Түзетілген:\n")[-1].split("###")[0].strip()
        status = ""
        if exp:
            status = " OK" if corrected.strip() == exp.strip() else " DIFF"
        print(f"\n  IN:  {inp}")
        print(f"  OUT: {corrected}{status}")
        if exp:
            print(f"  EXP: {exp}")
        results_log.append(f"IN: {inp}\nOUT: {corrected}")

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
        "task": "kazakh-gec",
        "method": "LoRA SFT (merged)",
        "dataset": f"{DATASET_REPO}/{data_file}",
        "dataset_size": len(ds),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "effective_batch": effective_batch,
        "max_len": args.max_len,
        "training_minutes": round(train_time / 60, 1),
        "smoke_test": results_log,
    }
    with open("/tmp/gec_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    api.upload_file(
        path_or_fileobj="/tmp/gec_results.json", path_in_repo="gec_results.json",
        repo_id=args.repo, token=token,
    )

    print(f"\nDone! https://huggingface.co/{args.repo}")
    send_tg(
        f"[GEC Qwen500M] DONE! https://huggingface.co/{args.repo}\n"
        f"Time: {train_time/60:.1f}min, data: {len(ds)} examples\n\n"
        + "\n\n".join(results_log[:3])
    )


if __name__ == "__main__":
    main()
