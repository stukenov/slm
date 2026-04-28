"""KTO training for Kazakh GEC v4.
Focus: improve punctuation (53% → 80%+).
Base: v3 SFT model, apply KTO with comma/period/identity preference data.
Output: stukenov/sozkz-fix-qwen-500m-kk-gec-v4
"""
import os, json, time, random
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
from trl import KTOTrainer, KTOConfig
from huggingface_hub import hf_hub_download, HfApi, create_repo

MODEL_REPO = "stukenov/sozkz-fix-qwen-500m-kk-gec-v3"
OUTPUT_REPO = "stukenov/sozkz-fix-qwen-500m-kk-gec-v4"
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


def format_prompt(text):
    return f"### Нұсқау:\n{INSTRUCTION}\n\n### Мәтін:\n{text}\n\n### Түзетілген:\n"


def extract_comma_pairs(dataset, n=3000):
    """Extract comma-insertion pairs from correct outputs."""
    random.seed(42)
    pairs = []
    for row in dataset:
        out = row.get("output", "")
        if not out or "," not in out:
            continue
        comma_positions = [i for i, c in enumerate(out) if c == ","]
        pos = random.choice(comma_positions)
        if pos + 1 < len(out) and out[pos + 1] == " ":
            without = out[:pos] + out[pos + 2:]
        else:
            without = out[:pos] + out[pos + 1:]
        pairs.append((without.strip(), out.strip()))
    random.shuffle(pairs)
    return pairs[:n]


def extract_period_pairs(dataset, n=1000):
    """Create period-at-end pairs from correct outputs."""
    pairs = []
    for row in dataset:
        out = row.get("output", "")
        if not out or not out.endswith(".") or len(out) < 10:
            continue
        pairs.append((out.rstrip("."), out))
    random.shuffle(pairs)
    return pairs[:n]


def generate_template_pairs(n=2000):
    """Template-based compound sentences for comma training."""
    random.seed(123)
    subjects = [
        "Мен", "Ол", "Біз", "Олар", "Бала", "Анам", "Атам",
        "Досым", "Мұғалім", "Інім", "Әпкем", "Ағам", "Балалар",
        "Қыз", "Жігіт", "Дәрігер", "Оқушы", "Студент",
    ]
    verbs = [
        "келді", "кетті", "барды", "жазды", "оқыды", "істеді",
        "айтты", "берді", "алды", "көрді", "тұрды", "отырды",
        "жүрді", "сөйледі", "күлді", "ойнады", "тыңдады", "қарады",
    ]
    intros = ["Иә", "Жоқ", "Алайда", "Сондықтан", "Әрине", "Рас", "Мүмкін", "Шынында"]
    conjs = [
        (" ал ", ", ал "), (" бірақ ", ", бірақ "),
        (" себебі ", ", себебі "), (" өйткені ", ", өйткені "),
    ]
    pairs = []

    for _ in range(n // 3):
        s1, s2 = random.sample(subjects, 2)
        v1, v2 = random.sample(verbs, 2)
        correct = f"{s1} {v1}, {s2} {v2}."
        incorrect = f"{s1} {v1} {s2} {v2}"
        pairs.append((incorrect, correct))

    for _ in range(n // 3):
        s1, s2 = random.sample(subjects, 2)
        v1, v2 = random.sample(verbs, 2)
        bad_conj, good_conj = random.choice(conjs)
        correct = f"{s1} {v1}{good_conj}{s2.lower()} {v2}."
        incorrect = f"{s1} {v1}{bad_conj}{s2.lower()} {v2}"
        pairs.append((incorrect, correct))

    for _ in range(n // 3):
        s = random.choice(subjects)
        v = random.choice(verbs)
        intro = random.choice(intros)
        correct = f"{intro}, {s.lower()} {v}."
        incorrect = f"{intro} {s.lower()} {v}"
        pairs.append((incorrect, correct))

    random.shuffle(pairs)
    return pairs[:n]


def main():
    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.cache/huggingface/token")
    ).read().strip()
    send_tg("[GEC v4] KTO training starting")

    print(f"Loading {MODEL_REPO}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, device_map="auto", token=token,
    )
    tok_file = hf_hub_download(MODEL_REPO, "tokenizer.json", token=token)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1
    tokenizer.padding_side = "right"
    tokenizer.eos_token = "<pad>"
    tokenizer.eos_token_id = 1

    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading GEC dataset...")
    orig_ds = load_dataset(DATASET_REPO, split="train", token=token)
    print(f"  Original: {len(orig_ds)} examples")

    print("Extracting punctuation pairs from dataset...")
    comma_pairs = extract_comma_pairs(orig_ds, n=3000)
    period_pairs = extract_period_pairs(orig_ds, n=1000)
    template_pairs = generate_template_pairs(n=2000)
    print(f"  Comma: {len(comma_pairs)}, Period: {len(period_pairs)}, Template: {len(template_pairs)}")

    print("Building KTO dataset...")
    kto_rows = []

    for row in orig_ds:
        inp = row.get("input", "")
        out = row.get("output", "")
        if not inp or not out or inp.strip() == out.strip():
            continue
        prompt = format_prompt(inp)
        kto_rows.append({"prompt": prompt, "completion": out, "label": True})
        kto_rows.append({"prompt": prompt, "completion": inp, "label": False})

    for bad, good in comma_pairs + period_pairs + template_pairs:
        prompt = format_prompt(bad)
        kto_rows.append({"prompt": prompt, "completion": good, "label": True})
        kto_rows.append({"prompt": prompt, "completion": bad, "label": False})

    random.seed(42)
    random.shuffle(kto_rows)

    n_pos = sum(1 for r in kto_rows if r["label"])
    n_neg = sum(1 for r in kto_rows if not r["label"])
    print(f"  Total: {len(kto_rows)} ({n_pos} positive, {n_neg} negative)")

    ds = Dataset.from_list(kto_rows)
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    t0 = time.time()
    effective_batch = 4 * 8
    total_steps = len(train_ds) // effective_batch

    kto_config = KTOConfig(
        output_dir="/workspace/gec_v4_kto",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(total_steps // 4, 50),
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=max(total_steps // 8, 25),
        max_length=512,
        max_prompt_length=384,
        max_completion_length=128,
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"Training KTO: {total_steps} steps, effective_batch={effective_batch}")
    trainer.train()
    train_time = time.time() - t0
    final_metrics = trainer.evaluate()
    print(f"Done in {train_time/60:.1f}min")
    print(f"Metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in final_metrics.items()}, indent=2)}")

    print("Merging LoRA...")
    model = model.merge_and_unload()

    model.train(False)
    test_cases = [
        ("Мен бугін мектепке бардым", "Мен бүгін мектепке бардым."),
        ("Ол кеше келіп бүгін кетті", "Ол кеше келіп, бүгін кетті."),
        ("Иә мен келемін", "Иә, мен келемін."),
        ("Мен жұмысқа бардым ол үйде қалды", "Мен жұмысқа бардым, ол үйде қалды."),
        ("Жаңбыр жауды біз үйде отырдық", "Жаңбыр жауды, біз үйде отырдық."),
        ("Бала бакшага барады", "Бала бақшаға барады."),
        ("Менің досым келді.", "Менің досым келді."),
        ("Біздің мектеп ен жаксы мектеп", "Біздің мектеп ең жақсы мектеп."),
        ("Анам тамак пісірді", "Анам тамақ пісірді."),
        ("Казакстан Орталык Азиядагы ен ірі мемлекет", "Қазақстан Орталық Азиядағы ең ірі мемлекет."),
    ]
    correct_count = 0
    for inp, exp in test_cases:
        prompt = format_prompt(inp)
        ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=min(ids.shape[1], 512),
                num_beams=4, do_sample=False, repetition_penalty=1.0, pad_token_id=1,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        corrected = decoded.split("### Түзетілген:\n")[-1].split("###")[0].strip()
        ok = corrected.strip().rstrip(".") == exp.strip().rstrip(".")
        if ok:
            correct_count += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}: {inp[:50]} -> {corrected}")
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
        "base_model": MODEL_REPO,
        "version": "v4",
        "task": "kazakh-gec",
        "method": "KTO on SFT v3 (LoRA r=32, merged)",
        "focus": "punctuation improvement",
        "kto_data_size": len(train_ds) + len(eval_ds),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "kto_positive": n_pos,
        "kto_negative": n_neg,
        "training_minutes": round(train_time / 60, 1),
        "smoke_test_score": score,
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in final_metrics.items()},
    }
    with open("/tmp/gec_results_v4.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    api.upload_file(
        path_or_fileobj="/tmp/gec_results_v4.json", path_in_repo="gec_results.json",
        repo_id=OUTPUT_REPO, token=token,
    )

    print(f"\nDone! https://huggingface.co/{OUTPUT_REPO}")
    send_tg(
        f"[GEC v4] DONE! KTO training\n"
        f"https://huggingface.co/{OUTPUT_REPO}\n"
        f"Time: {train_time/60:.1f}min\n"
        f"Data: {len(train_ds)+len(eval_ds)} ({n_pos} pos, {n_neg} neg)\n"
        f"Smoke: {score}"
    )


if __name__ == "__main__":
    main()
