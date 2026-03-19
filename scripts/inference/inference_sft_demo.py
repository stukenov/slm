"""Inference demo for sozkz-core-llama-150m-kk-instruct-v1."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

MODEL = "saken-tukenov/sozkz-core-llama-150m-kk-instruct-v1"
TOKENIZER = "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1"

EXAMPLES = [
    # Тарих / История
    {"instruction": "Қазақстанның астанасы туралы айтып бер.", "input": ""},
    # Ғылым / Наука
    {"instruction": "Фотосинтез дегеніміз не?", "input": ""},
    # Тіл / Язык
    {"instruction": "Мына сөйлемді ағылшын тіліне аудар.", "input": "Менің атым Айдос, мен Алматыда тұрамын."},
    # Мәдениет / Культура
    {"instruction": "Наурыз мейрамы туралы қысқаша айтып бер.", "input": ""},
    # Математика
    {"instruction": "Мына есепті шеш.", "input": "Егер 3x + 7 = 22 болса, x неге тең?"},
    # Кеңес / Совет
    {"instruction": "Қазақ тілін үйренуге кеңес бер.", "input": ""},
    # География
    {"instruction": "Каспий теңізі туралы не білесің?", "input": ""},
    # Технология
    {"instruction": "Жасанды интеллект дегеніміз не?", "input": ""},
    # Денсаулық / Здоровье
    {"instruction": "Дұрыс тамақтану ережелері қандай?", "input": ""},
    # Шығармашылық / Творчество
    {"instruction": "Көктем туралы қысқа өлең жаз.", "input": ""},
    # Экономика
    {"instruction": "Инфляция дегеніміз не?", "input": ""},
    # Әдебиет / Литература
    {"instruction": "Абай Құнанбаев туралы айтып бер.", "input": ""},
]


def format_prompt(instruction: str, inp: str) -> str:
    if inp:
        return f"### Нұсқаулық:\n{instruction}\n\n### Кіріс:\n{inp}\n\n### Жауап:\n"
    return f"### Нұсқаулық:\n{instruction}\n\n### Жауап:\n"


def main():
    print(f"Loading tokenizer: {TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    print(f"Loading model: {MODEL}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}\n")

    results = []
    for i, ex in enumerate(EXAMPLES, 1):
        prompt = format_prompt(ex["instruction"], ex["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # Cut at next "###" if model generates another turn
        if "###" in generated:
            generated = generated[:generated.index("###")].strip()

        results.append({
            "instruction": ex["instruction"],
            "input": ex["input"],
            "output": generated,
        })
        domain = ["Тарих", "Ғылым", "Аударма", "Мәдениет", "Математика", "Кеңес",
                   "География", "Технология", "Денсаулық", "Шығармашылық", "Экономика", "Әдебиет"][i-1]
        print(f"[{i:2d}] [{domain}]")
        print(f"    Q: {ex['instruction']}")
        if ex["input"]:
            print(f"    Input: {ex['input']}")
        print(f"    A: {generated}")
        print("-" * 70)

    import os
    os.makedirs("results", exist_ok=True)
    out_path = "results/inference_sft_demo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} examples to {out_path}")

    # Also save markdown for README
    md_path = "results/inference_sft_demo.md"
    with open(md_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"**Нұсқаулық:** {r['instruction']}\n")
            if r["input"]:
                f.write(f"**Кіріс:** {r['input']}\n")
            f.write(f"**Жауап:** {r['output']}\n\n---\n\n")
    print(f"Saved markdown to {md_path}")


if __name__ == "__main__":
    main()
