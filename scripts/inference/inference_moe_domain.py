"""Domain-specific inference demo for sozkz-moe-mix-160m-kk-domain-v1."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "saken-tukenov/sozkz-moe-mix-160m-kk-domain-v1"
TOKENIZER = "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1"

DOMAIN_PROMPTS = {
    "Новости / Журналистика": [
        "Астанада бүгін өткен саммитте",
        "Президент Тоқаев мәлімдеме жасап,",
        "Қазақстан экономикасы 2024 жылы",
    ],
    "Веб / Социальный текст": [
        "Сәлем достар, бүгін мен сендерге",
        "Менің ойымша, бұл мәселе",
        "Интернетте жиі кездесетін",
    ],
    "Энциклопедический / Фактический": [
        "Қазақстан — Орталық Азиядағы",
        "Алматы қаласының тарихы",
        "Қазақ тілі — түркі тілдер",
    ],
    "Литературный / Художественный": [
        "Абай Құнанбайұлы — ұлы қазақ",
        "Кеш батып, аспан қызыл нұрға",
        "Даланың ортасында жалғыз үй",
    ],
    "Академический / Научный": [
        "Ғылыми зерттеулер көрсеткендей,",
        "Білім беру жүйесін жаңғырту",
        "Математикалық модельдеу әдістері",
    ],
    "Разговорный / QA": [
        "Сұрақ: Қазақстанның астанасы қай қала? Жауап:",
        "Сұрақ: Наурыз мейрамы қашан тойланады? Жауап:",
        "Сұрақ: Қазақ тілінде неше дауысты дыбыс бар? Жауап:",
    ],
}


def main():
    print(f"Loading tokenizer: {TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    print(f"Loading model: {MODEL}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    device = "cpu"  # MPS doesn't support histc for MoE routing
    model.to(device)
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    print("=" * 70)

    results = []
    for domain, prompts in DOMAIN_PROMPTS.items():
        header = f"\n{'='*70}\n  {domain}\n{'='*70}"
        print(header)
        results.append(header)

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    repetition_penalty=1.2,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            line = f"\nPrompt: {prompt}\nOutput: {text}"
            print(line)
            print("-" * 50)
            results.append(line)

    import os
    os.makedirs("results", exist_ok=True)
    out_path = "results/inference_moe_domain.txt"
    with open(out_path, "w") as f:
        f.write(f"Model: {MODEL}\nTokenizer: {TOKENIZER}\nDevice: {device}\n\n")
        f.write("\n".join(results))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
