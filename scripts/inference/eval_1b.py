#!/usr/bin/env python3
"""Generate text from the 1.08B Kazakh model for quality check."""
import torch
import json
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

MODEL = "stukenov/sozkz-core-llama-1b-kk-base-v1"

PROMPTS = [
    ("Саясат", "Қазақстан Президенті"),
    ("Экономика", "Экономика министрлігі"),
    ("Білім", "Білім беру жүйесі"),
    ("Ауа райы", "Ауа райы болжамы бойынша"),
    ("Тарих", "Қазақ халқының тарихы"),
    ("Технология", "Жасанды интеллект технологиясы"),
    ("Спорт", "Футбол чемпионаты"),
    ("Денсаулық", "Денсаулық сақтау министрлігі"),
    ("Күнделікті", "Алматы қаласында бүгін"),
    ("Ғылым", "Қазіргі заманғы физика ғылымында"),
    ("Әдебиет", "Алыстағы ауылда бір кәрі шал тұратын,"),
    ("Жалпы", "Қазақстан — бұл"),
]


def main():
    print("Loading model...")
    tok_file = hf_hub_download(MODEL, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {params_m:.1f}M params")

    results = []
    for category, prompt in PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"category": category, "prompt": prompt, "generation": text})
        print(f"\n=== [{category}] {prompt} ===")
        print(text)
        print()

    with open("/root/slm/results/eval_1b_generations.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} generations")


if __name__ == "__main__":
    main()
