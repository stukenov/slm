#!/usr/bin/env python3
"""Run inference demos on EkiTil models."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

prompts = [
    "Қазақстан — Орталық Азиядағы",
    "Москва — это столица",
    "Бүгін ауа райы өте",
    "Искусственный интеллект — это",
    "<|kk|> Менің атым Сакен. <|translate|> <|ru|>",
]

models = [
    "stukenov/ekitil-core-qwen3-123m-kkru-base-v1",
    "stukenov/ekitil-core-qwen3-300m-kkru-base-v1",
]

for model_name in models:
    sep = "=" * 60
    print(f"\n{sep}")
    short = model_name.split("/")[-1]
    print(f"  MODEL: {short}")
    print(sep)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {n_params:.1f}M params\n")

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"PROMPT: {prompt}")
        print(f"OUTPUT: {text}")
        print()

    del model
    torch.cuda.empty_cache()

print("INFERENCE_DONE")
