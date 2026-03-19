#!/usr/bin/env python3
"""Step 4: Generate text with the trained Kazakh model.

Usage:
    python generate.py --model saken-tukenov/sozkz-core-llama-50m-kk-base-v2
    python generate.py --model ./output/final
    python generate.py --model saken-tukenov/sozkz-core-llama-50m-kk-base-v2 --prompt "Қазақстан — бұл"
    python generate.py --model saken-tukenov/sozkz-core-llama-50m-kk-base-v2 --prompts-file prompts_kk.txt
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "Қазақстан — бұл",
    "Біздің елдің астанасы",
    "Қазақ тілі — ол",
    "Ғылым мен білім",
    "Алматы қаласында",
    "Тарихта қазақ халқы",
    "Бүгінгі күні технология",
    "Табиғатты қорғау үшін",
    "Мектепте балалар",
    "Қазақстанның болашағы",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.8, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="saken-tukenov/sozkz-core-llama-50m-kk-base-v2")
    parser.add_argument("--prompt", default=None, help="Single prompt")
    parser.add_argument("--prompts-file", default=None, help="File with prompts (one per line)")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Device: {device}, Parameters: {model.num_parameters()/1e6:.1f}M\n")

    # Collect prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        prompts = DEFAULT_PROMPTS

    for i, prompt in enumerate(prompts, 1):
        text = generate(model, tokenizer, prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature)
        print(f"[{i:2d}] {text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
