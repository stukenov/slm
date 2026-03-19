"""Inference script for GEC model."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm.data_gec import SEP, SRC, TASK_FIX


def main():
    parser = argparse.ArgumentParser(description="GEC Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--text", type=str, required=True, help="Text to correct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    prompt = f"{TASK_FIX}{SRC}{args.text}{SEP}"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else 1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][prompt_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    print(f"Input:  {args.text}")
    print(f"Output: {result}")


if __name__ == "__main__":
    main()
