"""Quick inference demo for sozkz-core-llama-150m-kk-balanced-v1."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "saken-tukenov/sozkz-core-llama-150m-kk-balanced-v1"
TOKENIZER = "saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1"

PROMPTS = [
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
    for i, prompt in enumerate(PROMPTS, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=1.0,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append(f"[{i:2d}] Prompt: {prompt}\n    Output: {text}\n")
        print(f"[{i:2d}] {text}")
        print("-" * 60)

    out_path = "results/inference_demo.txt"
    import os
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Model: {MODEL}\nTokenizer: {TOKENIZER}\nTemperature: 1.0\nDevice: {device}\n\n")
        f.write("\n".join(results))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
