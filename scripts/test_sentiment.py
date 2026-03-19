"""Test sentiment classification with <sentiment> tag."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def classify(model, tokenizer, text: str, device: str = "cpu") -> str:
    prompt = f"<sentiment>{text}</sentiment>\n"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stukenov/sozkz-core-llama-600m-kk-sentiment-v1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(args.device).eval()

    tests = [
        # Positive
        "Тамақтары өте дәмді, қызмет көрсету керемет!",
        "Бұл қосымша өте ыңғайлы, маған ұнады",
        "Жақсы кітап, оқуға кеңес беремін",
        "Керемет орын, тағы келемін",
        # Negative
        "Қызмет көрсету нашар, тамақ суық",
        "Бұл қосымша жұмыс істемейді, ақша ысырап",
        "Өте нашар сапа, ақшаға тұрмайды",
        "Ешқашан бармаймын, уақытымды босқа өткіздім",
        # Ambiguous
        "Тамақ жаман емес, бірақ баға қымбат",
        "Қалыпты, ерекше ештеңе жоқ",
    ]

    print(f"\n{'Text':<55} {'Prediction'}")
    print("-" * 75)
    for text in tests:
        pred = classify(model, tokenizer, text, args.device)
        display = text[:52] + "..." if len(text) > 55 else text
        print(f"{display:<55} {pred}")


if __name__ == "__main__":
    main()
