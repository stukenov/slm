"""Run GEC inference on examples from the dataset and show results."""

import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from slm.data_gec import SEP, SRC, TASK_FIX
from slm.data_gec_filtered import classify_error

MODEL = "saken-tukenov/sozkz-fix-mt5-50m-kk-morph-v1"
TOKENIZER = MODEL
N_EXAMPLES = 20
MAX_NEW_TOKENS = 256


def infer(model, tokenizer, text, device):
    prompt = f"{TASK_FIX}{SRC}{text}{SEP}"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    print(f"Loading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Device: {device}\n")

    # Load validation split of the raw GEC dataset, filter morphology examples
    print("Loading dataset...")
    ds = load_dataset("saken-tukenov/sozkz-corpus-synthetic-kk-gec-v1", data_dir="data/grammar_balanced_v2")
    val = ds.get("validation") or ds.get("test") or ds["train"]

    # Filter morphology examples
    morpho_examples = []
    for i in range(len(val)):
        inp, tgt = val[i]["input"], val[i]["target"]
        if classify_error(inp, tgt) == "morphology":
            morpho_examples.append((inp, tgt))
        if len(morpho_examples) >= N_EXAMPLES:
            break

    print(f"Found {len(morpho_examples)} morphology examples\n")
    print("=" * 80)

    correct = 0
    for i, (inp, tgt) in enumerate(morpho_examples):
        pred = infer(model, tokenizer, inp, device)
        match = "✓" if pred.strip() == tgt.strip() else "✗"
        if pred.strip() == tgt.strip():
            correct += 1

        print(f"\n[{i+1}] {match}")
        print(f"  Input:    {inp}")
        print(f"  Expected: {tgt}")
        print(f"  Model:    {pred}")

    print("\n" + "=" * 80)
    print(f"Accuracy: {correct}/{len(morpho_examples)} ({correct*100/len(morpho_examples):.1f}%)")


if __name__ == "__main__":
    main()
