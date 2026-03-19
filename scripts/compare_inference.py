"""Compare inference across 150M, 300M, 600M models on the same prompts."""
import torch
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("150M", "stukenov/sozkz-core-llama-150m-kk-base-v1"),
    ("300M", "stukenov/sozkz-core-llama-300m-kk-base-v1"),
    ("600M", "stukenov/sozkz-core-llama-600m-kk-base-v1"),
]

PROMPTS = [
    "Қазақстан — ",
    "Қазақ халқының тарихы",
    "Тіл үйрену үшін",
    "Жасанды интеллект дегеніміз",
    "Алматы қаласында",
]

GEN_KWARGS = dict(
    max_new_tokens=150,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODELS[0][1])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

torch.manual_seed(42)

for name, model_id in MODELS:
    print(f"\n{'='*80}")
    print(f"  MODEL: {name} ({model_id})")
    print(f"{'='*80}")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    model.requires_grad_(False)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {num_params:.1f}M")

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.manual_seed(42)

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, **GEN_KWARGS)
        elapsed = time.time() - t0

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
        tps = new_tokens / elapsed

        print(f"\n  PROMPT: {prompt}")
        print(f"  OUTPUT ({new_tokens} tokens, {tps:.0f} tok/s):")
        print(f"  {text}")
        print(f"  {'_'*70}")

    del model
    torch.cuda.empty_cache()

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}")
