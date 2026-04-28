"""Upload trained 1.08B model to HuggingFace Hub with FULL verification.
exp028v2: No QK-Norm, no embedding scaling — pure HF-compatible Llama.
Uses strict=True and verifies inference before uploading.
"""
import os, sys, json, torch
from huggingface_hub import HfApi, create_repo

repo = os.environ.get("HF_REPO", "stukenov/sozkz-core-llama-1b-kk-base-v1")
token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set"); sys.exit(1)

print("Loading checkpoint...")
ckpt = torch.load("/root/checkpoints/exp028_1b/final/model.pt", map_location="cpu", weights_only=True)
results = json.load(open("/root/checkpoints/exp028_1b/final/results.json"))
print(f"Results: {json.dumps(results, indent=2)}")

# Strip _orig_mod. prefix (torch.compile artifact)
cleaned = {}
for k, v in ckpt.items():
    cleaned[k.removeprefix("_orig_mod.")] = v
if any(k.startswith("_orig_mod.") for k in ckpt):
    print(f"  Stripped _orig_mod. prefix from {sum(1 for k in ckpt if k.startswith('_orig_mod.'))} keys")
ckpt = cleaned

# Map custom keys to HF Llama format
# NOTE: No QK-Norm keys — they don't exist in this model (exp028v2)
print("Mapping state dict to HF format...")
state = {}
key_map = {
    "emb.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "lm_head.weight",
}
for k, v in ckpt.items():
    if k.startswith("rot."):
        print(f"  {k} -> (skipped, HF computes rotary internally)")
        continue
    new_k = k
    if k in key_map:
        new_k = key_map[k]
    else:
        new_k = new_k.replace("layers.", "model.layers.")
        new_k = new_k.replace(".ln1.", ".input_layernorm.")
        new_k = new_k.replace(".ln2.", ".post_attention_layernorm.")
        new_k = new_k.replace(".attn.q.", ".self_attn.q_proj.")
        new_k = new_k.replace(".attn.k.", ".self_attn.k_proj.")
        new_k = new_k.replace(".attn.v.", ".self_attn.v_proj.")
        new_k = new_k.replace(".attn.o.", ".self_attn.o_proj.")
        new_k = new_k.replace(".mlp.g.", ".mlp.gate_proj.")
        new_k = new_k.replace(".mlp.u.", ".mlp.up_proj.")
        new_k = new_k.replace(".mlp.d.", ".mlp.down_proj.")
    state[new_k] = v
    if k != new_k:
        print(f"  {k} -> {new_k}")

print("Creating HF model...")
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

config = LlamaConfig(
    vocab_size=50257, hidden_size=2048, intermediate_size=5504,
    num_hidden_layers=22, num_attention_heads=16, num_key_value_heads=4,
    max_position_embeddings=2048, tie_word_embeddings=True,
    bos_token_id=0, eos_token_id=0, pad_token_id=1,
    torch_dtype="bfloat16",
)
model = LlamaForCausalLM(config)

# CRITICAL: strict=True — fail on ANY missing/unexpected keys
# This is what caught the QK-Norm problem in exp028
print("Loading state dict with strict=True...")
missing, unexpected = model.load_state_dict(state, strict=True)
# strict=True raises on mismatch, but just in case:
assert not missing, f"FATAL: Missing keys: {missing}"
assert not unexpected, f"FATAL: Unexpected keys: {unexpected}"
print("  All keys matched perfectly!")

# ============================================================
# VERIFICATION: Run inference BEFORE uploading
# ============================================================
print("\n=== VERIFICATION: Testing inference before upload ===")
model = model.to(dtype=torch.bfloat16, device="cuda")

tok_file = hf_hub_download("stukenov/sozkz-core-gpt2-50k-kk-base-v1", "tokenizer.json")
tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
tok.pad_token_id = 1

test_prompts = [
    "Қазақстан Президенті",
    "Білім беру жүйесі",
    "Алматы қаласында бүгін",
]

all_ok = True
for prompt in test_prompts:
    ids = tok.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {text[:200]}")

    # Check: output should contain Cyrillic characters (Kazakh)
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
    cyrillic_ratio = cyrillic_count / max(len(text), 1)
    if cyrillic_ratio < 0.3:
        print(f"  WARNING: Low Cyrillic ratio ({cyrillic_ratio:.1%}) — model may be broken!")
        all_ok = False
    else:
        print(f"  OK: Cyrillic ratio {cyrillic_ratio:.1%}")

if not all_ok:
    print("\nFATAL: Inference verification FAILED. NOT uploading.")
    print("Checkpoint preserved at /root/checkpoints/exp028_1b/final/")
    sys.exit(1)

print("\n=== Inference verification PASSED ===\n")

# Move model back to CPU for upload
model = model.cpu()

print(f"Uploading to {repo}...")
create_repo(repo, token=token, exist_ok=True)
model.push_to_hub(repo, token=token)

print("Uploading tokenizer...")
tok.push_to_hub(repo, token=token)

# Upload results
api = HfApi()
with open("/tmp/results.json", "w") as f:
    json.dump(results, f, indent=2)
api.upload_file(
    path_or_fileobj="/tmp/results.json",
    path_in_repo="results.json",
    repo_id=repo,
    token=token,
)

# Upload README
readme_path = os.path.join(os.path.dirname(__file__), "..", "docs", "exp028_model_card_README.md")
if os.path.exists(readme_path):
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo,
        token=token,
    )
    print("  README uploaded")

print(f"\nDone! https://huggingface.co/{repo}")
