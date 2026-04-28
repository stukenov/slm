"""Upload trained 500M Qwen2 model to HuggingFace Hub with FULL verification.
exp036: Qwen2.5-style, native Qwen2ForCausalLM -- no key mapping issues.
Uses strict=True and verifies inference before uploading.
"""
import os, sys, json, torch
from huggingface_hub import HfApi, create_repo

repo = os.environ.get("HF_REPO", "stukenov/sozkz-core-qwen-500m-kk-base-v1")
token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set"); sys.exit(1)

CKPT_BASE = "/root/checkpoints/exp036_500m"

print("Loading checkpoint...")
ckpt = torch.load(f"{CKPT_BASE}/final/model.pt", map_location="cpu", weights_only=True)
results = json.load(open(f"{CKPT_BASE}/final/results.json"))
print(f"Results: {json.dumps(results, indent=2)}")

# Strip _orig_mod. prefix (torch.compile artifact)
cleaned = {}
for k, v in ckpt.items():
    cleaned[k.removeprefix("_orig_mod.")] = v
if any(k.startswith("_orig_mod.") for k in ckpt):
    print(f"  Stripped _orig_mod. prefix from {sum(1 for k in ckpt if k.startswith('_orig_mod.'))} keys")
ckpt = cleaned

# Map custom keys to HF Qwen2 format
print("Mapping state dict to HF Qwen2 format...")
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

print("Creating HF Qwen2 model...")
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

config = Qwen2Config(
    vocab_size=100000,
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    max_position_embeddings=32768,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_dropout=0.0,
    bos_token_id=0,
    eos_token_id=0,
    pad_token_id=1,
    torch_dtype="bfloat16",
)
model = Qwen2ForCausalLM(config)

# CRITICAL: strict=True -- fail on ANY missing/unexpected keys (exp028 lesson)
print("Loading state dict with strict=True...")
missing, unexpected = model.load_state_dict(state, strict=True)
assert not missing, f"FATAL: Missing keys: {missing}"
assert not unexpected, f"FATAL: Unexpected keys: {unexpected}"
print("  All keys matched perfectly!")

# ============================================================
# VERIFICATION: Run inference BEFORE uploading (exp028 lesson)
# ============================================================
print("\n=== VERIFICATION: Testing inference before upload ===")
model = model.to(dtype=torch.bfloat16, device="cuda")

tok_file = hf_hub_download("stukenov/sozkz-morphbpe-100k-kk-v1", "tokenizer.json")
tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
tok.pad_token_id = 1

test_prompts = [
    "Kazakstan Prezidenti",
    "Bilim beru zhuyesi",
    "Almaty qalasynda bugin",
]

all_ok = True
for prompt in test_prompts:
    ids = tok.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {text[:200]}")

    # Check output is not garbage (has reasonable token diversity)
    unique_tokens = len(set(out[0].tolist()))
    if unique_tokens < 5:
        print(f"  WARNING: Low token diversity ({unique_tokens} unique) -- model may be broken!")
        all_ok = False
    else:
        print(f"  OK: {unique_tokens} unique tokens")

if not all_ok:
    print(f"\nFATAL: Inference verification FAILED. NOT uploading.")
    print(f"Checkpoint preserved at {CKPT_BASE}/final/")
    sys.exit(1)

print("\n=== Inference verification PASSED ===\n")

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

print(f"\nDone! https://huggingface.co/{repo}")
