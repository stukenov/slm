"""Upload trained 300M model to HuggingFace Hub."""
import os, sys, json, torch
from huggingface_hub import HfApi, create_repo

repo = os.environ.get("HF_REPO", "stukenov/sozkz-core-llama-300m-kk-base-v1")
token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set"); sys.exit(1)

# Load checkpoint
print("Loading checkpoint...")
ckpt = torch.load("/root/checkpoints/final/model.pt", map_location="cpu", weights_only=True)
results = json.load(open("/root/checkpoints/final/results.json"))
print(f"Results: {json.dumps(results, indent=2)}")

# Strip _orig_mod. prefix if present (torch.compile artifact)
cleaned = {}
for k, v in ckpt.items():
    cleaned[k.removeprefix("_orig_mod.")] = v
if any(k.startswith("_orig_mod.") for k in ckpt):
    print(f"  Stripped _orig_mod. prefix from {sum(1 for k in ckpt if k.startswith('_orig_mod.'))} keys")
ckpt = cleaned

# Map custom keys to HF Llama format
print("Mapping state dict to HF format...")
state = {}
key_map = {
    "emb.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "lm_head.weight",
}
for k, v in ckpt.items():
    # Skip rotary embedding buffers (HF Llama computes them on the fly)
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

# Save as safetensors via transformers
print("Creating HF model...")
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

config = LlamaConfig(
    vocab_size=50257, hidden_size=1024, intermediate_size=3584,
    num_hidden_layers=18, num_attention_heads=16, num_key_value_heads=16,
    max_position_embeddings=2048, tie_word_embeddings=True,
    bos_token_id=0, eos_token_id=0, pad_token_id=1,
    torch_dtype="bfloat16",
)
model = LlamaForCausalLM(config)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"  Missing keys: {missing}")
if unexpected:
    print(f"  Unexpected keys: {unexpected}")

# Create repo and upload
print(f"Uploading to {repo}...")
create_repo(repo, token=token, exist_ok=True)
model.push_to_hub(repo, token=token)

# Upload tokenizer
print("Uploading tokenizer...")
tok = AutoTokenizer.from_pretrained("stukenov/sozkz-core-gpt2-50k-kk-base-v1")
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

print(f"Done! https://huggingface.co/{repo}")
