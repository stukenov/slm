"""Publish a training checkpoint to HuggingFace as a named branch.

Usage:
    python publish_checkpoint.py /root/checkpoints/exp028_1b/step_500 --step 500
    python publish_checkpoint.py /root/checkpoints/exp028_1b/final --step final

Each checkpoint becomes a branch: step-500, step-1000, etc.
Load with: LlamaForCausalLM.from_pretrained(repo, revision="step-500")
"""
import os, sys, json, torch, argparse, subprocess

def send_tg(msg):
    """Fire-and-forget Telegram notification."""
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({"chat_id": "47474471", "text": msg}).encode()
        urllib.request.urlopen(
            "https://api.telegram.org/bot8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g/sendMessage",
            data, timeout=10,
        )
    except Exception:
        pass

def publish(ckpt_dir: str, step: str, repo: str, token: str, skip_inference: bool = False):
    print(f"[publish] Loading checkpoint from {ckpt_dir}...")
    ckpt = torch.load(f"{ckpt_dir}/model.pt", map_location="cpu", weights_only=True)

    # Strip _orig_mod. prefix (torch.compile artifact)
    ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}

    # Map custom keys to HF Llama format
    key_map = {
        "emb.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "head.weight": "lm_head.weight",
    }
    state = {}
    for k, v in ckpt.items():
        if k.startswith("rot."):
            continue  # HF computes rotary internally
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

    # Build HF model
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from huggingface_hub import HfApi, hf_hub_download, create_repo

    config = LlamaConfig(
        vocab_size=50257, hidden_size=2048, intermediate_size=5504,
        num_hidden_layers=22, num_attention_heads=16, num_key_value_heads=4,
        max_position_embeddings=2048, tie_word_embeddings=True,
        bos_token_id=0, eos_token_id=0, pad_token_id=1,
        torch_dtype="bfloat16",
    )
    model = LlamaForCausalLM(config)

    # strict=True — catch any key mismatches
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing, f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    print(f"[publish] All keys matched.")

    # Optional inference verification (for final checkpoint)
    if not skip_inference and torch.cuda.is_available():
        print("[publish] Running inference verification...")
        model_gpu = model.to(dtype=torch.bfloat16, device="cuda")
        tok_file = hf_hub_download("stukenov/sozkz-core-gpt2-50k-kk-base-v1", "tokenizer.json")
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
        tok.pad_token_id = 1

        ids = tok.encode("Қазақстан Президенті", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model_gpu.generate(ids, max_new_tokens=30, do_sample=True, temperature=0.7)
        text = tok.decode(out[0], skip_special_tokens=True)
        cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        ratio = cyrillic / max(len(text), 1)
        print(f"[publish] Inference: '{text[:100]}' (cyrillic: {ratio:.0%})")
        if ratio < 0.3:
            send_tg(f"[exp028v2] WARNING: step {step} inference check failed (cyrillic {ratio:.0%})")
        model = model_gpu.cpu()
        del model_gpu
        torch.cuda.empty_cache()

    # Create repo and branch
    branch = f"step-{step}"
    api = HfApi(token=token)
    create_repo(repo, token=token, exist_ok=True)

    # Create branch (ignore if exists)
    try:
        api.create_branch(repo, branch=branch, token=token)
    except Exception:
        pass  # branch may already exist

    print(f"[publish] Pushing to {repo} branch={branch}...")
    model.push_to_hub(repo, token=token, revision=branch)

    # Push tokenizer too
    tok_file = hf_hub_download("stukenov/sozkz-core-gpt2-50k-kk-base-v1", "tokenizer.json")
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tok.pad_token_id = 1
    tok.push_to_hub(repo, token=token, revision=branch)

    # Push meta.json if exists
    meta_path = f"{ckpt_dir}/meta.json"
    if os.path.exists(meta_path):
        api.upload_file(
            path_or_fileobj=meta_path,
            path_in_repo="meta.json",
            repo_id=repo,
            token=token,
            revision=branch,
        )

    print(f"[publish] Done: https://huggingface.co/{repo}/tree/{branch}")
    send_tg(f"[exp028v2] Checkpoint step-{step} published to HF: https://huggingface.co/{repo}/tree/{branch}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", help="Path to checkpoint directory")
    parser.add_argument("--step", required=True, help="Step number or 'final'")
    parser.add_argument("--repo", default="stukenov/sozkz-core-llama-1b-kk-base-v1")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference check (for intermediate checkpoints)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try reading from file
        for p in ["~/.cache/huggingface/token", "/root/.cache/huggingface/token"]:
            p = os.path.expanduser(p)
            if os.path.exists(p):
                token = open(p).read().strip()
                break
    if not token:
        print("ERROR: HF_TOKEN not set"); sys.exit(1)

    publish(args.ckpt_dir, args.step, args.repo, token, args.skip_inference)
