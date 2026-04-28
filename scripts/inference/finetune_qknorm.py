#!/usr/bin/env python3
"""Recover lost QK-Norm weights by fine-tuning only 5,632 params.
Loads original model (pre-fix commit), adds QK-Norm, freezes everything else,
trains on small data, bakes embedding scaling, saves final model.
"""
import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
from datasets import load_dataset

MODEL = "stukenov/sozkz-core-llama-1b-kk-base-v1"
ORIGINAL_REV = "920f6ec52c36"
TOKEN = open("/root/.cache/huggingface/token").read().strip()
EMB_SCALE = math.sqrt(2048)
DEVICE = "cuda"
LR = 1e-3
STEPS = 2000
BATCH_SIZE = 4
SEQ_LEN = 512
LOG_EVERY = 100


class QKNormWrapper(nn.Module):
    def __init__(self, original_attn, head_dim, rotary_emb_fn):
        super().__init__()
        self.attn = original_attn
        self.rotary_emb_fn = rotary_emb_fn
        self.q_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(head_dim))
        self.head_dim = head_dim
        self.config = original_attn.config

    def _rms_norm(self, x, weight):
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + 1e-5)
        return (x * weight.float()).to(orig_dtype)

    def forward(self, hidden_states, **kwargs):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        bsz, q_len, _ = hidden_states.size()
        nh = self.config.num_attention_heads
        nkv = self.config.num_key_value_heads
        hd = self.head_dim

        q = self.attn.q_proj(hidden_states).view(bsz, q_len, nh, hd).transpose(1, 2)
        k = self.attn.k_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1, 2)
        v = self.attn.v_proj(hidden_states).view(bsz, q_len, nkv, hd).transpose(1, 2)

        q = self._rms_norm(q, self.q_norm_weight)
        k = self._rms_norm(k, self.k_norm_weight)

        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb_fn(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if nkv < nh:
            k = repeat_kv(k, nh // nkv)
            v = repeat_kv(v, nh // nkv)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.attn.o_proj(y), None


def main():
    print("Loading original model (rev %s)..." % ORIGINAL_REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=ORIGINAL_REV, torch_dtype=torch.bfloat16
    )
    print("Loaded: %.1fM params" % (sum(p.numel() for p in model.parameters()) / 1e6))

    print("Baking embedding scale (x%.2f)..." % EMB_SCALE)
    with torch.no_grad():
        model.model.embed_tokens.weight.mul_(EMB_SCALE)

    hd = model.config.hidden_size // model.config.num_attention_heads
    rotary = getattr(model.model, "rotary_emb", None)

    print("Adding QK-Norm wrappers (head_dim=%d)..." % hd)
    for layer in model.model.layers:
        rot = getattr(layer.self_attn, "rotary_emb", rotary)
        layer.self_attn = QKNormWrapper(layer.self_attn, hd, rot)

    model = model.to(DEVICE)

    trainable = 0
    for name, param in model.named_parameters():
        if "q_norm_weight" in name or "k_norm_weight" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    print("Trainable: %d params" % trainable)

    print("Loading training data...")
    tok_file = hf_hub_download(MODEL, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1

    ds = load_dataset("stukenov/sozkz-corpus-clean-kk-text-v4", split="train", streaming=True)
    texts = []
    for row in ds:
        t = row.get("text", "")
        if t and len(t) > 50:
            texts.append(t)
        if len(texts) >= STEPS * BATCH_SIZE:
            break
    print("Collected %d texts" % len(texts))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01
    )
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for step in range(STEPS):
        batch = texts[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        if not batch:
            break
        enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=SEQ_LEN, padding=True)
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss = 0.9 * running_loss + 0.1 * out.loss.item() if running_loss > 0 else out.loss.item()
        if (step + 1) % LOG_EVERY == 0:
            print("  step %d/%d | loss %.4f | %.1fs" % (step + 1, STEPS, running_loss, time.time() - t0))

    print("\nTraining done in %.1fs" % (time.time() - t0))

    print("\nInference test...")
    model.eval()
    prompts = [
        "Қазақстан Президенті",
        "Білім беру жүйесі",
        "Алматы қаласында бүгін",
        "Қазақ халқының тарихы",
        "Жасанды интеллект технологиясы",
    ]
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
        print("\n=== %s ===" % prompt)
        print(tokenizer.decode(o[0], skip_special_tokens=True)[:400])

    save_dir = "/root/slm/results/exp028_1b_recovered"
    os.makedirs(save_dir, exist_ok=True)
    qk = {}
    for name, param in model.named_parameters():
        if "q_norm_weight" in name or "k_norm_weight" in name:
            qk[name] = param.data.cpu()
    torch.save(qk, os.path.join(save_dir, "qknorm_weights.pt"))
    print("\nQK-Norm weights saved to %s" % save_dir)


if __name__ == "__main__":
    main()
