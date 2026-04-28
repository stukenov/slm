#!/usr/bin/env python3
"""Inference with patched QK-Norm (weight=ones) to recover lost normalization."""
import torch
import torch.nn.functional as F
import json
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

MODEL = "stukenov/sozkz-core-llama-1b-kk-base-v1"

PROMPTS = [
    ("Саясат", "Қазақстан Президенті"),
    ("Экономика", "Экономика министрлігі"),
    ("Білім", "Білім беру жүйесі"),
    ("Ауа райы", "Ауа райы болжамы бойынша"),
    ("Тарих", "Қазақ халқының тарихы"),
    ("Технология", "Жасанды интеллект технологиясы"),
    ("Спорт", "Футбол чемпионаты"),
    ("Денсаулық", "Денсаулық сақтау министрлігі"),
    ("Күнделікті", "Алматы қаласында бүгін"),
    ("Ғылым", "Қазіргі заманғы физика ғылымында"),
    ("Әдебиет", "Алыстағы ауылда бір кәрі шал тұратын,"),
    ("Жалпы", "Қазақстан — бұл"),
]


def rms_norm(x, eps=1e-5):
    """RMSNorm with weight=ones (no learned parameters)."""
    return F.rms_norm(x, (x.size(-1),), None, eps)


def patch_attention_with_qk_norm(model):
    """Monkey-patch all attention layers to apply RMSNorm on Q,K after projection."""
    from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb

    # rotary_emb is on the model level in newer transformers
    rotary_emb = model.model.rotary_emb if hasattr(model.model, "rotary_emb") else None

    for layer in model.model.layers:
        attn = layer.self_attn

        # Check if rotary_emb is on attn or model level
        layer_rotary = getattr(attn, "rotary_emb", rotary_emb)

        def make_patched_forward(self_attn, rot_emb):
            def patched_forward(hidden_states, **kwargs):
                bsz, q_len, _ = hidden_states.size()

                q = self_attn.q_proj(hidden_states)
                k = self_attn.k_proj(hidden_states)
                v = self_attn.v_proj(hidden_states)

                num_heads = self_attn.config.num_attention_heads
                num_kv_heads = self_attn.config.num_key_value_heads
                head_dim = self_attn.head_dim

                q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

                # QK-Norm (weight=ones approximation)
                q = rms_norm(q)
                k = rms_norm(k)

                # Rotary embeddings
                position_ids = kwargs.get("position_ids")
                if position_ids is None:
                    position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
                cos, sin = rot_emb(v, position_ids)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                # GQA expand
                if num_kv_heads < num_heads:
                    k = repeat_kv(k, num_heads // num_kv_heads)
                    v = repeat_kv(v, num_heads // num_kv_heads)

                attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                attn_output = self_attn.o_proj(attn_output)

                return attn_output, None
            return patched_forward

        attn.forward = make_patched_forward(attn, layer_rotary)

    print("Patched %d attention layers with QK-Norm" % len(model.model.layers))


print("Loading model...")
tok_file = hf_hub_download(MODEL, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
tokenizer.pad_token_id = 1

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
params = sum(p.numel() for p in model.parameters()) / 1e6
print("Loaded: %.1fM params" % params)

# Patch with QK-Norm
patch_attention_with_qk_norm(model)

results = []
for cat, prompt in PROMPTS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=200, do_sample=True,
            temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    results.append({"category": cat, "prompt": prompt, "generation": text})
    print("\n=== [%s] %s ===" % (cat, prompt))
    print(text[:400])

with open("/root/slm/results/eval_1b_patched.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved %d generations" % len(results))
