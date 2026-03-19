"""Upcycle a dense Llama model into a Mixtral-style MoE with shared router.

Usage:
    python -m slm.moe_upcycle \
        --dense-model stukenov/sozkz-core-llama-150m-kk-base-v1 \
        --output-dir ./outputs/moe_init \
        --num-experts 128 \
        --num-experts-per-tok 2 \
        --expert-intermediate-size 640 \
        [--push-to-hub stukenov/sozkz-moe-mix-3b-kk-base-v1-init]
"""
from __future__ import annotations

import argparse
import logging

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MixtralConfig,
    MixtralForCausalLM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def upcycle_to_moe(
    dense_model_name: str,
    num_experts: int = 128,
    num_experts_per_tok: int = 2,
    expert_intermediate_size: int = 640,
    router_jitter_noise: float = 0.1,
    router_aux_loss_coef: float = 0.01,
    noise_std: float = 0.01,
) -> tuple[MixtralForCausalLM, AutoTokenizer]:
    """Convert a dense Llama model to Mixtral MoE with shared router."""

    logger.info("Loading dense model: %s", dense_model_name)
    dense_model = AutoModelForCausalLM.from_pretrained(dense_model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(dense_model_name)
    dense_config = dense_model.config

    moe_config = MixtralConfig(
        vocab_size=dense_config.vocab_size,
        hidden_size=dense_config.hidden_size,
        intermediate_size=expert_intermediate_size,
        num_hidden_layers=dense_config.num_hidden_layers,
        num_attention_heads=dense_config.num_attention_heads,
        num_key_value_heads=getattr(dense_config, "num_key_value_heads", dense_config.num_attention_heads),
        max_position_embeddings=dense_config.max_position_embeddings,
        rms_norm_eps=getattr(dense_config, "rms_norm_eps", 1e-5),
        tie_word_embeddings=getattr(dense_config, "tie_word_embeddings", True),
        pad_token_id=dense_config.pad_token_id,
        bos_token_id=dense_config.bos_token_id,
        eos_token_id=dense_config.eos_token_id,
        num_local_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        output_router_logits=True,
        router_aux_loss_coef=router_aux_loss_coef,
        router_jitter_noise=router_jitter_noise,
    )

    logger.info("Creating MoE model: %d experts, top-%d, intermediate=%d",
                num_experts, num_experts_per_tok, expert_intermediate_size)
    moe_model = MixtralForCausalLM(moe_config)

    # Copy embeddings
    moe_model.model.embed_tokens.weight.data.copy_(dense_model.model.embed_tokens.weight.data)
    if hasattr(moe_model, "lm_head") and not moe_config.tie_word_embeddings:
        moe_model.lm_head.weight.data.copy_(dense_model.lm_head.weight.data)

    # Copy final norm
    moe_model.model.norm.weight.data.copy_(dense_model.model.norm.weight.data)

    # Copy per-layer weights
    for layer_idx in range(dense_config.num_hidden_layers):
        dense_layer = dense_model.model.layers[layer_idx]
        moe_layer = moe_model.model.layers[layer_idx]

        # Attention
        moe_layer.self_attn.q_proj.weight.data.copy_(dense_layer.self_attn.q_proj.weight.data)
        moe_layer.self_attn.k_proj.weight.data.copy_(dense_layer.self_attn.k_proj.weight.data)
        moe_layer.self_attn.v_proj.weight.data.copy_(dense_layer.self_attn.v_proj.weight.data)
        moe_layer.self_attn.o_proj.weight.data.copy_(dense_layer.self_attn.o_proj.weight.data)

        # Layer norms
        moe_layer.input_layernorm.weight.data.copy_(dense_layer.input_layernorm.weight.data)
        moe_layer.post_attention_layernorm.weight.data.copy_(dense_layer.post_attention_layernorm.weight.data)

        # Expert FFN: Xavier init + noise for symmetry breaking
        # transformers 5.x uses fused MixtralExperts with batched tensors:
        #   experts.gate_up_proj: [num_experts, intermediate*2, hidden]
        #   experts.down_proj: [num_experts, hidden, intermediate]
        experts = moe_layer.mlp.experts
        nn.init.xavier_uniform_(experts.gate_up_proj.data.view(-1, experts.gate_up_proj.shape[-1]))
        nn.init.xavier_uniform_(experts.down_proj.data.view(-1, experts.down_proj.shape[-1]))
        # Add per-expert noise for symmetry breaking
        experts.gate_up_proj.data += torch.randn_like(experts.gate_up_proj) * noise_std
        experts.down_proj.data += torch.randn_like(experts.down_proj) * noise_std

    # Shared router: all layers share layer 0's gate
    # transformers 5.x: gate is MixtralTopKRouter with .weight parameter
    shared_gate = moe_model.model.layers[0].mlp.gate
    nn.init.xavier_uniform_(shared_gate.weight.data.unsqueeze(0))
    for layer_idx in range(1, dense_config.num_hidden_layers):
        moe_model.model.layers[layer_idx].mlp.gate = shared_gate

    total_params = sum(p.numel() for p in moe_model.parameters())
    unique_params = sum(p.numel() for p in set(moe_model.parameters()))
    logger.info("Total params (with sharing): %.2fB", total_params / 1e9)
    logger.info("Unique params: %.2fB", unique_params / 1e9)

    return moe_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Upcycle dense Llama to MoE")
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--output-dir", default="./outputs/moe_init")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--expert-intermediate-size", type=int, default=640)
    parser.add_argument("--push-to-hub", default=None)
    args = parser.parse_args()

    model, tokenizer = upcycle_to_moe(
        dense_model_name=args.dense_model,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        expert_intermediate_size=args.expert_intermediate_size,
    )

    # Register shared router weights as tied so save_pretrained works
    tied_keys = [f"model.layers.{i}.mlp.gate.weight"
                 for i in range(1, model.config.num_hidden_layers)]
    existing = getattr(model, "_tied_weights_keys", None)
    if isinstance(existing, dict):
        for k in tied_keys:
            existing[k] = "model.layers.0.mlp.gate.weight"
    elif isinstance(existing, (list, set)):
        model._tied_weights_keys = list(existing) + tied_keys
    else:
        model._tied_weights_keys = tied_keys

    logger.info("Saving to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        logger.info("Pushing to %s", args.push_to_hub)
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        logger.info("Upload complete!")


if __name__ == "__main__":
    main()
