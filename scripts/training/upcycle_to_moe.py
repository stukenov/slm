"""Sparse upcycling: convert a dense Llama model into a Mixtral MoE model.

Each MLP is duplicated into N experts; attention weights are copied 1-to-1.
Router weights are randomly initialized.
"""

import argparse
import logging
import torch
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def upcycle(source: str, output: str, num_experts: int = 8, top_k: int = 2) -> None:
    logger.info("Loading dense model from %s", source)
    dense = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16)
    dense_config = dense.config
    tokenizer = AutoTokenizer.from_pretrained(source)

    logger.info("Dense model: %.2fM params", dense.num_parameters() / 1e6)

    # Build Mixtral config from dense Llama config
    mixtral_config = AutoConfig.for_model(
        "mixtral",
        vocab_size=dense_config.vocab_size,
        hidden_size=dense_config.hidden_size,
        intermediate_size=dense_config.intermediate_size,
        num_hidden_layers=dense_config.num_hidden_layers,
        num_attention_heads=dense_config.num_attention_heads,
        num_key_value_heads=getattr(dense_config, "num_key_value_heads", dense_config.num_attention_heads),
        max_position_embeddings=dense_config.max_position_embeddings,
        rms_norm_eps=getattr(dense_config, "rms_norm_eps", 1e-5),
        tie_word_embeddings=getattr(dense_config, "tie_word_embeddings", False),
        bos_token_id=dense_config.bos_token_id,
        eos_token_id=dense_config.eos_token_id,
        pad_token_id=dense_config.pad_token_id,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        router_aux_loss_coef=0.001,
        rope_theta=getattr(dense_config, "rope_theta", 10000.0),
    )

    logger.info("Creating Mixtral model with %d experts, top-%d routing", num_experts, top_k)
    moe = AutoModelForCausalLM.from_config(mixtral_config)
    moe = moe.to(dtype=torch.bfloat16)

    dense_sd = dense.state_dict()
    moe_sd = moe.state_dict()

    # Copy embeddings and final norm
    for key in list(moe_sd.keys()):
        if "layers." not in key:
            # embed_tokens, lm_head, model.norm
            if key in dense_sd:
                logger.debug("Copy %s", key)
                moe_sd[key] = dense_sd[key].clone()

    # Copy per-layer weights
    num_layers = dense_config.num_hidden_layers
    for layer_idx in range(num_layers):
        dp = f"model.layers.{layer_idx}."

        # Attention weights (direct copy)
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]:
            src_key = dp + suffix
            dst_key = dp + suffix
            if src_key in dense_sd and dst_key in moe_sd:
                moe_sd[dst_key] = dense_sd[src_key].clone()

        # MLP -> Experts: duplicate dense MLP into each expert
        gate_proj = dense_sd[dp + "mlp.gate_proj.weight"]  # [inter, hidden]
        up_proj = dense_sd[dp + "mlp.up_proj.weight"]      # [inter, hidden]
        down_proj = dense_sd[dp + "mlp.down_proj.weight"]   # [hidden, inter]

        # gate_up_proj: [num_experts, 2*inter, hidden]
        gate_up = torch.cat([gate_proj, up_proj], dim=0)  # [2*inter, hidden]
        gate_up_stacked = gate_up.unsqueeze(0).expand(num_experts, -1, -1).clone()

        # down_proj: [num_experts, hidden, inter]
        down_stacked = down_proj.unsqueeze(0).expand(num_experts, -1, -1).clone()

        moe_key_prefix = dp + "mlp."
        moe_sd[moe_key_prefix + "experts.gate_up_proj"] = gate_up_stacked
        moe_sd[moe_key_prefix + "experts.down_proj"] = down_stacked

        # Router: random init (already initialized by from_config), just leave it
        router_key = moe_key_prefix + "gate.weight"
        if router_key in moe_sd:
            logger.debug("Router layer %d: random init [%s]", layer_idx, moe_sd[router_key].shape)

    moe.load_state_dict(moe_sd)

    logger.info("MoE model: %.2fM params", moe.num_parameters() / 1e6)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    moe.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Sparse upcycling: Llama → Mixtral MoE")
    parser.add_argument("--source", required=True, help="Source dense model (HF hub or local path)")
    parser.add_argument("--output", required=True, help="Output directory for MoE model")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    upcycle(args.source, args.output, args.num_experts, args.top_k)


if __name__ == "__main__":
    main()
