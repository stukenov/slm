"""Drop-upcycling: convert dense Llama into Mixtral MoE with differentiated experts.

Key differences from v1 (upcycle_to_moe.py):
- 16 experts with top-4 routing (2 shared + top-2 from 14 routed)
- Drop-upcycling: random re-init of a fraction of expert weights to break symmetry
- Shared expert bias: experts 0-1 get large positive router bias to always be selected
"""

import argparse
import logging
import random

import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def upcycle_v2(
    source: str,
    output: str,
    num_experts: int = 16,
    top_k: int = 4,
    drop_ratio: float = 0.5,
    shared_experts: int = 2,
    shared_bias: float = 5.0,
    seed: int = 42,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Loading dense model from %s", source)
    dense = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16)
    dense_config = dense.config
    tokenizer = AutoTokenizer.from_pretrained(source)

    logger.info("Dense model: %.2fM params", dense.num_parameters() / 1e6)

    # Build Mixtral config
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

    logger.info(
        "Creating Mixtral MoE: %d experts, top-%d routing, %d shared experts",
        num_experts, top_k, shared_experts,
    )
    moe = AutoModelForCausalLM.from_config(mixtral_config)
    moe = moe.to(dtype=torch.bfloat16)

    dense_sd = dense.state_dict()
    moe_sd = moe.state_dict()

    # Copy non-layer weights (embeddings, final norm, lm_head)
    for key in list(moe_sd.keys()):
        if "layers." not in key and key in dense_sd:
            moe_sd[key] = dense_sd[key].clone()

    num_layers = dense_config.num_hidden_layers
    for layer_idx in range(num_layers):
        dp = f"model.layers.{layer_idx}."

        # Copy attention weights
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]:
            src_key = dp + suffix
            if src_key in dense_sd and src_key in moe_sd:
                moe_sd[src_key] = dense_sd[src_key].clone()

        # MLP -> Experts
        gate_proj = dense_sd[dp + "mlp.gate_proj.weight"]
        up_proj = dense_sd[dp + "mlp.up_proj.weight"]
        down_proj = dense_sd[dp + "mlp.down_proj.weight"]

        gate_up = torch.cat([gate_proj, up_proj], dim=0)  # [2*inter, hidden]
        gate_up_stacked = gate_up.unsqueeze(0).expand(num_experts, -1, -1).clone()
        down_stacked = down_proj.unsqueeze(0).expand(num_experts, -1, -1).clone()

        # Drop-upcycling: re-initialize a fraction of routed experts
        for expert_idx in range(shared_experts, num_experts):
            if random.random() < drop_ratio:
                logger.debug("Layer %d, Expert %d: re-initializing (drop-upcycling)", layer_idx, expert_idx)
                nn.init.kaiming_uniform_(gate_up_stacked[expert_idx])
                nn.init.kaiming_uniform_(down_stacked[expert_idx])

        moe_key_prefix = dp + "mlp."
        moe_sd[moe_key_prefix + "experts.gate_up_proj"] = gate_up_stacked
        moe_sd[moe_key_prefix + "experts.down_proj"] = down_stacked

        # Router: add bias for shared experts so they're always selected
        router_key = moe_key_prefix + "gate.weight"
        if router_key in moe_sd:
            router_w = moe_sd[router_key]
            # Add large positive bias to shared expert rows
            router_w[:shared_experts, :] += shared_bias
            moe_sd[router_key] = router_w
            logger.debug(
                "Layer %d router: shared bias %.1f applied to experts 0-%d",
                layer_idx, shared_bias, shared_experts - 1,
            )

    moe.load_state_dict(moe_sd)

    logger.info("MoE model: %.2fM params", moe.num_parameters() / 1e6)
    logger.info(
        "Architecture: %d experts, top-%d, %d shared (bias=%.1f), drop_ratio=%.2f",
        num_experts, top_k, shared_experts, shared_bias, drop_ratio,
    )

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    moe.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Drop-upcycling: Llama → Mixtral MoE v2")
    parser.add_argument("--source", required=True, help="Source dense model (HF hub or local)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--drop-ratio", type=float, default=0.5)
    parser.add_argument("--shared-experts", type=int, default=2)
    parser.add_argument("--shared-bias", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    upcycle_v2(
        args.source, args.output,
        num_experts=args.num_experts,
        top_k=args.top_k,
        drop_ratio=args.drop_ratio,
        shared_experts=args.shared_experts,
        shared_bias=args.shared_bias,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
