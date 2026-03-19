"""Analyze MoE router specialization: domain×expert affinity heatmap.

Runs test data from each domain through the model, collects router logits,
and produces a domain×expert affinity matrix and utilization stats.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_router_decisions(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Forward pass collecting router logits from each MoE layer."""
    router_logits_all = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Mixtral gate outputs logits of shape [batch*seq, num_experts]
            if isinstance(output, torch.Tensor):
                router_logits_all.append((layer_idx, output.detach().cpu()))
        return hook_fn

    # Register hooks on router (gate) modules
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            h = layer.mlp.gate.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return router_logits_all


def analyze(
    model_path: str,
    dataset_path: str,
    max_samples_per_domain: int = 100,
    output_dir: str = "outputs/moe_analysis",
):
    logger.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    num_experts = model.config.num_local_experts
    num_layers = model.config.num_hidden_layers

    # Load dataset
    logger.info("Loading dataset from %s", dataset_path)
    try:
        ds = load_from_disk(dataset_path)
        if hasattr(ds, "keys"):
            ds = ds["validation"] if "validation" in ds else ds["train"]
    except Exception:
        ds = load_dataset(dataset_path, split="validation")

    if "domain_id" not in ds.column_names:
        logger.error("Dataset has no domain_id column. Cannot analyze per-domain routing.")
        return

    # Group by domain
    domain_indices = defaultdict(list)
    for idx, dom in enumerate(ds["domain_id"]):
        domain_indices[dom].append(idx)

    domains = sorted(domain_indices.keys())
    logger.info("Found %d domains: %s", len(domains), domains)

    # Affinity matrix: [num_domains, num_experts] — average router probability
    affinity = np.zeros((len(domains), num_experts))
    expert_total_counts = np.zeros(num_experts)

    for dom_idx, dom in enumerate(domains):
        indices = domain_indices[dom][:max_samples_per_domain]
        logger.info("Processing domain %d: %d samples", dom, len(indices))

        domain_expert_probs = np.zeros(num_experts)
        n_tokens = 0

        for i in indices:
            input_ids = torch.tensor([ds[i]["input_ids"][:512]], device=device)
            router_data = collect_router_decisions(model, input_ids)

            for layer_idx, logits in router_data:
                probs = torch.softmax(logits.float(), dim=-1).numpy()
                domain_expert_probs += probs.sum(axis=0)
                n_tokens += probs.shape[0]

        if n_tokens > 0:
            domain_expert_probs /= n_tokens
            affinity[dom_idx] = domain_expert_probs
            expert_total_counts += domain_expert_probs * n_tokens

    # Print results
    logger.info("\n=== Domain × Expert Affinity Matrix ===")
    header = "Domain\t" + "\t".join(f"E{i}" for i in range(num_experts))
    logger.info(header)
    for dom_idx, dom in enumerate(domains):
        row = f"D{dom}\t" + "\t".join(f"{affinity[dom_idx, e]:.3f}" for e in range(num_experts))
        logger.info(row)

    # Expert utilization
    logger.info("\n=== Expert Utilization ===")
    total = expert_total_counts.sum()
    for e in range(num_experts):
        pct = 100 * expert_total_counts[e] / total if total > 0 else 0
        logger.info("Expert %2d: %.1f%%", e, pct)

    # Save to file
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "affinity_matrix.npy", affinity)
    np.save(out / "expert_utilization.npy", expert_total_counts)

    # Save as CSV for easy viewing
    import csv
    with open(out / "affinity_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain"] + [f"expert_{i}" for i in range(num_experts)])
        for dom_idx, dom in enumerate(domains):
            writer.writerow([dom] + [f"{affinity[dom_idx, e]:.4f}" for e in range(num_experts)])

    logger.info("Results saved to %s", out)

    # Try to generate heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(affinity, cmap="YlOrRd", aspect="auto")
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Domain ID")
        ax.set_xticks(range(num_experts))
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels([f"D{d}" for d in domains])
        plt.colorbar(im, label="Avg Router Probability")
        plt.title("Domain × Expert Affinity")
        plt.tight_layout()
        plt.savefig(out / "affinity_heatmap.png", dpi=150)
        logger.info("Heatmap saved to %s", out / "affinity_heatmap.png")
    except ImportError:
        logger.warning("matplotlib not available, skipping heatmap")


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE routing specialization")
    parser.add_argument("--model", required=True, help="Path to MoE model")
    parser.add_argument("--dataset", required=True, help="Path to domain-labeled dataset")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", default="outputs/moe_analysis")
    args = parser.parse_args()

    analyze(args.model, args.dataset, args.max_samples, args.output)


if __name__ == "__main__":
    main()
