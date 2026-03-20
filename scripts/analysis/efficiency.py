"""GPU inference efficiency benchmarking for SLM paper.

Measures inference latency, throughput (tok/s), and peak GPU memory
for all HuggingFace models in the registry. Designed to run on A10 GPU
(ssh kaznu) for consistent, comparable measurements.

Usage:
    # Full benchmark (run on A10 GPU):
    python scripts/analysis/efficiency.py

    # Specific models only:
    python scripts/analysis/efficiency.py --models sozkz-50m sozkz-150m qwen-0.5b

    # Generate table from existing results:
    python scripts/analysis/efficiency.py --table-only

Notes:
    - Must run on A10 GPU (ssh kaznu) for consistent measurements
    - Uses bf16 (not fp16) per CLAUDE.md
    - Quantized models (gemma-9b, llama-3-8b, qwen-7b, mistral-7b) measured as-configured
    - Clear CUDA cache between models to avoid memory contamination
    - API model (gpt-oss-120b) excluded from efficiency measurement
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from scripts.analysis.config import MODEL_ORDER, PARAM_SIZES
from scripts.eval.model_registry import MODEL_REGISTRY, list_models, load_model


def measure_efficiency(
    model,
    tokenizer,
    prompt: str = "Kazaqstan Respwblikasy",
    max_new_tokens: int = 128,
    warmup: int = 5,
    repeats: int = 20,
    device: str = "cuda",
) -> dict:
    """Measure inference latency, throughput, and peak GPU memory.

    Args:
        model: HuggingFace causal LM model.
        tokenizer: Corresponding tokenizer.
        prompt: Input text for generation.
        max_new_tokens: Number of tokens to generate.
        warmup: Number of warmup iterations (not timed).
        repeats: Number of timed iterations.
        device: Target device.

    Returns:
        Dict with latency_ms, throughput_tok_s, peak_memory_mb,
        generated_tokens, repeats, std_ms.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Warmup loop
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

    # Reset memory stats after warmup
    torch.cuda.reset_peak_memory_stats()

    # Timing loop
    latencies: list[float] = []
    generated_tokens = 0
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        generated_tokens = output.shape[1] - input_ids.shape[1]

    median_latency = sorted(latencies)[len(latencies) // 2]
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        "latency_ms": round(median_latency * 1000, 1),
        "throughput_tok_s": round(generated_tokens / median_latency, 1),
        "peak_memory_mb": round(peak_memory_mb, 0),
        "generated_tokens": int(generated_tokens),
        "repeats": repeats,
        "std_ms": round(float(np.std(latencies)) * 1000, 1),
    }


def run_efficiency_benchmark(
    models: list[str] | None = None,
    output_path: str = "paper/results/efficiency.json",
    warmup: int = 5,
    repeats: int = 20,
    max_new_tokens: int = 128,
) -> dict:
    """Run inference benchmarks for all HF models.

    Args:
        models: List of model keys to benchmark. None = all HF models.
        output_path: Path to save JSON results.
        warmup: Number of warmup iterations.
        repeats: Number of timed iterations.
        max_new_tokens: Tokens to generate per iteration.

    Returns:
        Dict mapping model_key -> benchmark results.
    """
    if models is None:
        models = list_models("hf")

    results: dict[str, dict] = {}

    for model_key in models:
        print(f"Benchmarking {model_key}...")
        model, tokenizer = load_model(model_key, device="cuda")

        if model is None:
            print(f"  Skipping {model_key} (API model)")
            continue

        result = measure_efficiency(
            model,
            tokenizer,
            warmup=warmup,
            repeats=repeats,
            max_new_tokens=max_new_tokens,
        )
        result["model"] = model_key
        result["params"] = MODEL_REGISTRY[model_key]["params"]
        results[model_key] = result

        # Clean up to avoid memory contamination
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print(
            f"  {model_key}: {result['latency_ms']}ms, "
            f"{result['throughput_tok_s']} tok/s, "
            f"{result['peak_memory_mb']}MB"
        )

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return results


def generate_efficiency_table(
    results: dict | None = None,
    input_path: str = "paper/results/efficiency.json",
    output_dir: str = "paper/tables",
) -> str:
    """Generate LaTeX efficiency comparison table.

    Args:
        results: Pre-loaded results dict. If None, loads from input_path.
        input_path: Path to efficiency.json.
        output_dir: Directory to write efficiency.tex.

    Returns:
        Path to the generated .tex file.
    """
    if results is None:
        with open(input_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    # Build rows in MODEL_ORDER, skipping models not in results
    rows: list[dict] = []
    for key in MODEL_ORDER:
        if key not in results:
            continue
        r = results[key]
        rows.append(
            {
                "Model": key,
                "Params": PARAM_SIZES.get(key, 0),
                "Latency (ms)": r["latency_ms"],
                "Throughput (tok/s)": r["throughput_tok_s"],
                "Peak Memory (MB)": r["peak_memory_mb"],
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("No results to generate table from.")
        return ""

    # Bold best values
    best_latency = df["Latency (ms)"].min()
    best_throughput = df["Throughput (tok/s)"].max()
    best_memory = df["Peak Memory (MB)"].min()

    def bold_if_best(val, best, lower_is_better=True):
        """Wrap value in \\textbf{} if it equals the best."""
        if (lower_is_better and val == best) or (not lower_is_better and val == best):
            return f"\\textbf{{{val}}}"
        return str(val)

    df["Latency (ms)"] = df["Latency (ms)"].apply(
        lambda x: bold_if_best(x, best_latency, lower_is_better=True)
    )
    df["Throughput (tok/s)"] = df["Throughput (tok/s)"].apply(
        lambda x: bold_if_best(x, best_throughput, lower_is_better=False)
    )
    df["Peak Memory (MB)"] = df["Peak Memory (MB)"].apply(
        lambda x: bold_if_best(x, best_memory, lower_is_better=True)
    )

    # Format params column (e.g., 50 -> "50M")
    df["Params"] = df["Params"].apply(
        lambda x: f"{x / 1000:.1f}B" if x >= 1000 else f"{x}M"
    )

    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrr",
        caption="Inference efficiency on NVIDIA A10 (bf16, 128 generated tokens).",
        label="tab:efficiency",
    )

    # Add auto-generated comment header
    output = f"% Auto-generated by scripts/analysis/efficiency.py\n{latex}"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "efficiency.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Table saved to {output_path}")

    return output_path


def main():
    """CLI entry point for efficiency benchmarking."""
    parser = argparse.ArgumentParser(
        description="GPU inference efficiency benchmarking for SLM paper"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to benchmark (default: all HF models)",
    )
    parser.add_argument(
        "--output",
        default="paper/results/efficiency.json",
        help="Output path for JSON results (default: paper/results/efficiency.json)",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Skip benchmarking, just generate table from existing JSON",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate per iteration (default: 128)",
    )
    args = parser.parse_args()

    if args.table_only:
        generate_efficiency_table(input_path=args.output)
    else:
        results = run_efficiency_benchmark(
            models=args.models,
            output_path=args.output,
            warmup=args.warmup,
            repeats=args.repeats,
            max_new_tokens=args.max_new_tokens,
        )
        generate_efficiency_table(results=results)


if __name__ == "__main__":
    main()
