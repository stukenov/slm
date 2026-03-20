"""Contamination checker: character-level n-gram overlap between training data and benchmarks.

Follows the GPT-3 methodology using character-level 13-grams to detect potential
data contamination between the training corpus and evaluation benchmarks.

Memory note: This script is memory-intensive. With 500K training samples and
13-char n-grams, expect ~50-100M unique n-grams consuming 5-10GB RAM.
Run on a machine with 16GB+ RAM.

Usage:
    python scripts/analysis/contamination.py --help
    python scripts/analysis/contamination.py --sample-size 100000  # quick test
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Callable

from datasets import load_dataset


# ---------------------------------------------------------------------------
# N-gram extraction
# ---------------------------------------------------------------------------


def extract_char_ngrams(text: str, n: int = 13) -> set[str]:
    """Extract all character-level n-grams from text.

    Args:
        text: Input text string.
        n: N-gram size (default 13, following GPT-3 methodology).

    Returns:
        Set of character n-grams. Empty set if text is shorter than n.
    """
    text = text.lower().strip()
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


# ---------------------------------------------------------------------------
# Training n-gram builder
# ---------------------------------------------------------------------------


def _detect_text_column(example: dict, preferred: str) -> str:
    """Detect the text column name from an example."""
    if preferred in example:
        return preferred
    if "text" in example:
        return "text"
    # Fallback: first string-valued key
    for key, value in example.items():
        if isinstance(value, str):
            return key
    raise KeyError(
        f"No text column found. Tried '{preferred}', 'text', and all keys: {list(example.keys())}"
    )


def build_training_ngrams(
    dataset_name: str,
    text_column: str = "text",
    n: int = 13,
    sample_size: int = 500_000,
    seed: int = 42,
) -> set[str]:
    """Build a set of character n-grams from a sample of the training corpus.

    Args:
        dataset_name: HuggingFace dataset identifier.
        text_column: Column containing text data.
        n: N-gram size.
        sample_size: Number of training examples to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        Set of all unique character n-grams found in the sample.
    """
    print(f"Loading training dataset: {dataset_name} (streaming, sample={sample_size})")
    ds = load_dataset(dataset_name, split="train", streaming=True)
    ds = ds.shuffle(seed=seed).take(sample_size)

    ngrams: set[str] = set()
    resolved_column: str | None = None
    count = 0

    for example in ds:
        if resolved_column is None:
            resolved_column = _detect_text_column(example, text_column)
            print(f"Using text column: '{resolved_column}'")

        text = example[resolved_column]
        if isinstance(text, str):
            ngrams.update(extract_char_ngrams(text, n))

        count += 1
        if count % 50_000 == 0:
            print(
                f"Processed {count}/{sample_size} training examples, "
                f"{len(ngrams):,} unique {n}-grams"
            )

    print(
        f"Done: {count} training examples, {len(ngrams):,} unique {n}-grams"
    )
    return ngrams


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------


def compute_ngram_overlap(
    train_ngrams: set[str],
    bench_texts: list[str],
    n: int = 13,
    contamination_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute n-gram overlap between training set and benchmark texts.

    Args:
        train_ngrams: Pre-computed set of training n-grams.
        bench_texts: List of benchmark text strings to check.
        n: N-gram size (must match train_ngrams).
        contamination_threshold: Fraction of n-grams overlapping to flag
            an example as contaminated.

    Returns:
        Dict with overlap statistics including contamination rate and
        overlap distribution (mean, median, p95, max).
    """
    overlaps: list[float] = []
    contaminated_count = 0
    skipped = 0

    for text in bench_texts:
        bench_ngrams = extract_char_ngrams(text, n)
        if len(bench_ngrams) == 0:
            skipped += 1
            continue

        overlap_ratio = len(bench_ngrams & train_ngrams) / len(bench_ngrams)
        overlaps.append(overlap_ratio)

        if overlap_ratio > contamination_threshold:
            contaminated_count += 1

    total = len(overlaps)

    if total == 0:
        return {
            "n": n,
            "char_level": True,
            "threshold": contamination_threshold,
            "train_ngrams_count": len(train_ngrams),
            "total_bench_examples": 0,
            "skipped_short": skipped,
            "contaminated_examples": 0,
            "contamination_rate": 0.0,
            "overlap_distribution": {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "max": 0.0,
            },
        }

    sorted_overlaps = sorted(overlaps)
    p95_idx = min(int(0.95 * len(sorted_overlaps)), len(sorted_overlaps) - 1)

    return {
        "n": n,
        "char_level": True,
        "threshold": contamination_threshold,
        "train_ngrams_count": len(train_ngrams),
        "total_bench_examples": total,
        "skipped_short": skipped,
        "contaminated_examples": contaminated_count,
        "contamination_rate": contaminated_count / total,
        "overlap_distribution": {
            "mean": statistics.mean(overlaps),
            "median": statistics.median(overlaps),
            "p95": sorted_overlaps[p95_idx],
            "max": max(overlaps),
        },
    }


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------


def _mc_qa_text(ex: dict) -> str:
    """Extract text from MC QA benchmark example."""
    question = ex.get("question", "")
    choices = ex.get("choices", [])
    if isinstance(choices, list):
        return question + " " + " ".join(str(c) for c in choices)
    return question


def _belebele_text(ex: dict) -> str:
    """Extract text from Belebele benchmark example."""
    return ex.get("flores_passage", "") + " " + ex.get("question", "")


def _sentiment_text(ex: dict) -> str:
    """Extract text from sentiment benchmark example."""
    return ex.get("text", "")


def _ner_text(ex: dict) -> str:
    """Extract text from NER benchmark example."""
    tokens = ex.get("tokens", [])
    if isinstance(tokens, list):
        return " ".join(str(t) for t in tokens)
    return str(tokens)


def _sib200_text(ex: dict) -> str:
    """Extract text from SIB-200 benchmark example."""
    return ex.get("text", "")


def _ner_filter(ex: dict) -> bool:
    """Filter NER examples for Kazakh language."""
    return ex.get("lang") == "kk"


def _sib200_filter(ex: dict) -> bool:
    """Filter SIB-200 examples for Kazakh language."""
    return ex.get("language") == "kaz_Cyrl"


BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "mc_qa": {
        "dataset": "kk-nlp/kk-socio-cultural-bench-mc",
        "split": "test",
        "text_fn": _mc_qa_text,
    },
    "belebele": {
        "dataset": "facebook/belebele",
        "split": "kaz_Cyrl",
        "text_fn": _belebele_text,
    },
    "sentiment": {
        "dataset": "kk-nlp/kazsandra",
        "split": "test",
        "text_fn": _sentiment_text,
    },
    "ner": {
        "dataset": "Babelscape/multinerd",
        "split": "test",
        "filter_fn": _ner_filter,
        "text_fn": _ner_text,
    },
    "sib200": {
        "dataset": "Davlan/sib200",
        "split": "test",
        "filter_fn": _sib200_filter,
        "text_fn": _sib200_text,
    },
}


# ---------------------------------------------------------------------------
# Main contamination check
# ---------------------------------------------------------------------------


def run_contamination_check(
    training_dataset: str = "kz-transformers/multidomain-kazakh-dataset",
    text_column: str = "text",
    sample_size: int = 500_000,
    n: int = 13,
    output_path: str = "paper/results/contamination.json",
) -> dict[str, Any]:
    """Run contamination check against all benchmarks.

    Args:
        training_dataset: HuggingFace dataset identifier for training data.
        text_column: Column name for text in training data.
        sample_size: Number of training examples to sample.
        n: N-gram size (13 = GPT-3 methodology).
        output_path: Path to save JSON results.

    Returns:
        Dict with per-benchmark contamination results.
    """
    # Build training n-grams (once)
    train_ngrams = build_training_ngrams(
        training_dataset, text_column, n, sample_size
    )

    results: dict[str, Any] = {
        "training_dataset": training_dataset,
        "training_sample_size": sample_size,
        "ngram_size": n,
        "benchmarks": {},
    }

    for bench_name, config in BENCHMARK_CONFIGS.items():
        print(f"\nChecking benchmark: {bench_name}")
        print(f"  Dataset: {config['dataset']}, split: {config['split']}")

        try:
            ds = load_dataset(config["dataset"], split=config["split"])
        except Exception as e:
            print(f"  ERROR loading {bench_name}: {e}")
            results["benchmarks"][bench_name] = {"error": str(e)}
            continue

        # Apply filter if present
        filter_fn: Callable | None = config.get("filter_fn")
        if filter_fn is not None:
            ds = ds.filter(filter_fn)
            print(f"  After filtering: {len(ds)} examples")

        # Extract texts
        text_fn: Callable = config["text_fn"]
        bench_texts = [text_fn(ex) for ex in ds]
        print(f"  Total texts: {len(bench_texts)}")

        # Compute overlap
        overlap = compute_ngram_overlap(train_ngrams, bench_texts, n)
        results["benchmarks"][bench_name] = overlap

        rate = overlap["contamination_rate"]
        contaminated = overlap["contaminated_examples"]
        total = overlap["total_bench_examples"]
        print(
            f"  {bench_name}: {rate * 100:.1f}% contaminated "
            f"({contaminated}/{total})"
        )

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for contamination checking."""
    parser = argparse.ArgumentParser(
        description="Check n-gram overlap contamination between training data and benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training-dataset",
        default="kz-transformers/multidomain-kazakh-dataset",
        help="HuggingFace dataset ID for training data",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name for text in training dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500_000,
        help="Number of training examples to sample",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=13,
        help="Character n-gram size (13 = GPT-3 methodology)",
    )
    parser.add_argument(
        "--output",
        default="paper/results/contamination.json",
        help="Output path for JSON results",
    )

    args = parser.parse_args()

    run_contamination_check(
        training_dataset=args.training_dataset,
        text_column=args.text_column,
        sample_size=args.sample_size,
        n=args.ngram_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
