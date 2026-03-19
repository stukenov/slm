"""Model evaluation: perplexity, generation quality, Kazakh character ratio."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm.data import prepare_datasets

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Kazakh-specific Cyrillic characters (beyond standard Russian)
KAZAKH_SPECIFIC = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")
CYRILLIC_RANGE = re.compile(r"[\u0400-\u04FF]")


def compute_perplexity(
    model_path: str,
    dataset_name: str = "kz-transformers/multidomain-kazakh-dataset",
    block_size: int = 512,
    max_samples: int = 1000,
) -> float:
    """Compute perplexity on held-out validation data."""
    logger.info("Computing perplexity for %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets = prepare_datasets(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        block_size=block_size,
    )
    val_dataset = datasets["validation"]

    if len(val_dataset) > max_samples:
        val_dataset = val_dataset.select(range(max_samples))

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(val_dataset):
            input_ids = torch.tensor([sample["input_ids"]], device=device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

            if (i + 1) % 100 == 0:
                logger.info("Processed %d/%d samples", i + 1, len(val_dataset))

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    logger.info("Perplexity: %.2f (avg loss: %.4f)", perplexity, avg_loss)
    return perplexity


def generate_text(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> list[str]:
    """Generate text continuations for given prompts."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(text)
        logger.info("Prompt: %s...", prompt[:50])
        logger.info("Generated: %s...", text[:100])

    return results


def kazakh_char_ratio(text: str) -> float:
    """Compute ratio of Kazakh-specific characters to all Cyrillic characters."""
    cyrillic_chars = CYRILLIC_RANGE.findall(text)
    if not cyrillic_chars:
        return 0.0
    kazakh_count = sum(1 for c in cyrillic_chars if c in KAZAKH_SPECIFIC)
    return kazakh_count / len(cyrillic_chars)


def evaluate_model(
    model_path: str,
    prompts_file: str | None = None,
    dataset_name: str = "kz-transformers/multidomain-kazakh-dataset",
    output_file: str | None = None,
) -> dict:
    """Run full evaluation pipeline."""
    results = {"model_path": model_path}

    # Perplexity
    ppl = compute_perplexity(model_path, dataset_name=dataset_name)
    results["perplexity"] = ppl

    # Generation
    if prompts_file:
        prompts = Path(prompts_file).read_text().strip().split("\n")
        prompts = [p.strip() for p in prompts if p.strip() and not p.startswith("#")]

        generations = generate_text(model_path, prompts)
        results["generations"] = []

        for prompt, gen in zip(prompts, generations):
            kk_ratio = kazakh_char_ratio(gen)
            results["generations"].append({
                "prompt": prompt,
                "generated": gen,
                "kazakh_char_ratio": kk_ratio,
            })

        avg_kk_ratio = sum(g["kazakh_char_ratio"] for g in results["generations"]) / len(results["generations"])
        results["avg_kazakh_char_ratio"] = avg_kk_ratio
        logger.info("Average Kazakh character ratio: %.4f", avg_kk_ratio)

    # Save results
    if output_file is None:
        output_file = str(Path(model_path) / "eval_results.json")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_file)

    return results


def main():
    parser = argparse.ArgumentParser(description="SLM Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompts", type=str, default=None, help="Path to prompts file")
    parser.add_argument("--dataset", default="kz-transformers/multidomain-kazakh-dataset")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        prompts_file=args.prompts,
        dataset_name=args.dataset,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
