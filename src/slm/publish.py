"""Publish models and tokenizers to HuggingFace Hub."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, ModelCard, ModelCardData
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_model_card(
    model_path: str,
    repo_name: str,
    base_model: str | None = None,
    eval_results_path: str | None = None,
) -> str:
    """Create a Model Card markdown string."""
    # Load eval results if available
    metrics_text = ""
    if eval_results_path and Path(eval_results_path).exists():
        with open(eval_results_path) as f:
            results = json.load(f)
        ppl = results.get("perplexity", "N/A")
        loss = results.get("eval_loss", "N/A")
        metrics_text = f"""
## Evaluation Results

| Metric | Value |
|--------|-------|
| Eval loss | {loss} |
| Perplexity (validation) | {ppl} |
"""

    # Load model config for architecture details
    config_path = Path(model_path) / "config.json"
    arch_text = ""
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        arch_text = f"""
## Architecture

| Parameter | Value |
|-----------|-------|
| Type | {cfg.get('model_type', 'N/A')} |
| Parameters | ~{_count_params_from_config(cfg)}M |
| Hidden size | {cfg.get('hidden_size', 'N/A')} |
| Layers | {cfg.get('num_hidden_layers', 'N/A')} |
| Attention heads | {cfg.get('num_attention_heads', 'N/A')} |
| Vocab size | {cfg.get('vocab_size', 'N/A'):,} |
| Context length | {cfg.get('max_position_embeddings', 'N/A')} |
"""

    if base_model:
        description = (
            f"A Kazakh language model created by continuing pre-training of "
            f"[{base_model}](https://huggingface.co/{base_model}) on a curated Kazakh text corpus."
        )
        tags_extra = f"- dapt\nbase_model: {base_model}"
    else:
        description = (
            "A Kazakh language model trained from scratch on a curated, "
            "deduplicated Kazakh text corpus (~1B tokens). "
            "Llama architecture with Chinchilla-optimal compute budget."
        )
        tags_extra = "- from-scratch\n- llama"

    card_content = f"""---
language: kk
license: apache-2.0
tags:
- kazakh
- language-model
- causal-lm
{tags_extra}
---

# {repo_name.split('/')[-1]}

{description}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

text = "Қазақстан — "
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
{arch_text}{metrics_text}
## Training Data

Trained on [saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2) — a curated Kazakh corpus processed through a 9-stage cleaning pipeline:

1. NFC normalization + whitespace collapsing
2. Kazakh character verification
3. Script profile filtering (Cyrillic >= 60%)
4. fastText language identification (kk >= 0.5)
5. Junk removal (URLs, HTML, boilerplate)
6. Repetition filtering
7. Exact + near deduplication (MinHash LSH)
8. Domain balancing

Sources: CC-100, OSCAR, Wikipedia, Leipzig, Kazakh News, Kazakh Books.

## Training

See the [SLM project](https://github.com/sakentukenov/slm) for full experiment details.
"""
    return card_content


def _count_params_from_config(cfg: dict) -> int:
    """Rough parameter count from config (in millions)."""
    vocab = cfg.get("vocab_size", 0)
    hidden = cfg.get("hidden_size", 0)
    layers = cfg.get("num_hidden_layers", 0)
    intermediate = cfg.get("intermediate_size", 0)
    n_heads = cfg.get("num_attention_heads", 0)
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    tied = cfg.get("tie_word_embeddings", False)

    head_dim = hidden // n_heads if n_heads else 0
    embed = vocab * hidden
    per_layer = (
        hidden * n_heads * head_dim  # Q
        + hidden * n_kv_heads * head_dim  # K
        + hidden * n_kv_heads * head_dim  # V
        + n_heads * head_dim * hidden  # O
        + 3 * hidden * intermediate  # MLP (gate, up, down)
    )
    total = embed + layers * per_layer
    if not tied:
        total += vocab * hidden  # output projection
    return round(total / 1e6)


def publish_model(
    model_path: str,
    repo_name: str,
    base_model: str | None = None,
    private: bool = False,
) -> str:
    """Push model to HuggingFace Hub."""
    logger.info("Publishing %s to %s", model_path, repo_name)

    model_path = Path(model_path)

    # Check for eval results
    eval_path = model_path / "eval_results.json"
    eval_results_path = str(eval_path) if eval_path.exists() else None

    # Load and push model
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path))

    tokenizer.push_to_hub(repo_name, private=private)
    model.push_to_hub(repo_name, private=private)

    # Create and push model card
    card_content = create_model_card(
        str(model_path), repo_name, base_model, eval_results_path
    )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model",
    )

    url = f"https://huggingface.co/{repo_name}"
    logger.info("Model published: %s", url)
    return url


def main():
    parser = argparse.ArgumentParser(description="Publish model to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--repo_name", type=str, required=True, help="e.g. sakentukenov/slm-kk-14m-dapt-v1")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name for model card")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    publish_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        base_model=args.base_model,
        private=args.private,
    )


if __name__ == "__main__":
    main()
