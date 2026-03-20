"""Model registry for SLM evaluation pipeline.

Central registry of all 14 models (4 own + 9 competitors + 1 API)
with loading helpers, quantization config, and device fallback.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REGISTRY: dict[str, dict] = {
    # === Own models (type="hf", quantize=False) ===
    "sozkz-50m": {
        "hf_id": "saken-tukenov/sozkz-core-llama-50m-kk-base-v4",
        "tokenizer": "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        "params": "50M",
        "type": "hf",
        "quantize": False,
    },
    "sozkz-150m": {
        "hf_id": "saken-tukenov/sozkz-core-llama-150m-kk-base-v1",
        "tokenizer": "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        "params": "150M",
        "type": "hf",
        "quantize": False,
    },
    "sozkz-300m": {
        "hf_id": "stukenov/sozkz-core-llama-300m-kk-base-v1",
        "tokenizer": "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        "params": "300M",
        "type": "hf",
        "quantize": False,
    },
    "sozkz-600m": {
        "hf_id": "stukenov/sozkz-core-llama-600m-kk-base-v1",
        "tokenizer": "saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        "params": "600M",
        "type": "hf",
        "quantize": False,
    },
    # === Competitors (type="hf") ===
    "gemma-2b": {
        "hf_id": "google/gemma-2-2b",
        "params": "2B",
        "type": "hf",
        "quantize": False,
    },
    "gemma-9b": {
        "hf_id": "google/gemma-2-9b",
        "params": "9B",
        "type": "hf",
        "quantize": True,
    },
    "llama-3-1b": {
        "hf_id": "meta-llama/Llama-3.2-1B",
        "params": "1B",
        "type": "hf",
        "quantize": False,
    },
    "llama-3-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B",
        "params": "3B",
        "type": "hf",
        "quantize": False,
    },
    "llama-3-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "params": "8B",
        "type": "hf",
        "quantize": True,
    },
    "qwen-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "params": "0.5B",
        "type": "hf",
        "quantize": False,
    },
    "qwen-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B",
        "params": "1.5B",
        "type": "hf",
        "quantize": False,
    },
    "qwen-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "params": "7B",
        "type": "hf",
        "quantize": True,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.3",
        "params": "7B",
        "type": "hf",
        "quantize": True,
    },
    # === API models ===
    "gpt-oss-120b": {
        "api_url": "http://localhost:8080",
        "params": "120B",
        "type": "api",
        "quantize": False,
    },
}


def _resolve_device(device: str) -> str:
    """Resolve device with fallback: cuda -> mps -> cpu."""
    if device == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def load_model(
    model_key: str, device: str = "cuda"
) -> tuple[AutoModelForCausalLM | None, AutoTokenizer | None]:
    """Load a model and tokenizer from the registry.

    Args:
        model_key: Key in MODEL_REGISTRY.
        device: Target device (cuda/mps/cpu). Falls back automatically.

    Returns:
        (model, tokenizer) tuple. Both None for API-type models.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{model_key}'. "
            f"Available: {', '.join(sorted(MODEL_REGISTRY))}"
        )

    entry = MODEL_REGISTRY[model_key]

    if entry["type"] == "api":
        return None, None

    device = _resolve_device(device)
    hf_id = entry["hf_id"]

    # Quantization config for large models
    load_kwargs: dict = {"torch_dtype": torch.bfloat16}
    if entry.get("quantize"):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # quantized models manage their own device placement
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model.eval()

    # Tokenizer: use explicit tokenizer field if present, else hf_id
    tok_id = entry.get("tokenizer", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_model_short_name(model_key: str) -> str:
    """Return the short name for a model (the registry key itself)."""
    return model_key


def list_models(model_type: str | None = None) -> list[str]:
    """List registered model keys, optionally filtered by type.

    Args:
        model_type: "hf", "api", or None for all.

    Returns:
        List of model keys.
    """
    if model_type is None:
        return list(MODEL_REGISTRY.keys())
    return [k for k, v in MODEL_REGISTRY.items() if v["type"] == model_type]


def main():
    parser = argparse.ArgumentParser(description="SLM Model Registry")
    parser.add_argument(
        "--type", choices=["hf", "api"], default=None, help="Filter by model type"
    )
    args = parser.parse_args()

    models = list_models(args.type)
    print(f"{'Key':<20} {'Params':<8} {'Type':<6} {'Quantize':<10} {'HF ID / API URL'}")
    print("-" * 90)
    for key in models:
        entry = MODEL_REGISTRY[key]
        identifier = entry.get("hf_id", entry.get("api_url", "N/A"))
        quantize = entry.get("quantize", False)
        print(f"{key:<20} {entry['params']:<8} {entry['type']:<6} {str(quantize):<10} {identifier}")

    print(f"\nTotal: {len(models)} models")


if __name__ == "__main__":
    main()
