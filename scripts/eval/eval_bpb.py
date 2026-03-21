"""Evaluate model bits-per-byte (BPB) on external Kazakh text.

BPB = total_nll / (num_utf8_bytes * ln(2))

Uses sliding window for texts longer than model context.
Supports FLORES-200 kaz_Cyrl and Kazakh Wikipedia as corpora.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys

import requests
import torch
from tqdm import tqdm

# Allow imports from scripts/eval/ when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import MODEL_REGISTRY, load_model, get_model_short_name

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_bpb(
    model,
    tokenizer,
    text: str,
    device: str = "cuda",
    stride: int = 512,
) -> float:
    """Compute bits-per-byte on a text using sliding window.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        text: Raw text to measure.
        device: Device string.
        stride: Sliding window stride in tokens.

    Returns:
        BPB value (float). Typical range: 0.5-3.0 for language models.
    """
    # Count raw UTF-8 bytes (Kazakh Cyrillic = 2 bytes per char)
    num_bytes = len(text.encode("utf-8"))
    if num_bytes == 0:
        return 0.0

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    # Get max context length
    max_len = getattr(model.config, "max_position_embeddings", 2048)

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for begin in tqdm(
        range(0, seq_len, stride),
        desc="BPB windows",
        leave=False,
        disable=seq_len <= max_len,
    ):
        end = min(begin + max_len, seq_len)
        window_ids = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(window_ids)
            logits = outputs.logits  # [1, window_len, vocab]

        # Shift: predict token i from position i-1
        shift_logits = logits[0, :-1, :]  # [window_len-1, vocab]
        shift_labels = window_ids[0, 1:]  # [window_len-1]

        token_nll = loss_fn(shift_logits, shift_labels)  # [window_len-1]

        # Only count tokens not already counted from previous window
        # offset is relative to the shifted sequence (hence -1 adjustments)
        offset = max(0, prev_end - begin - 1) if prev_end > begin else 0
        total_nll += token_nll[offset:].sum().item()
        total_tokens += len(token_nll) - offset

        prev_end = end

        if end >= seq_len:
            break

    # Convert nats to bits, normalize by bytes
    bpb = total_nll / (num_bytes * math.log(2))

    logger.info(
        "BPB=%.4f | tokens=%d | bytes=%d | total_nll=%.2f",
        bpb, total_tokens, num_bytes, total_nll,
    )
    return bpb


def compute_bpb_api(
    text: str,
    api_url: str = "http://localhost:8080",
) -> float | None:
    """Compute BPB via GPT-OSS-120B API.

    Posts text to the /completion endpoint with logprobs enabled.
    Extracts total prompt log-probability for BPB calculation.

    Args:
        text: Raw text to measure.
        api_url: Base URL of the API server.

    Returns:
        BPB value or None if API does not support logprobs.
    """
    num_bytes = len(text.encode("utf-8"))
    if num_bytes == 0:
        return 0.0

    try:
        resp = requests.post(
            f"{api_url}/completion",
            json={"prompt": text, "n_predict": 0, "logprobs": True},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        prompt_logprob = data.get("prompt_logprob", None)
        if prompt_logprob is None:
            logger.warning("API did not return prompt_logprob; BPB unavailable")
            return None

        # prompt_logprob is log-probability (negative), convert to BPB
        bpb = -prompt_logprob / (num_bytes * math.log(2))
        logger.info("API BPB=%.4f | bytes=%d", bpb, num_bytes)
        return bpb

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning("API call failed: %s", e)
        return None


def load_flores_kazakh() -> str:
    """Load FLORES-200 Kazakh (Cyrillic) devtest split.

    Returns:
        Concatenated text from all sentences.
    """
    from datasets import load_dataset

    logger.info("Loading FLORES-200 kaz_Cyrl devtest...")
    ds = load_dataset("facebook/flores", "kaz_Cyrl", split="devtest", trust_remote_code=True)
    text = "\n".join(row["sentence"] for row in ds)
    logger.info(
        "FLORES loaded: %d sentences, %d chars, %d UTF-8 bytes",
        len(ds), len(text), len(text.encode("utf-8")),
    )
    return text


def load_wikipedia_kazakh(max_articles: int = 1000) -> str:
    """Load Kazakh Wikipedia articles (streaming).

    Args:
        max_articles: Maximum number of articles to load.

    Returns:
        Concatenated text from articles.
    """
    from datasets import load_dataset

    logger.info("Loading Kazakh Wikipedia (max %d articles)...", max_articles)
    ds = load_dataset("wikipedia", "20231101.kk", split="train", streaming=True)

    texts = []
    for i, row in enumerate(ds):
        if i >= max_articles:
            break
        texts.append(row["text"])

    text = "\n\n".join(texts)
    logger.info(
        "Wikipedia loaded: %d articles, %d chars, %d UTF-8 bytes",
        len(texts), len(text), len(text.encode("utf-8")),
    )
    return text


def main():
    parser = argparse.ArgumentParser(
        description="BPB evaluation on external Kazakh text"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model registry key (e.g. sozkz-50m) or HuggingFace model ID",
    )
    parser.add_argument(
        "--corpus",
        choices=["flores", "wikipedia"],
        default="flores",
        help="Which corpus to use (default: flores)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: paper/results/bpb/{model_short}.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of characters for quick testing (0=all)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Force 4-bit quantization (overrides registry default)",
    )
    args = parser.parse_args()

    # Resolve model
    is_api = False
    if args.model in MODEL_REGISTRY:
        model_key = args.model
        model_short = get_model_short_name(model_key)
        entry = MODEL_REGISTRY[model_key]

        if entry["type"] == "api":
            model, tokenizer = None, None
            api_url = entry["api_url"]
            is_api = True
        else:
            model, tokenizer = load_model(model_key)
            device = str(next(model.parameters()).device)
    else:
        model_short = args.model.split("/")[-1]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": device}
        if args.quantize:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            del load_kwargs["device_map"]

        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = str(next(model.parameters()).device)

    # Output path
    output_path = args.output or f"paper/results/bpb/{model_short}.json"

    # Load corpus
    if args.corpus == "flores":
        text = load_flores_kazakh()
    else:
        text = load_wikipedia_kazakh()

    if args.limit > 0:
        text = text[: args.limit]
        logger.info("Truncated to %d chars (%d UTF-8 bytes)", len(text), len(text.encode("utf-8")))

    num_bytes = len(text.encode("utf-8"))

    # Compute BPB
    if is_api:
        bpb = compute_bpb_api(text, api_url)
        num_tokens = None
        total_nll = None
    else:
        num_tokens_pre = len(tokenizer.encode(text))
        bpb = compute_bpb(model, tokenizer, text, device=device)
        num_tokens = num_tokens_pre
        # Reconstruct total_nll from bpb for reporting
        total_nll = bpb * num_bytes * math.log(2) if bpb is not None else None

    # Validate range
    if bpb is not None:
        if bpb < 0.5 or bpb > 3.0:
            logger.warning(
                "BPB=%.4f is outside expected range [0.5, 3.0] -- "
                "verify model and corpus are correct",
                bpb,
            )
        print(f"\nBPB: {bpb:.4f}")
    else:
        print("\nBPB: unavailable (API error)")

    results = {
        "model": args.model,
        "model_short": model_short,
        "task": "bpb",
        "corpus": args.corpus,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": {
            "bpb": round(bpb, 6) if bpb is not None else None,
            "num_bytes": num_bytes,
            "num_tokens": num_tokens,
            "total_nll": round(total_nll, 4) if total_nll is not None else None,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
