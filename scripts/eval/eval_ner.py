"""Evaluate model on NER entity classification using KazNERD dataset.

Scoring: prompted entity classification via score_choices() from eval_mc_bench.
Extracts entity spans from BIO annotations, presents each entity in context,
and scores 6 simplified entity type labels in Kazakh.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

# Allow imports from scripts/eval/ when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import MODEL_REGISTRY, load_model, get_model_short_name
from eval_mc_bench import score_choices, score_choices_api

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Simplified 6-class entity types with Kazakh labels
ENTITY_TYPES = {
    "PERSON": "адам",
    "LOCATION": "орын",
    "ORGANIZATION": "ұйым",
    "DATE": "күн",
    "MONEY": "ақша",
    "OTHER": "басқа",
}

# Map KazNERD 25 fine-grained classes to 6 simplified categories
KAZNERD_TO_SIMPLE = {
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOCATION": "LOCATION",
    "ORGANIZATION": "ORGANIZATION",
    "FACILITY": "LOCATION",
    "DATE": "DATE",
    "CARDINAL": "OTHER",
    "MONEY": "MONEY",
    "NORP": "ORGANIZATION",
    "PRODUCT": "OTHER",
    "EVENT": "OTHER",
    "LANGUAGE": "OTHER",
    "ORDINAL": "OTHER",
    "QUANTITY": "OTHER",
    "TIME": "DATE",
    "PERCENT": "OTHER",
    "LAW": "OTHER",
    "WORK_OF_ART": "OTHER",
    "AGE": "OTHER",
    "DISEASE": "OTHER",
    "CONTACT": "OTHER",
    "ADAGE": "OTHER",
    "POSITION": "OTHER",
    "PROJECT": "OTHER",
    "MISCELLANEOUS": "OTHER",
}


def extract_entities(
    tokens: list[str],
    ner_tags: list[int | str],
    tag_names: list[str] | None = None,
) -> list[dict]:
    """Extract entity spans from BIO-tagged tokens.

    Args:
        tokens: List of word tokens.
        ner_tags: BIO tag indices (int) or tag strings (str).
        tag_names: If ner_tags are ints, maps index -> tag string (e.g. from dataset features).

    Returns:
        List of entity dicts with keys: entity_text, entity_type, context_before, context_after.
    """
    entities = []
    current_entity_tokens: list[str] = []
    current_entity_type: str | None = None
    current_start: int = 0

    def _resolve_tag(tag) -> tuple[str, str]:
        """Return (bio_prefix, entity_type) from a tag."""
        if isinstance(tag, int):
            if tag_names is not None:
                tag_str = tag_names[tag]
            else:
                return ("O", "O")
        else:
            tag_str = str(tag)

        if tag_str == "O" or tag_str == "0":
            return ("O", "O")
        if "-" in tag_str:
            prefix, etype = tag_str.split("-", 1)
            return (prefix, etype)
        return ("O", "O")

    def _flush():
        if current_entity_tokens and current_entity_type:
            simplified = KAZNERD_TO_SIMPLE.get(current_entity_type, "OTHER")
            entity_text = " ".join(current_entity_tokens)
            ctx_before = " ".join(tokens[max(0, current_start - 20) : current_start])
            end_idx = current_start + len(current_entity_tokens)
            ctx_after = " ".join(tokens[end_idx : end_idx + 10])
            entities.append(
                {
                    "entity_text": entity_text,
                    "entity_type": simplified,
                    "context_before": ctx_before,
                    "context_after": ctx_after,
                }
            )

    for i, tag in enumerate(ner_tags):
        prefix, etype = _resolve_tag(tag)

        if prefix == "B":
            _flush()
            current_entity_tokens = [tokens[i]]
            current_entity_type = etype
            current_start = i
        elif prefix == "I" and current_entity_type is not None:
            current_entity_tokens.append(tokens[i])
        else:
            _flush()
            current_entity_tokens = []
            current_entity_type = None

    _flush()
    return entities


def load_kaznerd() -> list[dict]:
    """Load KazNERD dataset and extract entities.

    Tries HuggingFace Hub first, with fallback for alternative names.

    Returns:
        List of entity dicts ready for evaluation.
    """
    from datasets import load_dataset

    ds = None
    for dataset_id in ["issai/KazNERD", "issai/kaznerd"]:
        try:
            logger.info("Trying to load %s ...", dataset_id)
            ds = load_dataset(dataset_id, split="test")
            logger.info("Loaded %s: %d examples", dataset_id, len(ds))
            break
        except Exception as e:
            logger.warning("Failed to load %s: %s", dataset_id, e)

    if ds is None:
        raise RuntimeError(
            "KazNERD not found on HF. Download CoNLL files from "
            "https://github.com/IS2AI/KazNERD and convert."
        )

    # Resolve tag names from dataset features if ner_tags are ints
    tag_names = None
    if hasattr(ds.features.get("ner_tags", None), "feature"):
        tag_feature = ds.features["ner_tags"].feature
        if hasattr(tag_feature, "names"):
            tag_names = tag_feature.names

    all_entities = []
    for row in ds:
        entities = extract_entities(row["tokens"], row["ner_tags"], tag_names=tag_names)
        all_entities.extend(entities)

    logger.info("Extracted %d entities total", len(all_entities))
    return all_entities


def main():
    parser = argparse.ArgumentParser(
        description="NER evaluation on KazNERD via prompted entity classification"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model registry key (e.g. sozkz-50m) or HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: paper/results/ner/{model_short}.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of entities to evaluate (0=all)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Force 4-bit quantization (overrides registry default)",
    )
    args = parser.parse_args()

    # Resolve model
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
            is_api = False
            device = next(model.parameters()).device
    else:
        model_short = args.model.split("/")[-1]
        model_key = model_short
        is_api = False
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": device_str}
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
        device = next(model.parameters()).device

    output_path = args.output or f"paper/results/ner/{model_short}.json"

    # Load entities
    entities = load_kaznerd()
    if args.limit > 0:
        entities = entities[: args.limit]

    # Choices for entity type classification (Kazakh labels)
    choices = {k: v for k, v in ENTITY_TYPES.items()}

    correct = 0
    total = 0
    per_entity_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for ent in tqdm(entities, desc=f"NER ({model_short})"):
        ctx = ent["context_before"]
        entity_text = ent["entity_text"]
        prompt = f"{ctx} {entity_text} -- бұл "
        gold = ent["entity_type"]

        if is_api:
            pred, _ = score_choices_api(prompt, choices, api_url)
        else:
            pred, _ = score_choices(
                model, tokenizer, prompt, choices, device=str(device)
            )

        hit = pred == gold
        correct += int(hit)
        total += 1
        per_entity_type[gold]["total"] += 1
        per_entity_type[gold]["correct"] += int(hit)

    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall: {correct}/{total} = {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Random baseline: {100 / len(ENTITY_TYPES):.1f}%\n")

    per_type_results = {}
    for etype in sorted(per_entity_type):
        c = per_entity_type[etype]["correct"]
        t = per_entity_type[etype]["total"]
        acc = c / t if t > 0 else 0
        per_type_results[etype] = {"correct": c, "total": t, "accuracy": round(acc, 4)}
        print(f"  {etype}: {c}/{t} = {acc * 100:.1f}%")

    results = {
        "model": args.model,
        "model_short": model_short,
        "task": "ner",
        "dataset": "issai/KazNERD",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "total": total,
            "correct": correct,
        },
        "per_entity_type": per_type_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
