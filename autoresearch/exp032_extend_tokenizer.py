#!/usr/bin/env python3
"""exp032: Extend TinyLlama tokenizer with Kazakh+Russian tokens.

1. Train BPE on MDBKD corpus
2. Extract new tokens not in TinyLlama vocab
3. Filter by min frequency
4. Merge into TinyLlama tokenizer
5. Initialize embeddings via subword mean (EEVE method)
6. Save extended model + tokenizer

Reference: Chinese-LLaMA (Cui 2023), EEVE (Kim 2024), Swallow (Fujii 2024)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--dataset", default="kz-transformers/multidomain-kazakh-dataset")
    parser.add_argument("--new-vocab-size", type=int, default=20000,
                        help="Size of the new BPE tokenizer trained on target data")
    parser.add_argument("--max-new-tokens", type=int, default=10000,
                        help="Max new tokens to add to the base tokenizer")
    parser.add_argument("--min-frequency", type=int, default=100,
                        help="Min freq in corpus for a new token to be added")
    parser.add_argument("--output-dir", default="/root/exp032_extended_model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load base model and tokenizer
    # =========================================================================
    log.info("Loading base model: %s", args.base_model)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    )
    base_vocab = set(base_tokenizer.get_vocab().keys())
    log.info("Base vocab size: %d", len(base_vocab))

    # =========================================================================
    # Step 2: Train new BPE on MDBKD
    # =========================================================================
    corpus_file = output / "_corpus.txt"
    if not corpus_file.exists():
        log.info("Loading dataset for tokenizer training...")
        ds = load_dataset(args.dataset, split="train")
        ds = ds.shuffle(seed=args.seed)

        log.info("Dumping text to %s...", corpus_file)
        with open(corpus_file, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    f.write(text + "\n")
                if (i + 1) % 5_000_000 == 0:
                    log.info("  %dM rows written...", (i + 1) // 1_000_000)
        log.info("Corpus file ready: %s", corpus_file)
    else:
        log.info("Using cached corpus: %s", corpus_file)

    log.info("Training new BPE tokenizer (vocab=%d)...", args.new_vocab_size)
    t0 = time.time()
    new_bpe = ByteLevelBPETokenizer()
    new_bpe.train(
        files=[str(corpus_file)],
        vocab_size=args.new_vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
    )
    log.info("BPE training done in %.1fs", time.time() - t0)

    new_vocab = new_bpe.get_vocab()
    log.info("New tokenizer vocab: %d tokens", len(new_vocab))

    # =========================================================================
    # Step 3: Find tokens to add (not in base vocab)
    # =========================================================================
    candidates = set(new_vocab.keys()) - base_vocab
    # Remove special tokens
    candidates -= {"<s>", "</s>", "<unk>", "<pad>"}
    log.info("Candidate new tokens (not in base): %d", len(candidates))

    # Sort by frequency (approximated by BPE merge order = vocab ID)
    candidates_sorted = sorted(candidates, key=lambda t: new_vocab.get(t, 999999))
    tokens_to_add = candidates_sorted[:args.max_new_tokens]
    log.info("Tokens to add: %d (capped at %d)", len(tokens_to_add), args.max_new_tokens)

    # Log some examples
    log.info("Sample new tokens (first 30): %s", tokens_to_add[:30])
    log.info("Sample new tokens (last 30): %s", tokens_to_add[-30:])

    # =========================================================================
    # Step 4: Add tokens to base tokenizer
    # =========================================================================
    num_added = base_tokenizer.add_tokens(tokens_to_add)
    log.info("Added %d tokens to tokenizer. New vocab size: %d", num_added, len(base_tokenizer))

    # Pad to multiple of 64 for GPU efficiency
    current_size = len(base_tokenizer)
    padded_size = ((current_size + 63) // 64) * 64
    if padded_size > current_size:
        pad_tokens = [f"<pad_{i}>" for i in range(padded_size - current_size)]
        base_tokenizer.add_special_tokens({"additional_special_tokens": pad_tokens})
        log.info("Padded vocab from %d to %d (multiple of 64)", current_size, len(base_tokenizer))

    # =========================================================================
    # Step 5: Resize model embeddings + subword mean init (EEVE method)
    # =========================================================================
    old_embed_weight = base_model.get_input_embeddings().weight.data.clone()
    old_lm_head_weight = base_model.lm_head.weight.data.clone()
    old_vocab_size = old_embed_weight.shape[0]

    base_model.resize_token_embeddings(len(base_tokenizer))
    log.info("Resized embeddings: %d -> %d", old_vocab_size, len(base_tokenizer))

    # Subword mean initialization
    log.info("Initializing new embeddings via subword mean (EEVE method)...")
    embed_weight = base_model.get_input_embeddings().weight.data
    lm_head_weight = base_model.lm_head.weight.data

    # For tokens beyond old vocab size, initialize with subword mean
    original_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    init_count = 0

    for idx in range(old_vocab_size, len(base_tokenizer)):
        token_str = base_tokenizer.convert_ids_to_tokens(idx)
        if token_str is None or token_str.startswith("<pad_"):
            # Pad tokens: use mean of all embeddings
            embed_weight[idx] = old_embed_weight.mean(dim=0)
            lm_head_weight[idx] = old_lm_head_weight.mean(dim=0)
            continue

        # Tokenize the new token with the ORIGINAL tokenizer
        sub_ids = original_tokenizer.encode(token_str, add_special_tokens=False)

        if len(sub_ids) > 0:
            # Input embedding: mean of subword embeddings
            sub_embeds = old_embed_weight[sub_ids]
            embed_weight[idx] = sub_embeds.mean(dim=0)

            # Output embedding: first subword (EEVE method)
            lm_head_weight[idx] = old_lm_head_weight[sub_ids[0]]
            init_count += 1
        else:
            # Fallback: mean of all embeddings
            embed_weight[idx] = old_embed_weight.mean(dim=0)
            lm_head_weight[idx] = old_lm_head_weight.mean(dim=0)

    log.info("Initialized %d new token embeddings via subword mean", init_count)

    # =========================================================================
    # Step 6: Measure new fertility
    # =========================================================================
    test_sentences = {
        "kaz": "Қазақстан Республикасы — Орталық Азиядағы мемлекет",
        "rus": "Республика Казахстан — государство в Центральной Азии",
        "eng": "Republic of Kazakhstan — a state in Central Asia",
    }
    log.info("Fertility comparison (before -> after):")
    for lang, text in test_sentences.items():
        old_toks = original_tokenizer.encode(text, add_special_tokens=False)
        new_toks = base_tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        old_f = len(old_toks) / len(words)
        new_f = len(new_toks) / len(words)
        log.info("  %s: %.2f -> %.2f  (%d -> %d tokens for %d words)",
                 lang, old_f, new_f, len(old_toks), len(new_toks), len(words))

    # =========================================================================
    # Step 7: Save
    # =========================================================================
    log.info("Saving extended model + tokenizer to %s", output)
    base_model.save_pretrained(output)
    base_tokenizer.save_pretrained(output)

    # Save metadata
    meta = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "original_vocab_size": old_vocab_size,
        "new_tokens_added": num_added,
        "final_vocab_size": len(base_tokenizer),
        "padded_to": padded_size,
        "init_method": "subword_mean_eeve",
        "tokens_initialized": init_count,
    }
    with open(output / "extension_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved. DONE.")
    log.info("Extension metadata: %s", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
