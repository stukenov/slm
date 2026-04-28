#!/usr/bin/env python3
"""exp032 v2: Extend TinyLlama tokenizer by merging SentencePiece models.

The correct approach (Chinese-LLaMA method):
1. Train a new SentencePiece model on Kazakh+Russian corpus
2. Merge at protobuf level: add new pieces with merge rules
3. This ensures the tokenizer ACTUALLY USES new tokens

Reference: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/merge_tokenizer/merge_tokenizers.py
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as sp_model

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def train_sp_model(corpus_path: str, output_prefix: str, vocab_size: int = 20000):
    """Train a SentencePiece model on corpus."""
    log.info("Training SentencePiece model (vocab=%d)...", vocab_size)
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,
        num_threads=os.cpu_count(),
        split_digits=True,
        byte_fallback=True,
        max_sentence_length=16384,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
    )
    log.info("SentencePiece model trained: %s.model", output_prefix)


def merge_sp_models(base_sp_path: str, new_sp_path: str, output_sp_path: str,
                    max_new_tokens: int = 10000):
    """Merge two SentencePiece models by adding new pieces from new_sp into base_sp."""

    # Load base model
    base_m = sp_model.ModelProto()
    with open(base_sp_path, "rb") as f:
        base_m.ParseFromString(f.read())

    # Load new model
    new_m = sp_model.ModelProto()
    with open(new_sp_path, "rb") as f:
        new_m.ParseFromString(f.read())

    base_pieces = {p.piece for p in base_m.pieces}
    log.info("Base model pieces: %d", len(base_pieces))
    log.info("New model pieces: %d", len(new_m.pieces))

    # Find pieces in new model that are not in base
    added = 0
    for piece in new_m.pieces:
        if piece.piece not in base_pieces and added < max_new_tokens:
            # Skip special tokens (keep type 1 = NORMAL, which is what BPE pieces are)
            # Types: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=BYTE, 6=UNUSED
            if piece.type != 1:
                continue
            new_piece = sp_model.ModelProto.SentencePiece()
            new_piece.piece = piece.piece
            new_piece.score = piece.score
            new_piece.type = piece.type
            base_m.pieces.append(new_piece)
            base_pieces.add(piece.piece)
            added += 1

    log.info("Added %d new pieces. Total: %d", added, len(base_m.pieces))

    with open(output_sp_path, "wb") as f:
        f.write(base_m.SerializeToString())
    log.info("Merged model saved: %s", output_sp_path)
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--dataset", default="kz-transformers/multidomain-kazakh-dataset")
    parser.add_argument("--new-sp-vocab-size", type=int, default=20000)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--output-dir", default="/root/exp032_extended_model_v2")
    parser.add_argument("--corpus-file", default="/root/exp032_extended_model/_corpus.txt",
                        help="Reuse corpus from v1 if available")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load base tokenizer and model
    # =========================================================================
    log.info("Loading base tokenizer: %s", args.base_model)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    log.info("Base vocab size: %d", base_tokenizer.vocab_size)

    # Find the SentencePiece model file
    base_sp_path = None
    from huggingface_hub import hf_hub_download
    try:
        base_sp_path = hf_hub_download(args.base_model, "tokenizer.model")
        log.info("Base SP model: %s", base_sp_path)
    except Exception:
        log.warning("No tokenizer.model found — base tokenizer may not be SentencePiece")
        # Fallback: check if tokenizer has a fast tokenizer with merges
        raise RuntimeError("TinyLlama must have tokenizer.model (SentencePiece)")

    # =========================================================================
    # Step 2: Prepare corpus
    # =========================================================================
    corpus_file = Path(args.corpus_file)
    if not corpus_file.exists():
        log.info("Loading dataset for corpus...")
        ds = load_dataset(args.dataset, split="train")
        ds = ds.shuffle(seed=args.seed)
        log.info("Dumping text to %s...", corpus_file)
        corpus_file.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_file, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    f.write(text + "\n")
                if (i + 1) % 5_000_000 == 0:
                    log.info("  %dM rows...", (i + 1) // 1_000_000)
        log.info("Corpus ready")
    else:
        log.info("Reusing existing corpus: %s", corpus_file)

    # =========================================================================
    # Step 3: Train new SentencePiece on corpus
    # =========================================================================
    new_sp_prefix = str(output / "new_kk_ru_sp")
    if not Path(new_sp_prefix + ".model").exists():
        train_sp_model(str(corpus_file), new_sp_prefix, vocab_size=args.new_sp_vocab_size)
    else:
        log.info("Reusing existing new SP model: %s.model", new_sp_prefix)

    # =========================================================================
    # Step 4: Merge SentencePiece models
    # =========================================================================
    merged_sp_path = str(output / "tokenizer.model")
    num_added = merge_sp_models(
        base_sp_path, new_sp_prefix + ".model", merged_sp_path,
        max_new_tokens=args.max_new_tokens,
    )

    # =========================================================================
    # Step 5: Create HuggingFace tokenizer from merged SP model
    # =========================================================================
    log.info("Creating HuggingFace tokenizer from merged SP model...")

    # Copy all tokenizer files from base, then overwrite tokenizer.model with merged
    from shutil import copy2
    # Save merged SP model to temp location, save base tokenizer, then overwrite
    import tempfile
    tmp_sp = output / "_merged_tokenizer.model"
    copy2(merged_sp_path, tmp_sp)
    base_tokenizer.save_pretrained(output)
    # Now overwrite the base tokenizer.model with our merged version
    copy2(tmp_sp, output / "tokenizer.model")
    tmp_sp.unlink()

    # CRITICAL: delete tokenizer.json — it contains old fast tokenizer BPE rules
    # and takes precedence over tokenizer.model. Without this, AutoTokenizer
    # loads the old rules and ignores our merged SP model.
    fast_tok = output / "tokenizer.json"
    if fast_tok.exists():
        fast_tok.unlink()
        log.info("Deleted tokenizer.json (forces fallback to merged tokenizer.model)")

    # Reload — AutoTokenizer will now use tokenizer.model (SentencePiece)
    merged_tokenizer = AutoTokenizer.from_pretrained(str(output))

    new_vocab_size = merged_tokenizer.vocab_size
    log.info("Merged tokenizer vocab: %d (was %d)", new_vocab_size, base_tokenizer.vocab_size)

    # Pad to multiple of 64
    padded_size = ((new_vocab_size + 63) // 64) * 64
    if padded_size > new_vocab_size:
        pad_tokens = [f"<pad_{i}>" for i in range(padded_size - new_vocab_size)]
        merged_tokenizer.add_special_tokens({"additional_special_tokens": pad_tokens})
        log.info("Padded: %d -> %d", new_vocab_size, len(merged_tokenizer))

    final_vocab_size = len(merged_tokenizer)

    # =========================================================================
    # Step 6: Fertility test
    # =========================================================================
    test_sentences = {
        "kaz": "Қазақстан Республикасы — Орталық Азиядағы мемлекет",
        "rus": "Республика Казахстан — государство в Центральной Азии",
        "eng": "Republic of Kazakhstan — a state in Central Asia",
    }
    fertility_results = {}
    log.info("Fertility comparison (before -> after):")
    for lang, text in test_sentences.items():
        old_toks = base_tokenizer.encode(text, add_special_tokens=False)
        new_toks = merged_tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        old_f = len(old_toks) / len(words)
        new_f = len(new_toks) / len(words)
        log.info("  %s: %.2f -> %.2f  (%d -> %d tokens for %d words)",
                 lang, old_f, new_f, len(old_toks), len(new_toks), len(words))
        fertility_results[lang] = {"before": round(old_f, 2), "after": round(new_f, 2),
                                    "tokens_before": len(old_toks), "tokens_after": len(new_toks)}

    # =========================================================================
    # Step 7: Load model, resize embeddings, subword mean init
    # =========================================================================
    log.info("Loading base model for embedding extension...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    )
    old_embed = base_model.get_input_embeddings().weight.data.clone()
    old_lm_head = base_model.lm_head.weight.data.clone()
    old_size = old_embed.shape[0]

    base_model.resize_token_embeddings(final_vocab_size)
    log.info("Resized embeddings: %d -> %d", old_size, final_vocab_size)

    embed = base_model.get_input_embeddings().weight.data
    lm_head = base_model.lm_head.weight.data

    # Copy original embeddings (first old_size entries stay the same)
    # For new entries (old_size to final_vocab_size), use subword mean
    log.info("Initializing new embeddings via subword mean...")
    init_count = 0
    mean_embed = old_embed.mean(dim=0)
    mean_lm = old_lm_head.mean(dim=0)

    for idx in range(old_size, final_vocab_size):
        token_str = merged_tokenizer.convert_ids_to_tokens(idx)
        if token_str is None or token_str.startswith("<pad_"):
            embed[idx] = mean_embed
            lm_head[idx] = mean_lm
            continue

        # Tokenize new token with OLD tokenizer to get subword decomposition
        sub_ids = base_tokenizer.encode(token_str, add_special_tokens=False)
        if sub_ids and all(sid < old_size for sid in sub_ids):
            embed[idx] = old_embed[sub_ids].mean(dim=0)
            lm_head[idx] = old_lm_head[sub_ids[0]]
            init_count += 1
        else:
            embed[idx] = mean_embed
            lm_head[idx] = mean_lm

    log.info("Initialized %d / %d new embeddings via subword mean", init_count, final_vocab_size - old_size)

    # =========================================================================
    # Step 8: Save everything
    # =========================================================================
    log.info("Saving extended model + tokenizer to %s", output)
    base_model.save_pretrained(output)
    merged_tokenizer.save_pretrained(output)

    meta = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "original_vocab_size": old_size,
        "sp_new_vocab_size": args.new_sp_vocab_size,
        "new_pieces_added": num_added,
        "final_vocab_size": final_vocab_size,
        "padded_to": padded_size,
        "init_method": "subword_mean_eeve",
        "tokens_initialized_subword": init_count,
        "tokens_initialized_mean": (final_vocab_size - old_size) - init_count,
        "fertility": fertility_results,
    }
    with open(output / "extension_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("DONE. Metadata: %s", json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
