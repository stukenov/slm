#!/usr/bin/env python3
"""Replace old HF repo names with new SozKZ names in local codebase.
Run from project root: python scripts/_rename_refs.py [--dry-run]
"""

import os
import re
import sys

# Order matters: longer/more specific names first to avoid partial matches
REPLACEMENTS = [
    # Models (longer first)
    ("kazakh-gec-mt5-base-run13-finetune", "sozkz-fix-mt5b-kk-gec-run13-v1"),
    ("kazakh-gec-morphology-50m", "sozkz-fix-mt5-50m-kk-morph-v1"),
    ("kazakh-llama-150m-balanced", "sozkz-core-llama-150m-kk-balanced-v1"),
    ("kazakh-llama-50m-balanced", "sozkz-core-llama-50m-kk-balanced-v1"),
    ("kazakh-llama-50m-v2", "sozkz-core-llama-50m-kk-base-v2"),
    ("kazakh-moe-200M-A50M", "sozkz-moe-mix-200m-kk-base-v1"),
    ("kazakh-moe-160M-A50M-domain", "sozkz-moe-mix-160m-kk-domain-v1"),
    ("kazakh-llama-50m", "sozkz-core-llama-50m-kk-base-v1"),
    ("kazakh-llama-30m", "sozkz-core-llama-30m-kk-base-v1"),
    ("llama-kazakh-150m", "sozkz-core-llama-150m-kk-base-v1"),
    ("slm-kk-pythia14m-dapt-13k", "sozkz-core-pythia-14m-kk-dapt-v1"),
    ("kazakh-t5-50m", "sozkz-seq-t5-50m-kk-base-v1"),
    ("kazakh-t5-sp-32k", "sozkz-vocab-sp-32k-kk-t5-v1"),
    ("kazakh-gec-50m", "sozkz-fix-mt5-50m-kk-gec-v1"),
    # Datasets (longer first)
    ("kazakh-gec-morphology-tokenized", "sozkz-corpus-tokenized-kk-morph-v1"),
    ("kazakh-synthetic-gec-datasets", "sozkz-corpus-synthetic-kk-gec-v1"),
    ("kazakh-llama-50m-tokenized", "sozkz-corpus-tokenized-kk-llama50m-v1"),
    ("kazakh-balanced-gpt2-style", "sozkz-corpus-balanced-kk-gpt2-v1"),
    ("kazakh-clean-pretrain-text-v2", "sozkz-corpus-clean-kk-text-v2"),
    ("kazakh-clean-pretrain-v2", "sozkz-corpus-clean-kk-pretrain-v2"),
    ("kazakh-moe-instruction-sft", "sozkz-corpus-synthetic-kk-moe-sft-v1"),
    ("kazakh-moe-domain-labeled", "sozkz-corpus-balanced-kk-moe-domain-v1"),
    ("kazakh-t5-50m-tokenized", "sozkz-corpus-tokenized-kk-t5-50m-v1"),
    ("gazeta-kazakh", "sozkz-corpus-raw-kk-gazeta-v1"),
    # Already migrated (GPT2 + tokenizer)
    ("kazakh-gpt2-45m-balanced", "sozkz-core-gpt2-45m-kk-balanced-v1"),
    ("kazakh-gpt2-45m-v2", "sozkz-core-gpt2-45m-kk-base-v2"),
    ("kazakh-gpt2-45m", "sozkz-core-gpt2-45m-kk-base-v1"),
    ("kazakh-gpt2-124m", "sozkz-core-gpt2-124m-kk-base-v1"),
    ("kazakh-gpt2-50k", "sozkz-core-gpt2-50k-kk-base-v1"),
    ("kazakh-gpt2-8m", "sozkz-core-gpt2-8m-kk-base-v1"),
    ("kazakh-gpt2-30m", "sozkz-core-gpt2-30m-kk-base-v1"),
    ("kazakh-gpt2-60m", "sozkz-core-gpt2-60m-kk-base-v1"),
    # Deprecated datasets - point to v2
    ("kazakh-clean-pretrain-text", "sozkz-corpus-clean-kk-text-v2"),
    ("kazakh-clean-pretrain", "sozkz-corpus-clean-kk-pretrain-v2"),
]

# Only process these extensions
EXTENSIONS = {".py", ".yaml", ".yml", ".md", ".txt", ".sh"}

# Skip these directories
SKIP_DIRS = {".venv", ".venv-cloud", ".git", "__pycache__", "node_modules", ".claude"}

# Skip the migration scripts themselves
SKIP_FILES = {"hf_migrate.py", "_rename_refs.py"}

# Don't rename local directory paths like ./tokenizers/kazakh-bpe-32k
# We only replace in HF repo reference contexts


def should_process(filepath):
    _, ext = os.path.splitext(filepath)
    if ext not in EXTENSIONS:
        return False
    if os.path.basename(filepath) in SKIP_FILES:
        return False
    return True


def process_file(filepath, dry_run=False):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    original = content
    for old, new in REPLACEMENTS:
        # Don't replace in local paths like ./tokenizers/kazakh-bpe-32k
        # Only replace when it looks like an HF reference (after / or at word boundary)
        # But also replace standalone names in comments, strings, etc.
        # Be careful: don't replace "kazakh-bpe-32k" in local path contexts
        if old == "kazakh-bpe-32k":
            # Only replace HF-style references (saken-tukenov/kazakh-bpe-32k)
            content = content.replace(f"saken-tukenov/{old}", f"saken-tukenov/{new}")
        else:
            content = content.replace(old, new)

    if content != original:
        if dry_run:
            print(f"  WOULD MODIFY: {filepath}")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  MODIFIED: {filepath}")
        return True
    return False


def main():
    dry_run = "--dry-run" in sys.argv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dry_run:
        print("=== DRY RUN ===\n")

    modified = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            filepath = os.path.join(dirpath, fname)
            if should_process(filepath):
                if process_file(filepath, dry_run):
                    modified += 1

    print(f"\n{'Would modify' if dry_run else 'Modified'}: {modified} files")


if __name__ == "__main__":
    main()
