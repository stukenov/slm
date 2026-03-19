#!/usr/bin/env python3
"""Migrate saken-tukenov HF repos to SozKZ naming standard."""

import argparse
import sys
from huggingface_hub import HfApi

OWNER = "saken-tukenov"

# (old_name, new_name)
MODEL_RENAMES = [
    ("kazakh-llama-30m", "sozkz-core-llama-30m-kk-base-v1"),
    ("kazakh-llama-50m", "sozkz-core-llama-50m-kk-base-v1"),
    ("kazakh-llama-50m-v2", "sozkz-core-llama-50m-kk-base-v2"),
    ("kazakh-llama-50m-balanced", "sozkz-core-llama-50m-kk-balanced-v1"),
    ("kazakh-llama-150m-balanced", "sozkz-core-llama-150m-kk-balanced-v1"),
    ("llama-kazakh-150m", "sozkz-core-llama-150m-kk-base-v1"),
    ("slm-kk-pythia14m-dapt-13k", "sozkz-core-pythia-14m-kk-dapt-v1"),
    ("kazakh-t5-50m", "sozkz-seq-t5-50m-kk-base-v1"),
    ("kazakh-t5-sp-32k", "sozkz-vocab-sp-32k-kk-t5-v1"),
    ("kazakh-gec-50m", "sozkz-fix-mt5-50m-kk-gec-v1"),
    ("kazakh-gec-mt5-base-run13-finetune", "sozkz-fix-mt5b-kk-gec-run13-v1"),
    ("kazakh-moe-200M-A50M", "sozkz-moe-mix-200m-kk-base-v1"),
    ("kazakh-moe-160M-A50M-domain", "sozkz-moe-mix-160m-kk-domain-v1"),
    ("kazakh-gec-morphology-50m", "sozkz-fix-mt5-50m-kk-morph-v1"),
]

DATASET_RENAMES = [
    ("kazakh-synthetic-gec-datasets", "sozkz-corpus-synthetic-kk-gec-v1"),
    ("kazakh-llama-50m-tokenized", "sozkz-corpus-tokenized-kk-llama50m-v1"),
    ("gazeta-kazakh", "sozkz-corpus-raw-kk-gazeta-v1"),
    ("kazakh-balanced-gpt2-style", "sozkz-corpus-balanced-kk-gpt2-v1"),
    ("kazakh-clean-pretrain-text-v2", "sozkz-corpus-clean-kk-text-v2"),
    ("kazakh-clean-pretrain-v2", "sozkz-corpus-clean-kk-pretrain-v2"),
    ("kazakh-moe-instruction-sft", "sozkz-corpus-synthetic-kk-moe-sft-v1"),
    ("kazakh-moe-domain-labeled", "sozkz-corpus-balanced-kk-moe-domain-v1"),
    ("kazakh-t5-50m-tokenized", "sozkz-corpus-tokenized-kk-t5-50m-v1"),
    ("kazakh-gec-morphology-tokenized", "sozkz-corpus-tokenized-kk-morph-v1"),
]

# Repos to mark as DEPRECATED (no rename, just update README)
DEPRECATE_MODELS = [
    "kazakh-llama-50m-smoke",
    "kazakh-gec-mt5-small-run1-processed",
    "kazakh-gec-mt5-small-run2-processed-v2",
    "kazakh-gec-mt5-small-run3-finetune-v2",
    "kazakh-gec-mt5-small-run6-grammar-focused",
    "kazakh-gec-mt5-small-run7-grammar-v2",
    "kazakh-gec-mt5-base-run4-processed-v2",
    "kazakh-gec-mt5-base-run8-grammar-combined",
    "kazakh-gec-mt5-base-run9-grammar-balanced-v2",
    "kazakh-gec-mt5-base-run10-hybrid-v1",
    "kazakh-gec-mt5-base-run11-kazsandra",
    "kazakh-gec-mt5-base-run12-kazsandra-new",
    "kazakh-gec-mt5-models",
]

DEPRECATE_DATASETS = [
    "kazakh-clean-pretrain-text",
    "kazakh-clean-pretrain",
]

# Old repos that already have sozkz- equivalents (mark deprecated)
ALREADY_MIGRATED_MODELS = [
    "kazakh-gpt2-50k",
    "kazakh-gpt2-8m",
    "kazakh-gpt2-30m",
    "kazakh-gpt2-60m",
    "kazakh-bpe-32k",
]

DEPRECATION_README = """\
---
tags:
- deprecated
---

# DEPRECATED

This repository has been deprecated and is no longer maintained.

Please use the new SozKZ-named repository instead. Check [saken-tukenov on HuggingFace](https://huggingface.co/saken-tukenov) for the latest versions.
"""

DEPRECATION_README_WITH_NEW = """\
---
tags:
- deprecated
---

# DEPRECATED → [{owner}/{new_name}](https://huggingface.co/{owner}/{new_name})

This repository has been renamed. Please use **[{owner}/{new_name}](https://huggingface.co/{owner}/{new_name})** instead.
"""


def rename_repos(api: HfApi, renames: list, repo_type: str, dry_run: bool):
    for old, new in renames:
        old_id = f"{OWNER}/{old}"
        new_id = f"{OWNER}/{new}"
        print(f"  {old_id} -> {new_id}", end=" ... ")
        if dry_run:
            print("DRY RUN")
            continue
        try:
            api.move_repo(from_id=old_id, to_id=new_id, repo_type=repo_type)
            print("OK")
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower():
                print("SKIP (not found — maybe already renamed)")
            else:
                print(f"ERROR: {e}")


def deprecate_repos(api: HfApi, repos: list, repo_type: str, dry_run: bool, new_name_map: dict = None):
    for name in repos:
        repo_id = f"{OWNER}/{name}"
        print(f"  {repo_id}", end=" ... ")
        if dry_run:
            print("DRY RUN")
            continue
        try:
            if new_name_map and name in new_name_map:
                content = DEPRECATION_README_WITH_NEW.format(owner=OWNER, new_name=new_name_map[name])
            else:
                content = DEPRECATION_README
            api.upload_file(
                path_or_fileobj=content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type=repo_type if repo_type != "model" else None,
                commit_message="DEPRECATED: migrated to SozKZ naming",
            )
            print("OK")
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower():
                print("SKIP (not found)")
            else:
                print(f"ERROR: {e}")


def verify(api: HfApi):
    print("\n=== Verification ===")
    print("\nModels:")
    for m in api.list_models(author=OWNER):
        tag = "OK" if m.id.split("/")[1].startswith("sozkz-") else "OLD"
        print(f"  [{tag}] {m.id}")
    print("\nDatasets:")
    for d in api.list_datasets(author=OWNER):
        tag = "OK" if d.id.split("/")[1].startswith("sozkz-") else "OLD"
        print(f"  [{tag}] {d.id}")


def main():
    parser = argparse.ArgumentParser(description="Migrate HF repos to SozKZ naming")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--verify-only", action="store_true", help="Only verify current state")
    parser.add_argument("--step", choices=["rename", "deprecate", "all"], default="all")
    args = parser.parse_args()

    api = HfApi()

    if args.verify_only:
        verify(api)
        return

    # Build map of old->new for already-migrated repos (for deprecation links)
    # These are the 5 GPT2 models + tokenizer already migrated
    already_migrated_map = {
        "kazakh-gpt2-50k": "sozkz-core-gpt2-50k-kk-base-v1",
        "kazakh-gpt2-8m": "sozkz-core-gpt2-8m-kk-base-v1",
        "kazakh-gpt2-30m": "sozkz-core-gpt2-30m-kk-base-v1",
        "kazakh-gpt2-60m": "sozkz-core-gpt2-60m-kk-base-v1",
        "kazakh-bpe-32k": "sozkz-vocab-bpe-32k-kk-base-v1",
    }

    if args.step in ("rename", "all"):
        print("=== Renaming Models ===")
        rename_repos(api, MODEL_RENAMES, "model", args.dry_run)

        print("\n=== Renaming Datasets ===")
        rename_repos(api, DATASET_RENAMES, "dataset", args.dry_run)

    if args.step in ("deprecate", "all"):
        print("\n=== Deprecating Models ===")
        deprecate_repos(api, DEPRECATE_MODELS, "model", args.dry_run)

        print("\n=== Deprecating Datasets ===")
        deprecate_repos(api, DEPRECATE_DATASETS, "dataset", args.dry_run)

        print("\n=== Deprecating Already-Migrated Models ===")
        deprecate_repos(api, ALREADY_MIGRATED_MODELS, "model", args.dry_run, already_migrated_map)

    if not args.dry_run:
        verify(api)

    print("\nDone!")


if __name__ == "__main__":
    main()
