#!/usr/bin/env python3
"""
Collect high-quality English instruct/chat datasets, normalize to unified
ChatML-compatible JSON format, deduplicate, filter, save as parquet.

Usage:
    python scripts/prepare_instruct_chatml.py
    python scripts/prepare_instruct_chatml.py --output data/instruct_chatml_en.parquet
    python scripts/prepare_instruct_chatml.py --smoke-test
"""

import argparse
import hashlib
import json
import os
import time

from datasets import Dataset, load_dataset

DEFAULT_OUTPUT = "data/instruct_chatml_en.parquet"

# ---------------------------------------------------------------------------
# Adapter functions: each returns list of {"id", "source", "messages"}
# ---------------------------------------------------------------------------

def adapt_openhermes(max_rows=None):
    """teknium/OpenHermes-2.5 — ShareGPT format with 'from'/'value' keys."""
    print("[OpenHermes] Loading...", flush=True)
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rows = []
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    for i, ex in enumerate(ds):
        convos = ex.get("conversations", [])
        messages = []
        for turn in convos:
            role = role_map.get(turn.get("from", ""), turn.get("from", ""))
            content = turn.get("value", "").strip()
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})
        if messages:
            rows.append({"id": f"openhermes-{i}", "source": "teknium/OpenHermes-2.5", "messages": messages})
    print(f"[OpenHermes] {len(rows)} conversations", flush=True)
    return rows


def adapt_slimorca(max_rows=None):
    """Open-Orca/SlimOrca-Dedup — ShareGPT format."""
    print("[SlimOrca] Loading...", flush=True)
    ds = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rows = []
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    for i, ex in enumerate(ds):
        convos = ex.get("conversations", [])
        messages = []
        for turn in convos:
            role = role_map.get(turn.get("from", ""), turn.get("from", ""))
            content = turn.get("value", "").strip()
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})
        if messages:
            rows.append({"id": f"slimorca-{i}", "source": "Open-Orca/SlimOrca-Dedup", "messages": messages})
    print(f"[SlimOrca] {len(rows)} conversations", flush=True)
    return rows


def adapt_ultrachat(max_rows=None):
    """HuggingFaceH4/ultrachat_200k — already in messages format."""
    print("[UltraChat] Loading...", flush=True)
    rows = []
    for split_name in ["train_sft", "test_sft"]:
        try:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split_name)
        except Exception:
            continue
        if max_rows:
            ds = ds.select(range(min(max_rows, len(ds))))
        for i, ex in enumerate(ds):
            msgs = ex.get("messages", [])
            messages = []
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "").strip()
                if role in ("user", "assistant", "system") and content:
                    messages.append({"role": role, "content": content})
            if messages:
                rows.append({"id": f"ultrachat-{split_name}-{i}", "source": "HuggingFaceH4/ultrachat_200k", "messages": messages})
    print(f"[UltraChat] {len(rows)} conversations", flush=True)
    return rows


def adapt_wizardlm(max_rows=None):
    """WizardLM/WizardLM_evol_instruct_V2_196k — ShareGPT."""
    print("[WizardLM] Loading...", flush=True)
    ds = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rows = []
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    for i, ex in enumerate(ds):
        convos = ex.get("conversations", [])
        messages = []
        for turn in convos:
            role = role_map.get(turn.get("from", ""), turn.get("from", ""))
            content = turn.get("value", "").strip()
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})
        if messages:
            rows.append({"id": f"wizardlm-{i}", "source": "WizardLM/WizardLM_evol_instruct_V2_196k", "messages": messages})
    print(f"[WizardLM] {len(rows)} conversations", flush=True)
    return rows


def adapt_capybara(max_rows=None):
    """LDJnr/Capybara — ShareGPT with 'conversation' key."""
    print("[Capybara] Loading...", flush=True)
    ds = load_dataset("LDJnr/Capybara", split="train")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rows = []
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    for i, ex in enumerate(ds):
        convos = ex.get("conversation", ex.get("conversations", []))
        messages = []
        for turn in convos:
            role = turn.get("role", role_map.get(turn.get("from", ""), ""))
            if role in role_map:
                role = role_map[role]
            content = turn.get("content", turn.get("value", "")).strip()
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})
        if messages:
            rows.append({"id": f"capybara-{i}", "source": "LDJnr/Capybara", "messages": messages})
    print(f"[Capybara] {len(rows)} conversations", flush=True)
    return rows


def adapt_deita(max_rows=None):
    """hkust-nlp/deita-10k-v0 — ShareGPT."""
    print("[Deita] Loading...", flush=True)
    ds = load_dataset("hkust-nlp/deita-10k-v0", split="train_sft")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rows = []
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    for i, ex in enumerate(ds):
        convos = ex.get("conversations", [])
        messages = []
        for turn in convos:
            role = role_map.get(turn.get("from", ""), turn.get("from", ""))
            content = turn.get("value", "").strip()
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})
        if messages:
            rows.append({"id": f"deita-{i}", "source": "hkust-nlp/deita-10k-v0", "messages": messages})
    print(f"[Deita] {len(rows)} conversations", flush=True)
    return rows


def adapt_alpaca_gpt4(max_rows=None):
    """vicgalle/alpaca-gpt4 — Alpaca format (instruction/input/output)."""
    print("[AlpacaGPT4] Loading...", flush=True)
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rows = []
    for i, ex in enumerate(ds):
        instruction = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        output = ex.get("output", "").strip()
        if not instruction or not output:
            continue
        user_msg = f"{instruction}\n\n{inp}".strip() if inp else instruction
        messages = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": output}]
        rows.append({"id": f"alpaca-gpt4-{i}", "source": "vicgalle/alpaca-gpt4", "messages": messages})
    print(f"[AlpacaGPT4] {len(rows)} conversations", flush=True)
    return rows


def adapt_dolly(max_rows=None):
    """databricks/databricks-dolly-15k — instruction/context/response."""
    print("[Dolly] Loading...", flush=True)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    rows = []
    for i, ex in enumerate(ds):
        instruction = ex.get("instruction", "").strip()
        context = ex.get("context", "").strip()
        response = ex.get("response", "").strip()
        if not instruction or not response:
            continue
        user_msg = f"{instruction}\n\nContext: {context}".strip() if context else instruction
        messages = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": response}]
        rows.append({"id": f"dolly-{i}", "source": "databricks/databricks-dolly-15k", "messages": messages})
    print(f"[Dolly] {len(rows)} conversations", flush=True)
    return rows


def adapt_oasst2(max_rows=None):
    """OpenAssistant/oasst2 — tree structure, flatten by best-ranked paths, English only."""
    print("[OASST2] Loading...", flush=True)
    ds = load_dataset("OpenAssistant/oasst2", split="train")

    # Build tree: parent_id → children
    by_id = {}
    children_map = {}
    for ex in ds:
        msg_id = ex["message_id"]
        by_id[msg_id] = ex
        parent = ex.get("parent_id")
        if parent:
            children_map.setdefault(parent, []).append(msg_id)

    # Find root messages (English only)
    roots = [mid for mid, ex in by_id.items()
             if ex.get("parent_id") is None and ex.get("lang", "") == "en"]

    rows = []
    role_map = {"prompter": "user", "assistant": "assistant"}

    def best_child(parent_id):
        kids = children_map.get(parent_id, [])
        if not kids:
            return None
        # Pick highest-ranked child
        kids.sort(key=lambda mid: by_id[mid].get("rank", 999))
        return kids[0]

    for root_id in roots:
        messages = []
        current = root_id
        while current:
            ex = by_id[current]
            if ex.get("lang", "") != "en":
                break
            role = role_map.get(ex.get("role", ""), "")
            content = ex.get("text", "").strip()
            if role and content:
                messages.append({"role": role, "content": content})
            current = best_child(current)

        if len(messages) >= 2:
            rows.append({"id": f"oasst2-{root_id}", "source": "OpenAssistant/oasst2", "messages": messages})
            if max_rows and len(rows) >= max_rows:
                break

    print(f"[OASST2] {len(rows)} conversations", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Dedup + filter
# ---------------------------------------------------------------------------

def dedup_and_filter(all_rows, min_chars=10):
    """Deduplicate by MD5 of first user message; filter short messages."""
    seen = set()
    kept = []
    dup_count = 0
    short_count = 0
    for row in all_rows:
        # Find first user message
        first_user = ""
        for m in row["messages"]:
            if m["role"] == "user":
                first_user = m["content"]
                break
        if len(first_user) < min_chars:
            short_count += 1
            continue
        h = hashlib.md5(first_user.encode()).hexdigest()
        if h in seen:
            dup_count += 1
            continue
        seen.add(h)
        kept.append(row)
    print(f"[Dedup] Kept {len(kept)}, removed {dup_count} dupes, {short_count} short", flush=True)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ADAPTERS = [
    ("OpenHermes", adapt_openhermes),
    ("SlimOrca", adapt_slimorca),
    ("UltraChat", adapt_ultrachat),
    ("WizardLM", adapt_wizardlm),
    ("Capybara", adapt_capybara),
    ("Deita", adapt_deita),
    ("AlpacaGPT4", adapt_alpaca_gpt4),
    ("Dolly", adapt_dolly),
    ("OASST2", adapt_oasst2),
]


def main():
    parser = argparse.ArgumentParser(description="Collect instruct datasets → unified ChatML JSON parquet")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-test", action="store_true", help="Only 100 rows per dataset")
    parser.add_argument("--datasets", nargs="*", help="Subset of dataset names to process")
    args = parser.parse_args()

    max_rows = 100 if args.smoke_test else None
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_rows = []
    t0 = time.time()

    for name, adapter in ADAPTERS:
        if args.datasets and name.lower() not in [d.lower() for d in args.datasets]:
            continue
        try:
            rows = adapter(max_rows=max_rows)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[{name}] ERROR: {e}", flush=True)

    print(f"\nTotal before dedup: {len(all_rows)}", flush=True)
    all_rows = dedup_and_filter(all_rows)

    # Serialize messages to JSON string for parquet storage
    records = []
    for row in all_rows:
        records.append({
            "id": row["id"],
            "source": row["source"],
            "messages": json.dumps(row["messages"], ensure_ascii=False),
            "num_turns": len([m for m in row["messages"] if m["role"] == "assistant"]),
        })

    ds = Dataset.from_list(records)
    ds.to_parquet(args.output)
    elapsed = time.time() - t0
    print(f"\nSaved {len(ds)} rows to {args.output} in {elapsed:.1f}s", flush=True)

    # Source distribution
    from collections import Counter
    sources = Counter(r["source"] for r in records)
    print("\nSource distribution:")
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt:,}")


if __name__ == "__main__":
    main()
