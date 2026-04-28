#!/usr/bin/env python3
"""Generate Kazakh instruct dataset from Alpaca via CloudRift Inference API
(Qwen/Qwen3.5-122B-A10B-FP8, non-thinking mode).

Adapted from instruct_generate_kk.py (which used local GPT-OSS-120B on kaznu).

Key differences:
  * Uses public CloudRift Inference endpoint (OpenAI-compatible)
  * REQUIRES chat_template_kwargs: {enable_thinking: false} — without it
    Qwen3.5 burns 60× more tokens on chain-of-thought in `reasoning` field
  * Concurrent requests via ThreadPoolExecutor (the remote API benefits
    from parallelism; the local GPT-OSS was single-stream)
  * Tracks cost in real time using current pricing ($0.25 in / $1.50 out per 1M)

Resumable: saves progress per chunk. Can restart.

Usage:
    # smoke test (20 instructions)
    python3 instruct_generate_kk_cloudrift.py --limit 20 --output_dir ./instruct_kk_smoke

    # full run with concurrency
    python3 instruct_generate_kk_cloudrift.py --output_dir /root/instruct_kk_qwen --concurrency 12

    # resume
    python3 instruct_generate_kk_cloudrift.py --output_dir /root/instruct_kk_qwen --resume
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Config ───────────────────────────────────────────────────────────────────

API_URL = "https://inference.cloudrift.ai/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"

# Prices (from dashboard, not API /v1/models which returns cents without units)
PRICE_IN_PER_1M = 0.25
PRICE_OUT_PER_1M = 1.50


def load_api_key():
    key_path = os.path.expanduser("~/.config/cloudrift/api_key")
    if os.path.exists(key_path):
        return open(key_path).read().strip()
    env = os.environ.get("CLOUDRIFT_API_KEY")
    if env:
        return env.strip()
    raise RuntimeError(f"No API key at {key_path} and CLOUDRIFT_API_KEY env not set")


API_KEY = load_api_key()


# ── Alpaca filter (kept identical to original) ───────────────────────────────

SKIP_CATEGORIES = [
    "code", "program", "python", "javascript", "html", "css", "sql", "algorithm",
    "function", "variable", "debug", "compile", "syntax",
    "rhyme", "alliteration", "acrostic", "limerick", "haiku",
    "spell", "abbreviation",
]


def should_skip(instruction, inp=""):
    text = (instruction + " " + inp).lower()
    if any(kw in text for kw in SKIP_CATEGORIES):
        return True
    if len(instruction.split()) < 3:
        return True
    return False


def load_alpaca():
    cache = "/tmp/alpaca_data.json"
    if not os.path.exists(cache):
        print("Downloading Alpaca dataset...", flush=True)
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        urllib.request.urlretrieve(url, cache)
    with open(cache) as f:
        data = json.load(f)
    filtered = []
    for item in data:
        inst = item.get("instruction", "")
        inp = item.get("input", "")
        if not should_skip(inst, inp):
            filtered.append({"instruction": inst, "input": inp})
    print(f"Alpaca: {len(data)} total, {len(filtered)} after filter", flush=True)
    return filtered


# ── Cost tracker (thread-safe) ───────────────────────────────────────────────

class CostTracker:
    def __init__(self):
        self.in_tokens = 0
        self.out_tokens = 0
        self.requests = 0
        self.errors = 0
        self.lock = Lock()

    def add(self, in_tok, out_tok):
        with self.lock:
            self.in_tokens += in_tok
            self.out_tokens += out_tok
            self.requests += 1

    def err(self):
        with self.lock:
            self.errors += 1

    def cost_usd(self):
        return (self.in_tokens * PRICE_IN_PER_1M / 1e6
                + self.out_tokens * PRICE_OUT_PER_1M / 1e6)

    def summary(self):
        return (f"req={self.requests} err={self.errors} "
                f"in={self.in_tokens:,} out={self.out_tokens:,} "
                f"cost=${self.cost_usd():.4f}")


TRACKER = CostTracker()


# ── API calls ────────────────────────────────────────────────────────────────

def call_api(messages, max_tokens=1024, temperature=0.7, retries=3):
    """Call CloudRift Inference (Qwen3.5-122B) in non-thinking mode."""
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        # CRITICAL: disable thinking — otherwise content=null and tokens burn
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API_URL, data=body, method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            usage = data.get("usage") or {}
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            TRACKER.add(in_tok, out_tok)
            return content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    TRACKER.err()
    raise last_err


# ── Rewrite step: EN instructions → KK (batched) ─────────────────────────────

REWRITE_SYSTEM = "Сен тәжірибелі аудармашысың. Тапсырмаларды қазақ тіліне бейімдейсің."


def build_rewrite_prompt(batch):
    numbered = "\n".join(
        f"{j+1}. {item['instruction']}"
        + (f"\n   Input: {item['input']}" if item["input"] else "")
        for j, item in enumerate(batch)
    )
    return f"""Төмендегі {len(batch)} ағылшынша тапсырманы қазақ тіліне БЕЙІМДЕ.

ЕРЕЖЕЛЕР:
- Дословно АУДАРМА, табиғи қазақша қайта жаз
- Ағылшын есімдерін қазақ есімдеріне ауыстыр (John→Арман, Mary→Айгүл, New York→Алматы, т.б.)
- Мәдени контекстті қазақстандық ет
- Тапсырманы ОРЫНДАМА, тек қайта жаз
- Нәтиже ТЕК қазақша болсын
- Input бар болса, оны да бейімде
- Формат: нөмір. тапсырма (егер input бар болса, жаңа жолдан INPUT: деп жаз)

{numbered}

Қазақша:"""


def parse_rewrite(response, batch):
    lines = response.strip().split("\n")
    parsed = []
    current = ""
    current_input = ""
    for line in lines:
        m = re.match(r"^\d+[\.\)]\s*(.+)", line.strip())
        if m:
            if current:
                parsed.append({"instruction_kk": current.strip(),
                               "input_kk": current_input.strip()})
            current = m.group(1)
            current_input = ""
        elif line.strip().upper().startswith("INPUT:"):
            current_input = line.strip()[6:].strip()
        elif current:
            current += " " + line.strip()
    if current:
        parsed.append({"instruction_kk": current.strip(),
                       "input_kk": current_input.strip()})

    results = []
    for j, item in enumerate(batch):
        if j >= len(parsed):
            continue
        kk = parsed[j]["instruction_kk"]
        kk_input = parsed[j].get("input_kk", "")
        cyrillic = sum(1 for c in kk if "\u0400" <= c <= "\u04FF")
        if cyrillic < len(kk) * 0.5:
            continue
        if kk == item["instruction"]:
            continue
        results.append({
            "instruction_en": item["instruction"],
            "input_en": item["input"],
            "instruction_kk": kk,
            "input_kk": kk_input,
        })
    return results


def rewrite_one_batch(batch):
    try:
        prompt = build_rewrite_prompt(batch)
        resp = call_api([
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": prompt},
        ], max_tokens=2048, temperature=0.5)
        return parse_rewrite(resp, batch)
    except Exception as e:
        print(f"  rewrite batch failed: {e}", flush=True)
        return []


# ── Answer step: KK instruction → KK answer ──────────────────────────────────

def generate_answer(instruction_kk, input_kk=""):
    prompt = instruction_kk
    if input_kk:
        prompt += f"\n\n{input_kk}"
    try:
        answer = call_api([
            {"role": "user", "content": prompt},
        ], max_tokens=1024, temperature=0.7)
        cyrillic = sum(1 for c in answer if "\u0400" <= c <= "\u04FF")
        if len(answer) < 20 or cyrillic < len(answer) * 0.3:
            return None
        return answer
    except Exception as e:
        print(f"  answer failed: {e}", flush=True)
        return None


# ── Progress ─────────────────────────────────────────────────────────────────

def load_progress(output_dir):
    p = os.path.join(output_dir, "progress.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"last_chunk": -1, "total_pairs": 0, "spent_usd": 0.0}


def save_progress(output_dir, progress):
    p = os.path.join(output_dir, "progress.json")
    with open(p, "w") as f:
        json.dump(progress, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./instruct_kk_qwen")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--rewrite_batch", type=int, default=10,
                        help="Instructions per rewrite call (batched in prompt)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Parallel API calls")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max total instructions to process (0 = all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget_usd", type=float, default=0.0,
                        help="Abort run when spent exceeds this (0 = no limit)")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "chunks"), exist_ok=True)

    # Load / cache filtered alpaca
    cache = os.path.join(args.output_dir, "alpaca_filtered.json")
    if os.path.exists(cache):
        filtered = json.load(open(cache))
        print(f"Loaded {len(filtered)} cached filtered instructions", flush=True)
    else:
        filtered = load_alpaca()
        with open(cache, "w") as f:
            json.dump(filtered, f, ensure_ascii=False)
        print(f"Saved {len(filtered)} filtered instructions", flush=True)

    if args.limit:
        filtered = filtered[: args.limit]
        print(f"Limited to first {args.limit}", flush=True)

    # Chunks
    chunks = [filtered[i:i + args.chunk_size]
              for i in range(0, len(filtered), args.chunk_size)]
    total_chunks = len(chunks)
    print(f"Total: {len(filtered)} instructions in {total_chunks} chunks "
          f"of {args.chunk_size}", flush=True)

    progress = load_progress(args.output_dir)
    start_chunk = progress["last_chunk"] + 1 if args.resume else 0
    if args.resume:
        print(f"Resuming from chunk {start_chunk}", flush=True)
        # Seed tracker with prior spend (approximate — only dollars, not tokens)
        TRACKER.in_tokens = 0
        TRACKER.out_tokens = 0
    total_pairs = progress.get("total_pairs", 0)
    prior_spent = progress.get("spent_usd", 0.0)

    for chunk_idx in range(start_chunk, total_chunks):
        chunk = chunks[chunk_idx]
        chunk_file = os.path.join(args.output_dir, "chunks",
                                  f"chunk_{chunk_idx:03d}.jsonl")

        print(f"\n{'='*60}", flush=True)
        print(f"Chunk {chunk_idx}/{total_chunks - 1} ({len(chunk)} instructions)",
              flush=True)
        print(f"{'='*60}", flush=True)

        # Step 1: Rewrite (batched, concurrent)
        print("  Step 1: Rewriting to Kazakh (concurrent)...", flush=True)
        t0 = time.time()
        batches = [chunk[i:i + args.rewrite_batch]
                   for i in range(0, len(chunk), args.rewrite_batch)]
        rewritten = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(rewrite_one_batch, b) for b in batches]
            for i, f in enumerate(as_completed(futures)):
                try:
                    rewritten.extend(f.result())
                except Exception as e:
                    print(f"    batch err: {e}", flush=True)
                if (i + 1) % 10 == 0:
                    print(f"    rewrite {i+1}/{len(batches)} batches  "
                          f"[{TRACKER.summary()}]", flush=True)
        t1 = time.time()
        print(f"  Rewritten: {len(rewritten)}/{len(chunk)} in {t1-t0:.0f}s",
              flush=True)

        # Step 2: Answer (concurrent)
        print("  Step 2: Generating answers (concurrent)...", flush=True)
        pairs = []

        def _one(item):
            ans = generate_answer(item["instruction_kk"],
                                  item.get("input_kk", ""))
            if ans:
                return {
                    "instruction_kk": item["instruction_kk"],
                    "input_kk": item.get("input_kk", ""),
                    "output_kk": ans,
                    "instruction_en": item["instruction_en"],
                    "input_en": item.get("input_en", ""),
                    "source": "alpaca_rewrite_qwen35_122b",
                }
            return None

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(_one, it) for it in rewritten]
            done = 0
            for f in as_completed(futures):
                done += 1
                try:
                    r = f.result()
                    if r:
                        pairs.append(r)
                except Exception:
                    pass
                if done % 50 == 0:
                    print(f"    answered {done}/{len(rewritten)} "
                          f"({len(pairs)} good) [{TRACKER.summary()}]",
                          flush=True)

        t2 = time.time()
        print(f"  Answers: {len(pairs)}/{len(rewritten)} in {t2-t1:.0f}s",
              flush=True)

        # Save chunk
        with open(chunk_file, "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        total_pairs += len(pairs)
        current_spent = prior_spent + TRACKER.cost_usd()

        save_progress(args.output_dir, {
            "last_chunk": chunk_idx,
            "total_chunks": total_chunks,
            "total_pairs": total_pairs,
            "spent_usd": current_spent,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tracker": TRACKER.summary(),
        })

        print(f"  Saved {len(pairs)} pairs  total={total_pairs}  "
              f"SPENT=${current_spent:.4f}", flush=True)

        # Budget guard
        if args.budget_usd and current_spent >= args.budget_usd:
            print(f"\n!!! BUDGET EXCEEDED (${current_spent:.2f} "
                  f">= ${args.budget_usd:.2f}) — stopping", flush=True)
            break

        # Sample quality check every 5 chunks
        if (chunk_idx + 1) % 5 == 0 and pairs:
            import random
            sample = random.sample(pairs, min(2, len(pairs)))
            print("\n  --- Quality sample ---", flush=True)
            for s in sample:
                print(f"    EN:  {s['instruction_en'][:90]}", flush=True)
                print(f"    KK:  {s['instruction_kk'][:90]}", flush=True)
                print(f"    ANS: {s['output_kk'][:120]}", flush=True)
                print(flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"DONE: {total_pairs} pairs", flush=True)
    print(f"Final tracker: {TRACKER.summary()}", flush=True)
    print(f"Total spent (including prior runs): "
          f"${prior_spent + TRACKER.cost_usd():.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
