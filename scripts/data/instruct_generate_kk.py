#!/usr/bin/env python3
"""Generate Kazakh instruct dataset from Alpaca via GPT-OSS-120B.

Resumable: saves progress per 1K chunk. Can restart on any machine.

Usage:
    python3 instruct_generate_kk.py --api_url http://localhost:15127 --output_dir /root/instruct_kk
    python3 instruct_generate_kk.py --api_url http://localhost:15127 --output_dir /root/instruct_kk --resume
"""

import argparse
import json
import os
import re
import time
import urllib.request

# ── Alpaca filter ────────────────────────────────────────────────────────────

SKIP_CATEGORIES = [
    "code", "program", "python", "javascript", "html", "css", "sql", "algorithm",
    "function", "variable", "debug", "compile", "syntax",
    "rhyme", "alliteration", "acrostic", "limerick", "haiku",
    "spell", "abbreviation",
]


def should_skip(instruction, inp=""):
    """Skip code, English-specific, and trivial tasks."""
    text = (instruction + " " + inp).lower()
    if any(kw in text for kw in SKIP_CATEGORIES):
        return True
    if len(instruction.split()) < 3:
        return True
    return False


def load_alpaca():
    """Download and filter Alpaca dataset."""
    cache = "/tmp/alpaca_data.json"
    if not os.path.exists(cache):
        print("Downloading Alpaca dataset...")
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

    print(f"Alpaca: {len(data)} total, {len(filtered)} after filter")
    return filtered


# ── API calls ────────────────────────────────────────────────────────────────

def call_api(url, messages, max_tokens=1024, temperature=0.7):
    """Call OpenAI-compatible chat API.

    GPT-OSS-120B puts output in reasoning_content when using --jinja.
    """
    data = json.dumps({
        "model": "GPT-OSS-120B",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    msg = result["choices"][0]["message"]
    content = msg.get("content", "") or ""
    return content


def rewrite_batch(url, instructions, batch_size=10):
    """Rewrite a batch of English instructions to Kazakh."""
    results = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {item['instruction']}"
                             + (f"\n   Input: {item['input']}" if item['input'] else "")
                             for j, item in enumerate(batch))

        prompt = f"""Төмендегі {len(batch)} ағылшынша тапсырманы қазақ тіліне БЕЙІМДЕ.

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

        try:
            response = call_api(url, [
                {"role": "system", "content": "Сен тәжірибелі аудармашысың. Тапсырмаларды қазақ тіліне бейімдейсің."},
                {"role": "user", "content": prompt},
            ], max_tokens=2048, temperature=0.5)

            # Parse numbered lines
            lines = response.strip().split("\n")
            parsed = []
            current = ""
            current_input = ""
            for line in lines:
                m = re.match(r"^\d+[\.\)]\s*(.+)", line.strip())
                if m:
                    if current:
                        parsed.append({"instruction_kk": current.strip(), "input_kk": current_input.strip()})
                    current = m.group(1)
                    current_input = ""
                elif line.strip().upper().startswith("INPUT:"):
                    current_input = line.strip()[6:].strip()
                elif current:
                    current += " " + line.strip()
            if current:
                parsed.append({"instruction_kk": current.strip(), "input_kk": current_input.strip()})

            # Match back to original
            for j, item in enumerate(batch):
                if j < len(parsed):
                    kk = parsed[j]["instruction_kk"]
                    kk_input = parsed[j].get("input_kk", "")
                    # Quality check: must be mostly Cyrillic
                    cyrillic = sum(1 for c in kk if '\u0400' <= c <= '\u04FF')
                    if cyrillic < len(kk) * 0.5:
                        continue
                    if kk == item["instruction"]:  # Not adapted
                        continue
                    results.append({
                        "instruction_en": item["instruction"],
                        "input_en": item["input"],
                        "instruction_kk": kk,
                        "input_kk": kk_input,
                    })

        except Exception as e:
            print(f"  Rewrite error at {i}: {e}")
            time.sleep(2)

    return results


def generate_answer(url, instruction_kk, input_kk=""):
    """Generate answer in Kazakh for a rewritten instruction."""
    prompt = instruction_kk
    if input_kk:
        prompt += f"\n\n{input_kk}"

    try:
        answer = call_api(url, [
            {"role": "user", "content": prompt},
        ], max_tokens=1024, temperature=0.7)

        # Quality: must be mostly Cyrillic, not too short
        cyrillic = sum(1 for c in answer if '\u0400' <= c <= '\u04FF')
        if len(answer) < 20 or cyrillic < len(answer) * 0.3:
            return None
        return answer
    except Exception as e:
        print(f"  Answer error: {e}")
        time.sleep(2)
        return None


# ── Progress management ──────────────────────────────────────────────────────

def load_progress(output_dir):
    """Load progress from previous run."""
    progress_file = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            return json.load(f)
    return {"last_chunk": -1, "total_pairs": 0, "rewrite_done": False}


def save_progress(output_dir, progress):
    """Save progress."""
    progress_file = os.path.join(output_dir, "progress.json")
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://localhost:15127")
    parser.add_argument("--output_dir", default="/root/instruct_kk")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rewrite_batch", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "chunks"), exist_ok=True)

    # Load or filter Alpaca
    filtered_cache = os.path.join(args.output_dir, "alpaca_filtered.json")
    if os.path.exists(filtered_cache):
        with open(filtered_cache) as f:
            filtered = json.load(f)
        print(f"Loaded {len(filtered)} cached filtered instructions")
    else:
        filtered = load_alpaca()
        with open(filtered_cache, "w") as f:
            json.dump(filtered, f, ensure_ascii=False)
        print(f"Saved {len(filtered)} filtered instructions")

    # Split into chunks
    chunks = []
    for i in range(0, len(filtered), args.chunk_size):
        chunks.append(filtered[i:i + args.chunk_size])
    total_chunks = len(chunks)
    print(f"Total: {len(filtered)} instructions in {total_chunks} chunks of {args.chunk_size}")

    # Load progress
    progress = load_progress(args.output_dir)
    if args.resume:
        start_chunk = progress["last_chunk"] + 1
        print(f"Resuming from chunk {start_chunk} (last completed: {progress['last_chunk']})")
    else:
        start_chunk = 0

    total_pairs = progress.get("total_pairs", 0)

    for chunk_idx in range(start_chunk, total_chunks):
        chunk = chunks[chunk_idx]
        chunk_file = os.path.join(args.output_dir, "chunks", f"chunk_{chunk_idx:03d}.jsonl")

        print(f"\n{'='*60}")
        print(f"Chunk {chunk_idx}/{total_chunks} ({len(chunk)} instructions)")
        print(f"{'='*60}")

        # Step 1: Rewrite instructions to Kazakh
        print("  Step 1: Rewriting to Kazakh...")
        t0 = time.time()
        rewritten = rewrite_batch(args.api_url, chunk, batch_size=args.rewrite_batch)
        t1 = time.time()
        print(f"  Rewritten: {len(rewritten)}/{len(chunk)} in {t1-t0:.0f}s")

        # Step 2: Generate answers
        print("  Step 2: Generating answers...")
        pairs = []
        for j, item in enumerate(rewritten):
            answer = generate_answer(args.api_url, item["instruction_kk"], item.get("input_kk", ""))
            if answer:
                pairs.append({
                    "instruction_kk": item["instruction_kk"],
                    "input_kk": item.get("input_kk", ""),
                    "output_kk": answer,
                    "instruction_en": item["instruction_en"],
                    "input_en": item.get("input_en", ""),
                    "source": "alpaca_rewrite",
                })

            if (j + 1) % 50 == 0:
                print(f"    [{j+1}/{len(rewritten)}] {len(pairs)} good answers")

        t2 = time.time()
        print(f"  Answers: {len(pairs)}/{len(rewritten)} in {t2-t1:.0f}s")

        # Save chunk
        with open(chunk_file, "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        total_pairs += len(pairs)

        # Update progress
        progress = {
            "last_chunk": chunk_idx,
            "total_chunks": total_chunks,
            "total_pairs": total_pairs,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_progress(args.output_dir, progress)

        print(f"  Saved {len(pairs)} pairs to {chunk_file}")
        print(f"  Total so far: {total_pairs} pairs")

        # Random quality check every 5 chunks
        if (chunk_idx + 1) % 5 == 0 and pairs:
            import random
            sample = random.sample(pairs, min(3, len(pairs)))
            print("\n  --- Quality check ---")
            for s in sample:
                print(f"    EN: {s['instruction_en'][:80]}")
                print(f"    KK: {s['instruction_kk'][:80]}")
                print(f"    ANS: {s['output_kk'][:80]}")
                print()

    print(f"\n{'='*60}")
    print(f"DONE: {total_pairs} total pairs in {total_chunks} chunks")
    print(f"Saved to {args.output_dir}/chunks/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
