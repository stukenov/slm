#!/usr/bin/env python3
"""GEC v3: Optimized generation with reasoning_effort=low.

Strategy: send 1 seed → model outputs ONLY the errored version → target = original seed.
Server must use chat_template_low.jinja (reasoning_effort=low).

Optimizations vs v2:
  - reasoning_effort=low (282 tok vs 3000+ tok per request)
  - 2 parallel workers
  - Resumable via .progress file
  - Strict quality filters

Usage:
    python3 gec_generate_v3.py --url http://localhost:15127 --output /root/gec_v2.jsonl --seeds_file /root/gec_seeds.txt
    python3 gec_generate_v3.py --url http://localhost:15127 --output /root/gec_v2.jsonl --seeds_file /root/gec_seeds.txt --resume
"""

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Error types ─────────────────────────────────────────────────────────────

ERROR_TYPES = {
    "1_septik": {
        "name": "септік жалғау",
        "tag": "<септік>",
        "desc": "Зат есімнің септік жалғауын дұрыс емес септікке ауыстыр (мысалы: барыс орнына табыс, шығыс орнына жатыс).",
        "example_bad": "Мен мектепте бардым.",
        "example_good": "Мен мектепке бардым.",
    },
    "2_zhiktik": {
        "name": "жіктік жалғау",
        "tag": "<жіктік>",
        "desc": "Етістіктің жақ жалғауын дұрыс емеске ауыстыр (мысал��: 3-жақ орнына 1-жақ).",
        "example_bad": "Ол мектепке бардым.",
        "example_good": "Ол мектепке барды.",
    },
    "3_taueldi": {
        "name": "тәуелдік жалғау",
        "tag": "<тәуелдік>",
        "desc": "Тәуелдік жалғауын алып таста немесе дұрыс емес жаққа ауыстыр.",
        "example_bad": "Менің кітап үстелде жатыр.",
        "example_good": "Менің кітабым үстелде жатыр.",
    },
    "4_undestim": {
        "name": "дауысты үндесім",
        "tag": "<үндесім>",
        "desc": "Жалғаудағы дауысты үндесімді бұз (-тен→-тан, -да→-де, -лар→-лер керісінше).",
        "example_bad": "Мектептан шықтым.",
        "example_good": "Мектептен шықтым.",
    },
    "5_koptik": {
        "name": "көптік жалғау",
        "tag": "<көптік>",
        "desc": "Көптік жалғауды дұрыс емес нұсқаға ауыстыр.",
        "example_bad": "Балалер бақшаға барды.",
        "example_good": "Балалар бақшаға барды.",
    },
    "6_shaq": {
        "name": "шақ жалғау",
        "tag": "<шақ>",
        "desc": "Етістік шағын дұрыс емеске ауыстыр (мысалы: өткен шақ орнына осы шақ).",
        "example_bad": "Мен ертең кітап оқыдым.",
        "example_good": "Мен ертең кітап оқимын.",
    },
    "7_bolymsy": {
        "name": "болымсыз етістік",
        "tag": "<болымсыз>",
        "desc": "Болымсыздық жұрнағын дұрыс емес қос немесе ауыстыр.",
        "example_bad": "Ол мектепке бармеді.",
        "example_good": "Ол мектепке бармады.",
    },
    "8_shylau": {
        "name": "шылау",
        "tag": "<шылау>",
        "desc": "Шылау сөзді дұрыс емеске ауыстыр (мысалы: дейін→кейін, туралы→арқылы).",
        "example_bad": "Ол мектепке кейін келді.",
        "example_good": "Ол мектепке дейін келді.",
    },
    "9_soz_tartibi": {
        "name": "сөз тәртібі",
        "tag": "<тәртіп>",
        "desc": "SOV сөз тәртібін бұз — бастауыш, толықтауыш немесе баяндауыш орнын ауыстыр.",
        "example_bad": "Мен оқыдым кітапты.",
        "example_good": "Мен кітапты оқыдым.",
    },
    "10_emle": {
        "name": "емле қатесі",
        "tag": "<емле>",
        "desc": "1-2 әріпті ауыстыр, алып таста немесе артық қос (табиғи опечатка).",
        "example_bad": "Қазақсатн тәуелсіздігін алды.",
        "example_good": "Қазақстан тәуелсіздігін алды.",
    },
}

UNIVERSAL_TAG = "<грамматика>"


def build_prompt(et_key, seed):
    """Build prompt for single seed — model outputs ONLY the errored sentence."""
    et = ERROR_TYPES[et_key]
    return f"""Осы сөйлемде "{et['name']}" қатесін қос. {et['desc']} Тек бір сөзді өзгерт.

Мысал:
ҚАТЕ: {et['example_bad']}
ДҰРЫС: {et['example_good']}
---

Дұрыс: {seed}

ҚАТЕ:"""


def call_api(url, prompt, max_tokens=1000, temperature=0.5):
    """Call llama-server chat API."""
    data = json.dumps({
        "model": "GPT-OSS-120B",
        "messages": [
            {"role": "system", "content": "Сен қазақ тілінің грамматика маманысың. Тапсырманы орында."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=180)
    result = json.loads(resp.read())
    msg = result["choices"][0]["message"]
    content = msg.get("content", "") or ""
    if not content.strip():
        content = msg.get("reasoning_content", "") or ""
    return content.strip()


def clean_line(line):
    """Strip markdown, labels, and noise from a line."""
    # Remove ҚАТЕ:/ДҰРЫС: prefixes
    line = re.sub(r'^(ҚАТЕ|Қате|қате|ДҰРЫС|Дұрыс)\s*:\s*', '', line)
    # Remove Қат-Е: malformed prefix
    line = re.sub(r'^Қат-Е:\s*', '', line)
    # Remove markdown bold
    line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
    # Remove leading ** or *
    line = re.sub(r'^\*+\s*', '', line)
    # Remove trailing ** or *
    line = re.sub(r'\s*\*+$', '', line)
    # Remove parenthetical notes like "(қате)" "(дұрыс)"
    line = re.sub(r'\s*\((?:қате|дұрыс|error|correct)[^)]*\)\s*', ' ', line, flags=re.IGNORECASE)
    # Collapse whitespace
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def parse_error_sentence(content, seed):
    """Extract errored sentence from model output. Target = original seed."""
    content = content.strip()
    if not content:
        return None

    lines = [l.strip() for l in content.split("\n") if l.strip()]

    for line in lines:
        line = clean_line(line)

        # Skip junk
        if not line or len(line) < 10:
            continue
        if '...' in line or '---' in line:
            continue
        if line == seed:
            continue  # No change = useless
        # Skip meta-text (model explaining what it did)
        if 'сөйлемде' in line.lower() and 'қатесін' in line.lower():
            continue
        if 'өзгерт' in line.lower() or 'ауыстыр' in line.lower():
            continue
        if 'орнына' in line.lower() and 'қате' in line.lower():
            continue
        if 'түзетілген' in line.lower() or 'нұсқа' in line.lower():
            continue
        # Skip if starts with "Бұрыс:" or "Дұрыс:" (model confused)
        if line.lower().startswith('бұрыс') or line.lower().startswith('дұрыс'):
            continue

        # Must be mostly Cyrillic
        cyrillic = sum(1 for c in line if '\u0400' <= c <= '\u04FF')
        if cyrillic < len(line) * 0.4:
            continue

        # Must be similar length to seed (within 60%)
        if len(line) < len(seed) * 0.4 or len(line) > len(seed) * 1.6:
            continue

        # Must share some words with seed (at least 30%)
        seed_words = set(seed.lower().replace('.', '').replace(',', '').split())
        line_words = set(line.lower().replace('.', '').replace(',', '').split())
        overlap = len(seed_words & line_words)
        if seed_words and overlap < len(seed_words) * 0.3:
            continue

        return line

    return None


def is_good_sentence(sent):
    """Validate seed sentence."""
    sent = sent.strip()
    words = sent.split()
    if not (8 <= len(words) <= 40):
        return False
    if not sent[0].isupper():
        return False
    junk = ["http", "www", "()", "[]", "|", "{", "}", "ISBN", "ISSN",
            "===", "**", "{{", "}}", "<ref", "<!--", ".jpg", ".png"]
    if any(j in sent for j in junk):
        return False
    if not sent.rstrip()[-1] in ".!?":
        return False
    cyrillic = sum(1 for c in sent if '\u0400' <= c <= '\u04FF')
    if cyrillic < len(sent) * 0.5:
        return False
    return True


def load_seeds(seeds_file, max_seeds=2000):
    with open(seeds_file) as f:
        seeds = [l.strip() for l in f if l.strip() and is_good_sentence(l.strip())]
    print(f"Loaded {len(seeds)} valid seeds from {seeds_file}")
    return seeds[:max_seeds]


def load_progress(progress_file):
    done = set()
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            for line in f:
                if line.strip():
                    done.add(line.strip())
    return done


def save_progress_item(progress_file, key):
    with open(progress_file, "a") as f:
        f.write(key + "\n")


def process_one(url, et_key, seed, seed_idx):
    """Process single seed. Returns (pair_dict, None) or (None, error)."""
    et = ERROR_TYPES[et_key]
    prompt = build_prompt(et_key, seed)

    try:
        content = call_api(url, prompt)
        errored = parse_error_sentence(content, seed)

        if errored:
            return {
                "input": errored,
                "target": seed,
                "error_type": et_key,
                "tag": et["tag"],
                "tag_universal": UNIVERSAL_TAG,
            }, None
        return None, "no_parse"

    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:15127")
    parser.add_argument("--output", default="/root/gec_v2.jsonl")
    parser.add_argument("--seeds_file", default="/root/gec_seeds.txt")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_seeds", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--error_types", nargs="*", default=None)
    args = parser.parse_args()

    seeds = load_seeds(args.seeds_file, args.max_seeds)
    if not seeds:
        print("ERROR: No valid seeds!")
        return

    error_types = args.error_types or list(ERROR_TYPES.keys())
    progress_file = args.output + ".progress"
    done = load_progress(progress_file) if args.resume else set()

    total = 0
    failed = 0
    t0 = time.time()

    mode = "a" if args.resume else "w"
    out = open(args.output, mode)

    for et_key in error_types:
        et = ERROR_TYPES[et_key]
        print(f"\n{'='*60}")
        print(f"[{et['tag']}] {et_key} ({et['name']})")
        print(f"{'='*60}")

        et_count = 0
        tasks = []

        for i, seed in enumerate(seeds):
            key = f"{et_key}:{i}"
            if args.resume and key in done:
                continue
            tasks.append((i, seed, key))

        if not tasks:
            print(f"  All done (resumed)")
            continue

        print(f"  {len(tasks)} seeds, {args.workers} workers")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for seed_idx, seed, key in tasks:
                f = pool.submit(process_one, args.url, et_key, seed, seed_idx)
                futures[f] = (seed_idx, key)

            for f in as_completed(futures):
                seed_idx, key = futures[f]
                pair, err = f.result()

                if pair:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    out.flush()
                    et_count += 1
                    total += 1
                else:
                    failed += 1

                save_progress_item(progress_file, key)

                if (et_count + failed) % 20 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed * 60 if elapsed > 0 else 0
                    print(f"  [{et_key}] {et_count} good, {failed} fail | total: {total} | {rate:.1f}/min")

        print(f"  {et_key}: {et_count} pairs")

    out.close()

    elapsed = time.time() - t0
    rate = total / elapsed * 60 if elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"DONE: {total} pairs, {failed} failed, {elapsed:.0f}s ({rate:.1f}/min)")
    print(f"Saved to {args.output}")
    print(f"{'='*60}")

    # Stats
    if os.path.exists(args.output):
        lines = open(args.output).readlines()
        type_counts = {}
        for line in lines:
            d = json.loads(line)
            et = d.get("error_type", "?")
            type_counts[et] = type_counts.get(et, 0) + 1
        print("\n--- Distribution ---")
        for et, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            tag = ERROR_TYPES.get(et, {}).get("tag", "?")
            print(f"  {tag} {et}: {c}")


if __name__ == "__main__":
    main()
