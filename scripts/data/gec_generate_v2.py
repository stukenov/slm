#!/usr/bin/env python3
"""Generate GEC dataset using GPT-OSS-120B via local vLLM server.

Workflow:
1. Load seed sentences from kk Wikipedia
2. For each error type × batch of seeds → prompt
3. Parse structured output (ҚАТЕ:/ДҰРЫС:) via regex
4. Save to JSONL

Usage:
    python3 gec_generate_v2.py --vllm_url http://localhost:8000 --output /root/gec_v2.jsonl
    python3 gec_generate_v2.py --vllm_url http://localhost:8000 --output /root/gec_v2.jsonl --seeds_file seeds.txt
"""

import argparse
import json
import os
import re
import time
import urllib.request

# ── Error types and prompts ──────────────────────────────────────────────────

ERROR_TYPES = {
    "1_septik": {
        "name": "септік жалғау",
        "tag": "<септік>",
        "instruction": "Зат есімнің септік жалғауын дұрыс емес септікке ауыстыр (мысалы: барыс орнына табыс, шығыс орнына жатыс, т.б.).",
    },
    "2_zhiktik": {
        "name": "жіктік жалғау",
        "tag": "<жіктік>",
        "instruction": "Етістіктің жақ жалғауын дұрыс емес жаққа ауыстыр (мысалы: 3-жақ орнына 1-жақ, біз орнына ол, т.б.).",
    },
    "3_taueldi": {
        "name": "тәуелдік жалғау",
        "tag": "<тәуелдік>",
        "instruction": "Тәуелдік жалғауын алып таста, артық қос немесе дұрыс емес жаққа ауыстыр (мысалы: кітабым→кітап, ағалары→ағалар).",
    },
    "4_undestim": {
        "name": "дауысты үндесім",
        "tag": "<үндесім>",
        "instruction": "Жалғаудағы дауысты үндесімді (сингармонизмді) бұз (мысалы: -тен→-тан, -да→-де, -лар→-лер керісінше).",
    },
    "5_koptik": {
        "name": "көптік жалғау",
        "tag": "<көптік>",
        "instruction": "Көптік жалғауды (-лар/-лер/-дар/-дер/-тар/-тер) дұрыс емес нұсқаға ауыстыр немесе алып таста.",
    },
    "6_shaq": {
        "name": "шақ жалғау",
        "tag": "<шақ>",
        "instruction": "Етістік шағын дұрыс емеске ауыстыр (мысалы: өткен шақ орнына осы шақ, келер шақ орнына өткен шақ).",
    },
    "7_bolymsy": {
        "name": "болымсыз етістік",
        "tag": "<болымсыз>",
        "instruction": "Болымсыздық жұрнағын (-ма/-ме/-ба/-бе/-па/-пе) дұрыс емес қос, алып таста немесе дұрыс емес нұсқаға ауыстыр.",
    },
    "8_shylau": {
        "name": "шылау",
        "tag": "<шылау>",
        "instruction": "Шылау сөзді дұрыс емеске ауыстыр немесе алып таста (мысалы: дейін→кейін, туралы→арқылы).",
    },
    "9_soz_tartibi": {
        "name": "сөз тәртібі",
        "tag": "<тәртіп>",
        "instruction": "SOV сөз тәртібін бұз — бастауыш, толықтауыш немесе баяндауыш орнын ауыстыр.",
    },
    "10_emle": {
        "name": "емле қатесі",
        "tag": "<емле>",
        "instruction": "1-2 әріпті ауыстыр, алып таста немесе артық қос (табиғи опечатка жаса).",
    },
}

# Universal tag for "fix everything"
UNIVERSAL_TAG = "<грамматика>"


EXAMPLES = {
    "1_septik": "ҚАТЕ: Мен мектепте бардым.\nДҰРЫС: Мен мектепке бардым.\n---",
    "2_zhiktik": "ҚАТЕ: Ол мектепке бардым.\nДҰРЫС: Ол мектепке барды.\n---",
    "3_taueldi": "ҚАТЕ: Менің кітап үстелде жатыр.\nДҰРЫС: Менің кітабым үстелде жатыр.\n---",
    "4_undestim": "ҚАТЕ: Мектептан шықтым.\nДҰРЫС: Мектептен шықтым.\n---",
    "5_koptik": "ҚАТЕ: Балалер бақшаға барды.\nДҰРЫС: Балалар бақшаға барды.\n---",
    "6_shaq": "ҚАТЕ: Мен ертең кітап оқыдым.\nДҰРЫС: Мен ертең кітап оқимын.\n---",
    "7_bolymsy": "ҚАТЕ: Ол мектепке бармеді.\nДҰРЫС: Ол мектепке бармады.\n---",
    "8_shylau": "ҚАТЕ: Ол мектепке кейін келді.\nДҰРЫС: Ол мектепке дейін келді.\n---",
    "9_soz_tartibi": "ҚАТЕ: Мен оқыдым кітапты.\nДҰРЫС: Мен кітапты оқыдым.\n---",
    "10_emle": "ҚАТЕ: Қазақсатн тәуелсіздігін алды.\nДҰРЫС: Қазақстан тәуелсіздігін алды.\n---",
}


def build_prompt(error_type_key, seeds):
    """Build prompt for a batch of seed sentences."""
    et = ERROR_TYPES[error_type_key]
    seed_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(seeds))
    example = EXAMPLES.get(error_type_key, EXAMPLES["1_septik"])

    # Single seed per prompt — GPT-OSS-120B needs most tokens for reasoning
    seed = seeds[0] if seeds else ""
    return f"""Осы сөйлемде "{et['name']}" қатесін қос. {et['instruction']} Тек бір сөзді өзгерт.

Мысал:
{example}

Дұрыс: {seed}

ҚАТЕ:"""


def parse_response(text):
    """Parse ҚАТЕ:/ДҰРЫС: pairs from response."""
    pairs = []
    # Split by --- separator
    blocks = re.split(r'-{3,}', text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Find ҚАТЕ and ДҰРЫС lines
        m = re.search(r'ҚАТЕ:\s*(.+)', block)
        d = re.search(r'ДҰРЫС:\s*(.+)', block)
        if m and d:
            error_text = m.group(1).strip()
            correct_text = d.group(1).strip()
            # Clean up markdown bold
            error_text = re.sub(r'\*\*(.+?)\*\*', r'\1', error_text)
            correct_text = re.sub(r'\*\*(.+?)\*\*', r'\1', correct_text)
            if error_text and correct_text and error_text != correct_text:
                pairs.append({"input": error_text, "target": correct_text})
    return pairs


def call_vllm(url, prompt, max_tokens=2048, temperature=0.7):
    """Call llama-server or vLLM OpenAI-compatible API.

    GPT-OSS-120B uses reasoning: real content may be in reasoning_content field.
    We check both content and reasoning_content and return whichever has actual text.
    """
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
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    msg = result["choices"][0]["message"]

    # GPT-OSS has reasoning_content (internal thinking) and content (final answer).
    # ONLY use content — reasoning is internal monologue, not structured output.
    # If content is empty, the model didn't produce a final answer — return empty.
    content = msg.get("content", "") or ""
    return content


def is_good_sentence(sent):
    """Validate a seed sentence is clean and usable for GEC."""
    sent = sent.strip()
    words = sent.split()

    # Length: 8-40 words
    if not (8 <= len(words) <= 40):
        return False

    # Must start with uppercase Kazakh/Cyrillic letter
    if not sent[0].isupper():
        return False

    # No junk characters
    junk = ["http", "www", "()", "[]", "|", "{", "}", "ISBN", "ISSN",
            "===", "**", "{{", "}}", "<ref", "<!--", "→", "←", "↑",
            ".jpg", ".png", ".svg", "File:", "Category:"]
    if any(j in sent for j in junk):
        return False

    # Must not start with punctuation
    if sent[0] in ",.)]:;":
        return False

    # Must end with proper punctuation
    if not sent.rstrip()[-1] in ".!?":
        return False

    # Must be mostly Cyrillic (Kazakh text, not Latin/numbers)
    cyrillic = sum(1 for c in sent if '\u0400' <= c <= '\u04FF' or '\u0500' <= c <= '\u052F')
    if cyrillic < len(sent) * 0.5:
        return False

    # No excessive numbers (coordinates, dates, etc.)
    digits = sum(1 for c in sent if c.isdigit())
    if digits > len(sent) * 0.2:
        return False

    # Must have at least one verb-like word (ends with common Kazakh verb suffixes)
    # This filters out stub fragments like "Commune INSEE code — 12345"
    verb_hints = ["ды", "ді", "ты", "ті", "ды.", "ді.", "ады", "еді",
                  "ған", "ген", "йді", "йды", "лады", "леді", "алады",
                  "еледі", "атын", "етін", "болады", "табылады", "саналады",
                  "тұрады", "жатады", "орналасқан", "құрайды"]
    has_verb = any(any(w.endswith(v) for v in verb_hints) for w in words)
    if not has_verb:
        return False

    return True


def load_seeds(seeds_file=None, max_seeds=2000):
    """Load seed sentences from file or fetch from Wikipedia."""
    if seeds_file and os.path.exists(seeds_file):
        with open(seeds_file) as f:
            seeds = [line.strip() for line in f if is_good_sentence(line.strip())]
        print(f"Loaded {len(seeds)} valid seeds from {seeds_file}")
        return seeds[:max_seeds]

    print("Fetching seeds from kk.wikipedia.org...")
    headers = {"User-Agent": "SozKZ-GEC/1.0"}
    raw = []
    seeds = []

    for batch in range(100):  # Up to 100 batches
        try:
            url = "https://kk.wikipedia.org/w/api.php?action=query&list=random&rnlimit=20&rnnamespace=0&format=json"
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=15)
            pages = json.loads(resp.read())["query"]["random"]

            for p in pages:
                pid = p["id"]
                ext_url = f"https://kk.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&explaintext=true&pageids={pid}&format=json"
                try:
                    req2 = urllib.request.Request(ext_url, headers=headers)
                    r = urllib.request.urlopen(req2, timeout=10)
                    d = json.loads(r.read())
                    text = list(d["query"]["pages"].values())[0].get("extract", "")
                    for sent in text.replace("\n", " ").split("."):
                        sent = sent.strip()
                        if sent:
                            raw.append(sent + ".")
                            if is_good_sentence(sent + "."):
                                seeds.append(sent + ".")
                except Exception:
                    pass
            time.sleep(0.3)
        except Exception:
            pass

        if len(seeds) >= max_seeds:
            break

        if (batch + 1) % 20 == 0:
            print(f"  [{batch+1}] {len(seeds)} valid / {len(raw)} raw")

    # Deduplicate
    seeds = list(dict.fromkeys(seeds))
    print(f"Collected {len(seeds)} valid seeds ({len(raw)} raw, "
          f"{len(seeds)*100//max(len(raw),1)}% pass rate)")

    # Print random samples for quality check
    import random
    random.seed(42)
    sample = random.sample(seeds, min(10, len(seeds)))
    print("\n--- Random seed samples ---")
    for i, s in enumerate(sample, 1):
        print(f"  {i}. {s}")
    print("---\n")

    return seeds[:max_seeds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_url", default="http://localhost:8000")
    parser.add_argument("--output", default="/root/gec_v2.jsonl")
    parser.add_argument("--seeds_file", default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_seeds", type=int, default=2000)
    parser.add_argument("--error_types", nargs="*", default=None, help="Specific types, or all")
    args = parser.parse_args()

    seeds = load_seeds(args.seeds_file, max_seeds=args.max_seeds)
    if not seeds:
        print("ERROR: No valid seeds found!")
        return
    print(f"Using {len(seeds)} validated seeds")

    error_types = args.error_types or list(ERROR_TYPES.keys())
    total_generated = 0
    total_failed = 0

    with open(args.output, "w") as out:
        for et_key in error_types:
            et = ERROR_TYPES[et_key]
            print(f"\n{'='*60}")
            print(f"Error type: {et_key} ({et['name']})")
            print(f"{'='*60}")

            et_count = 0
            for seed_idx, seed in enumerate(seeds):
                prompt = build_prompt(et_key, [seed])

                try:
                    response = call_vllm(args.vllm_url, prompt, max_tokens=5000)
                    pairs = parse_response(response)

                    for pair in pairs:
                        # Verify pair uses actual seed, not example
                        if pair["target"].strip() == seed.strip() or pair["input"].strip() != pair["target"].strip():
                            pair["error_type"] = et_key
                            pair["tag"] = et["tag"]
                            pair["tag_universal"] = UNIVERSAL_TAG
                            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            out.flush()
                            et_count += 1
                            total_generated += 1

                    if not pairs:
                        total_failed += 1

                except Exception as e:
                    print(f"  ERROR seed {seed_idx}: {e}")
                    total_failed += 1
                    time.sleep(2)
                    continue

                if (seed_idx + 1) % 50 == 0:
                    print(f"  [{seed_idx + 1}/{len(seeds)}] {et_count} pairs")

            print(f"  Total for {et_key}: {et_count} pairs")

    print(f"\n{'='*60}")
    print(f"DONE: {total_generated} pairs, {total_failed} failed batches")
    print(f"Saved to {args.output}")
    print(f"{'='*60}")

    # Quality check: print random samples and stats
    if os.path.exists(args.output):
        lines = open(args.output).readlines()
        import random
        random.seed(42)
        sample = random.sample(lines, min(20, len(lines)))
        type_counts = {}
        for line in lines:
            d = json.loads(line)
            et = d.get("error_type", "?")
            type_counts[et] = type_counts.get(et, 0) + 1

        print("\n--- Error type distribution ---")
        for et, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            tag = ERROR_TYPES.get(et, {}).get("tag", "?")
            print(f"  {tag} {et}: {c}")

        print("\n--- Random quality check (20 samples) ---")
        for s in sample:
            d = json.loads(s)
            print(f"  [{d['tag']}]")
            print(f"    IN:  {d['input']}")
            print(f"    TGT: {d['target']}")
            # Flag suspicious: same input/target, too different, too short
            if d["input"] == d["target"]:
                print(f"    *** WARNING: input == target!")
            if len(d["input"].split()) < 5:
                print(f"    *** WARNING: very short!")
            print()


if __name__ == "__main__":
    main()
