#!/usr/bin/env python3
"""Generate a synthetic Kazakh GEC dataset with OpenAI GPT-4o.

Simplified pipeline:
  1. Collect clean Kazakh seeds from Wikipedia
  2. Verify seeds are grammatically correct (GPT-4o)
  3. Generate erroneous versions (up to 2 attempts per seed)
  4. Deterministic filters
  5. Include identity examples

Usage:
    python3 scripts/data/gec_generate_openai.py \
      --output_dir ./gec_openai_5k \
      --target_pairs 5000

    python3 scripts/data/gec_generate_openai.py \
      --output_dir /tmp/gec_openai_dryrun \
      --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from tempfile import TemporaryDirectory


API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o"

UNIVERSAL_INSTRUCTION = (
    "Мәтіндегі грамматикалық, орфографиялық, пунктуациялық және сөз қолданысындағы "
    "қателерді түзет. Мағынаны өзгертпе. Егер мәтін дұрыс болса, оны өзгеріссіз қайтар. "
    "Тек түзетілген мәтінді қайтар."
)

ERROR_PROFILES = [
    ("morphology", 1),
    ("spelling", 1),
    ("punctuation", 1),
    ("word_order", 1),
    ("function_words", 2),
    ("mixed_light", 2),
    ("mixed_hard", 3),
]

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
MULTISPACE_RE = re.compile(r"\s+")
NEGATIVE_PATTERNS = (
    "мады", "меді", "май", "мей", "майды", "мейді",
    "мас", "мес", "ма", "ме", "ба", "бе", "па", "пе",
)


def load_api_key() -> str:
    key_path = os.path.expanduser("~/.config/openai/api_key")
    if os.path.exists(key_path):
        return open(key_path).read().strip()
    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env.strip()
    raise RuntimeError(f"No API key at {key_path} and OPENAI_API_KEY env not set")


API_KEY = load_api_key()


class Stats:
    def __init__(self) -> None:
        self.in_tokens = 0
        self.out_tokens = 0
        self.requests = 0
        self.errors = 0
        self._lock = Lock()

    def add(self, in_tok: int, out_tok: int) -> None:
        with self._lock:
            self.in_tokens += in_tok
            self.out_tokens += out_tok
            self.requests += 1

    def err(self) -> None:
        with self._lock:
            self.errors += 1

    def cost_usd(self) -> float:
        return self.in_tokens * 2.50 / 1e6 + self.out_tokens * 10.0 / 1e6

    def summary(self) -> str:
        return (
            f"req={self.requests} err={self.errors} "
            f"in={self.in_tokens:,} out={self.out_tokens:,} "
            f"cost=${self.cost_usd():.4f}"
        )


STATS = Stats()
_rate_lock = Lock()
_last_ts = 0.0


def _rate_limit(interval: float) -> None:
    global _last_ts
    if interval <= 0:
        return
    with _rate_lock:
        now = time.time()
        wait = _last_ts + interval - now
        if wait > 0:
            time.sleep(wait)
        _last_ts = time.time()


def call_api(
    messages: list[dict],
    max_tokens: int = 600,
    temperature: float = 0.7,
    retries: int = 4,
    interval: float = 0.0,
    json_mode: bool = True,
) -> str:
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()

    last_err = None
    for attempt in range(retries):
        try:
            _rate_limit(interval)
            req = urllib.request.Request(
                API_URL, data=data, method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            content = result["choices"][0]["message"].get("content", "")
            usage = result.get("usage", {})
            STATS.add(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            return content
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 429:
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    STATS.err()
    raise last_err


# ── Text utils ──────────────────────────────────────────────────────


def norm(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def cyr_ratio(text: str) -> float:
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return 0.0
    return len(CYRILLIC_RE.findall(text)) / alpha


def edit_dist(a: str, b: str) -> int:
    if len(a) < len(b):
        return edit_dist(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def word_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


def has_negation_shift(a: str, b: str) -> bool:
    wa = re.findall(r"\w+", a.lower())
    wb = re.findall(r"\w+", b.lower())
    if len(wa) != len(wb):
        return False
    for left, right in zip(wa, wb):
        if left == right:
            continue
        left_neg = any(p in left for p in NEGATIVE_PATTERNS)
        right_neg = any(p in right for p in NEGATIVE_PATTERNS)
        if left_neg != right_neg:
            if sum(1 for a, b in zip(left, right) if a == b) >= 2:
                return True
    return False


def is_good_seed(s: str) -> bool:
    s = norm(s)
    words = s.split()
    if not (6 <= len(words) <= 30):
        return False
    if not s or not s[0].isupper() or s[-1] not in ".!?":
        return False
    if cyr_ratio(s) < 0.75:
        return False
    junk = ["http", "www", "<ref", "{{", "}}", "ISBN", "File:", ".jpg", ".png"]
    if any(j in s for j in junk):
        return False
    if sum(1 for c in s if c.isdigit()) > max(3, len(s) * 0.1):
        return False
    return True


def validate_pair(incorrect: str, correct: str) -> tuple[bool, str]:
    incorrect, correct = norm(incorrect), norm(correct)
    if not incorrect or not correct:
        return False, "empty"
    if incorrect == correct:
        return False, "identity"
    if cyr_ratio(incorrect) < 0.7 or cyr_ratio(correct) < 0.7:
        return False, "low_cyr"
    if abs(len(incorrect.split()) - len(correct.split())) > 3:
        return False, "len_shift"
    if word_overlap(correct, incorrect) < 0.45:
        return False, "low_overlap"
    if has_negation_shift(incorrect, correct):
        return False, "negation_shift"
    dist = edit_dist(incorrect, correct)
    if dist < 1:
        return False, "no_edit"
    if dist > max(16, int(len(correct) * 0.35)):
        return False, "too_far"
    return True, "ok"


# ── Seeds ───────────────────────────────────────────────────────────


def load_seeds(seeds_file: str | None, max_seeds: int) -> list[str]:
    if seeds_file and os.path.exists(seeds_file):
        with open(seeds_file) as f:
            seeds = [norm(line) for line in f if is_good_seed(line)]
        print(f"Loaded {len(seeds)} seeds from {seeds_file}", flush=True)
        return list(dict.fromkeys(seeds))[:max_seeds]

    print("Fetching seeds from kk.wikipedia.org ...", flush=True)
    headers = {"User-Agent": "SozKZ-GEC/2.0"}
    seeds: list[str] = []
    for batch in range(400):
        try:
            url = (
                "https://kk.wikipedia.org/w/api.php?action=query&list=random"
                "&rnlimit=20&rnnamespace=0&format=json"
            )
            req = urllib.request.Request(url, headers=headers)
            pages = json.loads(urllib.request.urlopen(req, timeout=15).read())["query"]["random"]
            for page in pages:
                ext = (
                    "https://kk.wikipedia.org/w/api.php?action=query&prop=extracts"
                    f"&explaintext=true&pageids={page['id']}&format=json"
                )
                try:
                    r2 = urllib.request.Request(ext, headers=headers)
                    text = list(
                        json.loads(urllib.request.urlopen(r2, timeout=10).read())
                        ["query"]["pages"].values()
                    )[0].get("extract", "").replace("\n", " ")
                    for part in re.split(r"(?<=[.!?])\s+", text):
                        part = norm(part)
                        if is_good_seed(part):
                            seeds.append(part)
                except Exception:
                    pass
            time.sleep(0.15)
        except Exception:
            pass
        if len(seeds) >= max_seeds:
            break
        if (batch + 1) % 30 == 0:
            print(f"  batch={batch+1} seeds={len(seeds)}", flush=True)
    seeds = list(dict.fromkeys(seeds))
    print(f"Collected {len(seeds)} seeds", flush=True)
    return seeds[:max_seeds]


# ── Seed verification ───────────────────────────────────────────────


def verify_seed_clean(seed: str, interval: float = 0.0) -> tuple[bool, str]:
    """Ask GPT-4o whether the seed sentence is grammatically correct Kazakh."""
    messages = [
        {
            "role": "system",
            "content": (
                "Сен қазақ тілінің грамматика сарапшысысың. "
                "Берілген сөйлемде грамматикалық, орфографиялық немесе пунктуациялық қате бар-жоғын тексер."
            ),
        },
        {
            "role": "user",
            "content": f"""Сөйлемді тексер. Жауапты JSON форматында бер.

Егер сөйлем толығымен дұрыс болса: {{"clean": true, "reason": "дұрыс"}}
Егер қате болса: {{"clean": false, "reason": "қысқа себеп"}}

Сөйлем: {seed}""",
        },
    ]
    try:
        raw = call_api(messages, max_tokens=120, temperature=0.0, interval=interval)
        data = json.loads(raw)
        return bool(data.get("clean", False)), data.get("reason", "")
    except Exception:
        return True, "verify_error"


# ── Generation ──────────────────────────────────────────────────────


def generate_error(
    clean: str, profile: str, n_errors: int, interval: float = 0.0, retry: bool = False,
) -> dict | None:
    retry_note = (
        "\nҚОСЫМША: алдыңғы жауап сәтсіз болды. incorrect пен correct арасында нақты айырмашылық болсын.\n"
        if retry else ""
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Сен қазақ тіліндегі синтетикалық грамматикалық қателер жасайтын модельсің. "
                "Міндетің: дұрыс сөйлемнен мағынасы сақталған, бірақ табиғи қателері бар сөйлем жасау."
            ),
        },
        {
            "role": "user",
            "content": f"""Берілген ДҰРЫС сөйлемнен ҚАТЕ нұсқасын жаса.

ТАЛАПТАР:
- Мағынаны сақта
- Табиғи адамдық қателер енгіз
- Жаңа ақпарат қоспа
- Қате саны: {n_errors}
- Профиль: {profile}
{retry_note}
JSON:
{{
  "incorrect": "қатесі бар сөйлем",
  "correct": "{clean}",
  "error_types": ["қате типі"]
}}

ДҰРЫС СӨЙЛЕМ: {clean}""",
        },
    ]
    try:
        raw = call_api(messages, max_tokens=400, temperature=0.7, interval=interval)
        data = json.loads(raw)
        if isinstance(data, dict) and "incorrect" in data and "correct" in data:
            return data
    except Exception:
        pass
    return None


# ── Main loop ───────────────────────────────────────────────────────


def make_row(incorrect: str, correct: str, meta: dict) -> dict:
    return {
        "instruction": UNIVERSAL_INSTRUCTION,
        "input": norm(incorrect),
        "output": norm(correct),
        "source": "synthetic_gec_gpt4o",
        "meta": meta,
    }


def make_identity(text: str) -> dict:
    return {
        "instruction": UNIVERSAL_INSTRUCTION,
        "input": norm(text),
        "output": norm(text),
        "source": "identity_clean_verified",
        "meta": {"profile": "identity"},
    }


def append_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def upload_stage(
    hf_repo: str,
    synth_path: str,
    identity_rows: list[dict],
    stage_num: int,
    stage_size: int,
) -> None:
    """Upload a cumulative snapshot to HuggingFace."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  [HF] huggingface_hub not installed, skipping upload", flush=True)
        return

    api = HfApi()
    try:
        api.create_repo(hf_repo, repo_type="dataset", exist_ok=True, private=False)
    except Exception as exc:
        print(f"  [HF] create_repo failed: {exc}", flush=True)
        return

    synth_rows = read_jsonl(synth_path)
    all_rows = synth_rows + identity_rows
    label = f"stage_{stage_num:02d}_{len(synth_rows)}syn_{len(identity_rows)}id"

    with TemporaryDirectory() as tmp:
        train_path = os.path.join(tmp, "train.jsonl")
        with open(train_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        api.upload_file(
            path_or_fileobj=train_path,
            path_in_repo=f"data/{label}.jsonl",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Stage {stage_num}: {len(synth_rows)} synthetic + {len(identity_rows)} identity = {len(all_rows)} total",
        )

        readme = f"""---
language: kk
license: mit
task_categories:
  - text2text-generation
tags:
  - gec
  - grammar-correction
  - kazakh
  - synthetic
size_categories:
  - 1K<n<10K
---

# Kazakh GEC Synthetic Dataset (GPT-4o)

Synthetic Kazakh grammar error correction dataset generated with GPT-4o.

## Stages

Each stage file is a **cumulative snapshot** — pick any single file to train on.

| File | Synthetic | Identity | Total |
|------|-----------|----------|-------|
| `{label}.jsonl` | {len(synth_rows)} | {len(identity_rows)} | {len(all_rows)} |

## Format

```json
{{
  "instruction": "Мәтіндегі грамматикалық, орфографиялық, пунктуациялық және сөз қолданысындағы қателерді түзет...",
  "input": "қатесі бар мәтін",
  "output": "түзетілген мәтін",
  "source": "synthetic_gec_gpt4o | identity_clean_verified",
  "meta": {{"profile": "spelling"}}
}}
```

## Generation Pipeline

1. Seeds from kk.wikipedia.org
2. Seed verification (GPT-4o checks grammatical correctness)
3. Error generation with retry (GPT-4o creates natural errors)
4. Deterministic filters (cyrillic ratio, edit distance, negation shift, etc.)
5. Identity examples (~30%) to prevent overcorrection
"""
        readme_path = os.path.join(tmp, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme)
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Update README at stage {stage_num}",
        )

    print(
        f"  [HF] Uploaded stage {stage_num}: {len(all_rows)} rows → {hf_repo}/data/{label}.jsonl",
        flush=True,
    )


def verify_batch(
    seeds: list[str], concurrency: int, interval: float, skip: bool,
) -> list[str]:
    """Verify a batch of seeds, return only clean ones."""
    if skip:
        return list(seeds)
    verified: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(verify_seed_clean, s, interval): s for s in seeds}
        for fut in as_completed(futs):
            try:
                clean, _ = fut.result()
            except Exception:
                clean = False
            if clean:
                verified.append(futs[fut])
    return verified


def text_hash(text: str) -> str:
    return hashlib.md5(norm(text).lower().encode()).hexdigest()


def load_seen_hashes(path: str) -> set[str]:
    if os.path.exists(path):
        with open(path) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_seen_hashes(path: str, hashes: set[str]) -> None:
    with open(path, "w") as f:
        for h in sorted(hashes):
            f.write(h + "\n")


def generate_batch(
    seeds: list[str],
    target: int,
    concurrency: int,
    interval: float,
    rng: random.Random,
    seen: set[str] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Generate error pairs from seeds, return rows and failure counts."""
    tasks = []
    for idx, seed in enumerate(seeds):
        profile_name, n_errors = rng.choice(ERROR_PROFILES)
        tasks.append((idx, seed, profile_name, n_errors))
    rng.shuffle(tasks)
    tasks = tasks[: target * 2]

    rows: list[dict] = []
    failures: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        fmap = {
            ex.submit(generate_error, s, p, n, interval): (i, s, p, n)
            for i, s, p, n in tasks
        }
        for fut in as_completed(fmap):
            idx, seed, profile, n_err = fmap[fut]
            try:
                result = fut.result()
            except Exception:
                result = None

            if not result:
                failures["api_fail"] = failures.get("api_fail", 0) + 1
                continue

            incorrect = result.get("incorrect", "")
            correct = result.get("correct", seed)
            ok, reason = validate_pair(incorrect, correct)
            if ok:
                h = text_hash(correct)
                if seen is not None and h in seen:
                    reason = "duplicate"
                else:
                    if seen is not None:
                        seen.add(h)
                    rows.append(make_row(incorrect, correct, {
                        "seed_idx": idx, "profile": profile,
                        "error_types": result.get("error_types", []),
                    }))
                    reason = None
            if reason:
                try:
                    r2 = generate_error(seed, profile, n_err, interval, retry=True)
                    if r2:
                        ok2, reason2 = validate_pair(r2.get("incorrect", ""), r2.get("correct", seed))
                        if ok2:
                            h2 = text_hash(r2.get("correct", seed))
                            if seen is not None and h2 in seen:
                                reason = "duplicate"
                            else:
                                if seen is not None:
                                    seen.add(h2)
                                rows.append(make_row(r2["incorrect"], r2["correct"], {
                                    "seed_idx": idx, "profile": profile,
                                    "error_types": r2.get("error_types", []), "retry": True,
                                }))
                                reason = None
                        else:
                            reason = reason2
                except Exception:
                    pass
                if reason:
                    failures[reason] = failures.get(reason, 0) + 1

            if len(rows) >= target:
                break

    return rows[:target], failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./gec_openai")
    parser.add_argument("--seeds_file", default=None)
    parser.add_argument("--target_pairs", type=int, default=5000)
    parser.add_argument("--identity_ratio", type=float, default=0.30)
    parser.add_argument("--max_seeds", type=int, default=20000)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget_usd", type=float, default=0.0)
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--skip-seed-verify", action="store_true")
    parser.add_argument("--hf_repo", default=None)
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    synth_path = os.path.join(args.output_dir, "synthetic.jsonl")
    identity_path = os.path.join(args.output_dir, "identity.jsonl")

    target = min(args.target_pairs, 50) if args.dry_run else args.target_pairs
    target_identity = int(target * args.identity_ratio)
    target_synth = max(1, target - target_identity)
    num_batches = (target_synth + args.batch_size - 1) // args.batch_size

    # ── Load all seeds (no API calls) ──
    all_seeds = load_seeds(args.seeds_file, args.max_seeds)
    if len(all_seeds) < 100:
        raise RuntimeError(f"Too few seeds: {len(all_seeds)}")

    rng = random.Random(42)
    rng.shuffle(all_seeds)
    used_seeds: set[str] = set()
    seed_cursor = 0
    seen_hashes_path = os.path.join(args.output_dir, "seen_hashes.txt")
    seen: set[str] = load_seen_hashes(seen_hashes_path)

    # Resume from existing progress
    existing_synth = 0
    if os.path.exists(synth_path):
        with open(synth_path) as f:
            existing_synth = sum(1 for line in f if line.strip())
    all_identity: list[dict] = []
    if os.path.exists(identity_path):
        with open(identity_path) as f:
            all_identity = [json.loads(line) for line in f if line.strip()]
    total_synth = existing_synth
    if existing_synth > 0:
        batches_done = (existing_synth + args.batch_size - 1) // args.batch_size
        seed_cursor = batches_done * (int(args.batch_size * 2.2) + 50)
        print(f"Resuming: {existing_synth} synthetic, {len(all_identity)} identity, "
              f"skipping ~{seed_cursor} seeds", flush=True)

    total_failures: dict[str, int] = {}

    print(
        f"Plan: {target_synth} synthetic + {target_identity} identity = {target} total, "
        f"{num_batches} batches of {args.batch_size}",
        flush=True,
    )

    for batch_num in range(1, num_batches + 1):
        batch_target = min(args.batch_size, target_synth - total_synth)
        if batch_target <= 0:
            break

        # How many seeds to grab: ~1.4x batch target (for verify reject + gen filter losses)
        seeds_needed = int(batch_target * 2.2) + 50
        batch_seeds_raw: list[str] = []
        while len(batch_seeds_raw) < seeds_needed and seed_cursor < len(all_seeds):
            s = all_seeds[seed_cursor]
            seed_cursor += 1
            if s not in used_seeds:
                batch_seeds_raw.append(s)

        if len(batch_seeds_raw) < 20:
            print(f"Batch {batch_num}: not enough unused seeds, stopping.", flush=True)
            break

        # ── Verify ──
        print(
            f"\n── Batch {batch_num}/{num_batches}: verify {len(batch_seeds_raw)} seeds, "
            f"target {batch_target} pairs ──",
            flush=True,
        )
        clean_seeds = verify_batch(
            batch_seeds_raw, args.concurrency, args.interval, args.skip_seed_verify,
        )
        for s in batch_seeds_raw:
            used_seeds.add(s)
        print(
            f"  Verified: {len(clean_seeds)}/{len(batch_seeds_raw)} clean [{STATS.summary()}]",
            flush=True,
        )

        if len(clean_seeds) < 10:
            print(f"  Too few clean seeds, skipping batch.", flush=True)
            continue

        # ── Identity (proportional to this batch) ──
        id_for_batch = int(batch_target * args.identity_ratio / (1 - args.identity_ratio))
        id_for_batch = min(id_for_batch, len(clean_seeds) // 3)
        batch_identity = [make_identity(s) for s in clean_seeds[:id_for_batch]]
        if batch_identity:
            append_jsonl(identity_path, batch_identity)
            all_identity.extend(batch_identity)

        # ── Generate ──
        gen_seeds = clean_seeds[id_for_batch:]
        rows, fails = generate_batch(
            gen_seeds, batch_target, args.concurrency, args.interval, rng, seen,
        )
        if rows:
            append_jsonl(synth_path, rows)
        total_synth += len(rows)
        for k, v in fails.items():
            total_failures[k] = total_failures.get(k, 0) + v

        print(
            f"  Batch {batch_num} done: +{len(rows)} synthetic, +{len(batch_identity)} identity "
            f"| total: {total_synth} syn, {len(all_identity)} id [{STATS.summary()}]",
            flush=True,
        )

        # ── Save dedup hashes + upload to HF ──
        save_seen_hashes(seen_hashes_path, seen)

        if args.hf_repo:
            upload_stage(args.hf_repo, synth_path, all_identity, batch_num, args.batch_size)

        if args.budget_usd and STATS.cost_usd() >= args.budget_usd:
            print("Budget reached, stopping.", flush=True)
            break

    # ── Summary ──
    summary = {
        "model": MODEL,
        "synthetic": total_synth,
        "identity": len(all_identity),
        "total": total_synth + len(all_identity),
        "failures": total_failures,
        "stats": STATS.summary(),
        "cost_usd": round(STATS.cost_usd(), 4),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone.", flush=True)
    print(f"  synthetic: {total_synth}", flush=True)
    print(f"  identity:  {len(all_identity)}", flush=True)
    print(f"  total:     {total_synth + len(all_identity)}", flush=True)
    print(f"  failures:  {total_failures}", flush=True)
    print(f"  cost:      ${STATS.cost_usd():.4f}", flush=True)
    print(f"  output:    {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
