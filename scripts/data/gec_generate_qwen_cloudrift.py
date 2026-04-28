#!/usr/bin/env python3
"""Generate a synthetic Kazakh GEC dataset with CloudRift Qwen.

Design:
  - Teacher model corrupts a clean Kazakh sentence while preserving meaning.
  - Final training rows use a single universal correction instruction.
  - Includes identity examples so the downstream model learns not to overcorrect.

Usage:
    python3 scripts/data/gec_generate_qwen_cloudrift.py \
      --output_dir ./gec_qwen_5k \
      --target_pairs 5000

    python3 scripts/data/gec_generate_qwen_cloudrift.py \
      --output_dir ./gec_qwen_dryrun \
      --limit 30 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


API_URL = "https://inference.cloudrift.ai/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
PRICE_IN_PER_1M = 0.25
PRICE_OUT_PER_1M = 1.50

UNIVERSAL_INSTRUCTION = (
    "Мәтіндегі грамматикалық, орфографиялық, пунктуациялық және сөз қолданысындағы "
    "қателерді түзет. Мағынаны өзгертпе. Егер мәтін дұрыс болса, оны өзгеріссіз қайтар. "
    "Тек түзетілген мәтінді қайтар."
)

ERROR_PROFILES = [
    ("morphology", "morphology", "easy", 1),
    ("spelling", "spelling", "easy", 1),
    ("punctuation", "punctuation", "easy", 1),
    ("word_order", "word_order", "medium", 1),
    ("function_words", "function_words", "medium", 2),
    ("mixed_light", "mixed", "medium", 2),
    ("mixed_hard", "mixed", "hard", 3),
]

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
MULTISPACE_RE = re.compile(r"\s+")
NEGATIVE_PATTERNS = (
    "мады", "меді", "мады.", "меді.",
    "май", "мей", "майды", "мейді",
    "мас", "мес", "ма", "ме", "ба", "бе", "па", "пе",
)


def load_api_key() -> str:
    key_path = os.path.expanduser("~/.config/cloudrift/api_key")
    if os.path.exists(key_path):
        return open(key_path).read().strip()
    env = os.environ.get("CLOUDRIFT_API_KEY")
    if env:
        return env.strip()
    raise RuntimeError(f"No API key at {key_path} and CLOUDRIFT_API_KEY env not set")


API_KEY = load_api_key()


class CostTracker:
    def __init__(self) -> None:
        self.in_tokens = 0
        self.out_tokens = 0
        self.requests = 0
        self.errors = 0
        self.lock = Lock()

    def add(self, in_tok: int, out_tok: int) -> None:
        with self.lock:
            self.in_tokens += in_tok
            self.out_tokens += out_tok
            self.requests += 1

    def err(self) -> None:
        with self.lock:
            self.errors += 1

    def cost_usd(self) -> float:
        return (
            self.in_tokens * PRICE_IN_PER_1M / 1e6
            + self.out_tokens * PRICE_OUT_PER_1M / 1e6
        )

    def summary(self) -> str:
        return (
            f"req={self.requests} err={self.errors} "
            f"in={self.in_tokens:,} out={self.out_tokens:,} "
            f"cost=${self.cost_usd():.4f}"
        )


TRACKER = CostTracker()
RATE_LOCK = Lock()
LAST_REQUEST_TS = 0.0


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def cyrillic_ratio(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return 0.0
    return len(CYRILLIC_RE.findall(text)) / alpha


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return edit_distance(b, a)
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
        left_neg = any(pat in left for pat in NEGATIVE_PATTERNS)
        right_neg = any(pat in right for pat in NEGATIVE_PATTERNS)
        if left_neg != right_neg:
            shared_prefix = 0
            for ca, cb in zip(left, right):
                if ca != cb:
                    break
                shared_prefix += 1
            if shared_prefix >= 2:
                return True
    return False


def is_good_seed(sent: str) -> bool:
    sent = normalize_text(sent)
    words = sent.split()
    if not (6 <= len(words) <= 30):
        return False
    if not sent or not sent[0].isupper():
        return False
    if sent[-1] not in ".!?":
        return False
    if cyrillic_ratio(sent) < 0.75:
        return False
    junk = ["http", "www", "<ref", "{{", "}}", "ISBN", "File:", ".jpg", ".png"]
    if any(j in sent for j in junk):
        return False
    digits = sum(1 for c in sent if c.isdigit())
    if digits > max(3, len(sent) * 0.1):
        return False
    return True


def load_seeds(seeds_file: str | None, max_seeds: int) -> list[str]:
    if seeds_file and os.path.exists(seeds_file):
        with open(seeds_file) as f:
            seeds = [normalize_text(line) for line in f if is_good_seed(line)]
        print(f"Loaded {len(seeds)} valid seeds from {seeds_file}", flush=True)
        return list(dict.fromkeys(seeds))[:max_seeds]

    print("Fetching seeds from kk.wikipedia.org...", flush=True)
    headers = {"User-Agent": "SozKZ-GEC-Synthetic/1.0"}
    seeds: list[str] = []

    for batch in range(300):
        try:
            url = "https://kk.wikipedia.org/w/api.php?action=query&list=random&rnlimit=20&rnnamespace=0&format=json"
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=15)
            pages = json.loads(resp.read())["query"]["random"]
            for page in pages:
                pid = page["id"]
                ext_url = (
                    "https://kk.wikipedia.org/w/api.php?action=query&prop=extracts"
                    f"&explaintext=true&pageids={pid}&format=json"
                )
                try:
                    req2 = urllib.request.Request(ext_url, headers=headers)
                    resp2 = urllib.request.urlopen(req2, timeout=10)
                    data = json.loads(resp2.read())
                    text = list(data["query"]["pages"].values())[0].get("extract", "")
                    text = text.replace("\n", " ")
                    parts = re.split(r"(?<=[.!?])\s+", text)
                    for part in parts:
                        part = normalize_text(part)
                        if is_good_seed(part):
                            seeds.append(part)
                except Exception:
                    pass
            time.sleep(0.2)
        except Exception:
            pass
        if len(seeds) >= max_seeds:
            break
        if (batch + 1) % 20 == 0:
            print(f"  batch={batch+1} seeds={len(seeds)}", flush=True)

    seeds = list(dict.fromkeys(seeds))
    print(f"Collected {len(seeds)} valid seeds", flush=True)
    return seeds[:max_seeds]


def _rate_limit(min_interval_s: float) -> None:
    global LAST_REQUEST_TS
    if min_interval_s <= 0:
        return
    with RATE_LOCK:
        now = time.time()
        wait = LAST_REQUEST_TS + min_interval_s - now
        if wait > 0:
            time.sleep(wait)
        LAST_REQUEST_TS = time.time()


def call_api(
    messages: list[dict],
    max_tokens: int = 800,
    temperature: float = 0.6,
    retries: int = 5,
    min_interval_s: float = 0.0,
) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }).encode()

    last_err = None
    for attempt in range(retries):
        try:
            _rate_limit(min_interval_s)
            req = urllib.request.Request(
                API_URL,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            usage = data.get("usage") or {}
            TRACKER.add(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            return content
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 429:
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                continue
            time.sleep(1.0 * (attempt + 1))
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    TRACKER.err()
    raise last_err


def build_teacher_messages(
    clean_text: str,
    profile: tuple[str, str, str, int],
    retry_mode: bool = False,
) -> list[dict]:
    profile_name, focus, difficulty, n_errors = profile
    system = (
        "Сен қазақ тіліндегі грамматикалық қателерді модельдеу үшін синтетикалық дерек жасайсың. "
        "Сенің міндетің: дұрыс сөйлемнен мағынасы сақталған, бірақ табиғи қателері бар сөйлем жасау."
    )
    retry_note = (
        "\nҚОСЫМША ТАЛАП:\n- Алдыңғы жауапта қате жеткіліксіз болды. "
        "Бұл жолы incorrect міндетті түрде correct-тен кемінде бір нақты редакциялық айырмашылықпен өзгеше болсын.\n"
        if retry_mode else ""
    )
    user = f"""Берілген ДҰРЫС сөйлемнен ҚАТЕ сөйлем жаса.

ТАЛАПТАР:
- Мағынаны сақта.
- Табиғи адамдық қателерді енгіз.
- Жаңа ақпарат қоспа.
- Дұрыс сөйлемді қайта көшіріп қойма.
- ҚАТЕ сөйлем түзетілген нұсқадан қатты алшақ кетпесін.
- Қате саны: {n_errors}
- Негізгі профиль: {focus}
- Қиындық: {difficulty}

ҚАТЕ ТИПТЕРІ:
- жалғау, жұрнақ, сөз формасы
- емле, опечатка
- тыныс белгілері
- сөз тәртібі
- шылау/көмекші сөз
- артық немесе түсіп қалған сөз
- қате сөз қолданысы
{retry_note}

ҚАТАҢ JSON:
{{
  "incorrect": "...",
  "correct": "...",
  "error_types": ["..."],
  "difficulty": "{difficulty}"
}}

ДҰРЫС СӨЙЛЕМ:
{clean_text}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_judge_messages(incorrect: str, correct: str) -> list[dict]:
    system = (
        "Сен қазақ тіліндегі grammatical error correction деректерін тексеретін қатаң валидаторсың. "
        "Мақсатың: тек сенімді, мағынасы сақталған, шын қателері бар жұптарды ғана қабылдау."
    )
    user = f"""Төмендегі жұпты тексер.

КРИТЕРИЙ:
- incorrect мәтінінде шынымен тілдік қате болуы керек
- correct сол қатені түзетуі керек
- мағына сақталуы керек
- жаңа факт, жаңа эмоция, жаңа болымсыздық қоспау керек
- күшті парафраз жасамау керек
- егер жұп күмәнді болса, reject қыл

ҚАТАҢ JSON:
{{
  "verdict": "accept" немесе "reject",
  "confidence": 0.0,
  "reason": "қысқа себеп",
  "error_present": true,
  "meaning_preserved": true
}}

incorrect: {incorrect}
correct: {correct}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_corrector_messages(incorrect: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Сен қазақ тіліндегі мәтін түзеткішісің. "
                "Грамматикалық, орфографиялық, пунктуациялық және сөз қолданысындағы қателерді түзет. "
                "Мағынаны өзгертпе. Егер мәтін дұрыс болса, өзгеріссіз қайтар. "
                "Тек түзетілген мәтінді қайтар."
            ),
        },
        {"role": "user", "content": incorrect},
    ]


def build_equivalence_messages(candidate: str, reference: str) -> list[dict]:
    system = (
        "Сен қазақ тіліндегі түзетілген мәтіндердің эквиваленттілігін тексеретін қатаң валидаторсың."
    )
    user = f"""Екі мәтінді салыстыр.

Мақсат:
- екеуі де бір бастапқы қате мәтіннің дұрыс түзетілген нұсқалары бола ала ма?
- мағынасы бірдей ме?
- айырмашылық тек стильдік немесе ұсақ тыныс белгісі деңгейінде ме?

ҚАТАҢ JSON:
{{
  "equivalent": true,
  "confidence": 0.0,
  "reason": "қысқа себеп"
}}

candidate: {candidate}
reference: {reference}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_teacher_response(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    return data


def parse_json_response(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return None


def extract_text_response(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    parsed = parse_json_response(raw)
    if isinstance(parsed, dict):
        for key in ("response", "corrected", "output", "text", "answer", "content"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return normalize_text(value)
    return normalize_text(raw)


def validate_pair(incorrect: str, correct: str) -> tuple[bool, str]:
    incorrect = normalize_text(incorrect)
    correct = normalize_text(correct)
    if not incorrect or not correct:
        return False, "empty"
    if incorrect == correct:
        return False, "identity_as_error"
    if cyrillic_ratio(incorrect) < 0.7 or cyrillic_ratio(correct) < 0.7:
        return False, "low_cyrillic"
    if abs(len(incorrect.split()) - len(correct.split())) > 3:
        return False, "length_shift"
    if word_overlap(correct, incorrect) < 0.45:
        return False, "low_overlap"
    if has_negation_shift(incorrect, correct):
        return False, "negation_shift"
    dist = edit_distance(incorrect, correct)
    if dist < 1:
        return False, "no_edit"
    if dist > max(16, int(len(correct) * 0.35)):
        return False, "too_far"
    return True, "ok"


def verify_with_judge(incorrect: str, correct: str, min_interval_s: float = 0.0) -> tuple[bool, dict | None, str]:
    raw = call_api(
        build_judge_messages(incorrect, correct),
        max_tokens=300,
        temperature=0.0,
        min_interval_s=min_interval_s,
    )
    parsed = parse_json_response(raw)
    if not parsed:
        return False, None, "judge_bad_json"
    verdict = str(parsed.get("verdict", "")).lower()
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    error_present = bool(parsed.get("error_present", False))
    meaning_preserved = bool(parsed.get("meaning_preserved", False))
    if verdict != "accept":
        return False, parsed, "judge_reject"
    if confidence < 0.75:
        return False, parsed, "judge_low_conf"
    if not error_present or not meaning_preserved:
        return False, parsed, "judge_flags"
    return True, parsed, "ok"


def verify_with_roundtrip(incorrect: str, correct: str, min_interval_s: float = 0.0) -> tuple[bool, str, str]:
    raw = call_api(
        build_corrector_messages(incorrect),
        max_tokens=220,
        temperature=0.0,
        retries=2,
        min_interval_s=min_interval_s,
    )
    repaired = extract_text_response(raw)
    if not repaired:
        return False, repaired, "roundtrip_empty"
    if repaired == normalize_text(correct):
        return True, repaired, "ok"
    if repaired == normalize_text(incorrect):
        return False, repaired, "roundtrip_no_fix"
    if edit_distance(repaired, normalize_text(correct)) <= 3 and word_overlap(repaired, correct) >= 0.8:
        return True, repaired, "ok_near"
    if cyrillic_ratio(repaired) < 0.7:
        return False, repaired, "roundtrip_low_cyr"
    if word_overlap(repaired, correct) < 0.6:
        return False, repaired, "roundtrip_low_overlap"
    eq_raw = call_api(
        build_equivalence_messages(repaired, normalize_text(correct)),
        max_tokens=180,
        temperature=0.0,
        retries=2,
        min_interval_s=min_interval_s,
    )
    eq = parse_json_response(eq_raw)
    if not eq:
        return False, repaired, "roundtrip_equiv_bad_json"
    if bool(eq.get("equivalent", False)) and float(eq.get("confidence", 0.0) or 0.0) >= 0.7:
        return True, repaired, "ok_equiv"
    return False, repaired, "roundtrip_mismatch"


def make_training_row(incorrect: str, correct: str, meta: dict) -> dict:
    return {
        "instruction": UNIVERSAL_INSTRUCTION,
        "input": normalize_text(incorrect),
        "output": normalize_text(correct),
        "source": "synthetic_gec_qwen35_cloudrift",
        "meta": meta,
    }


def make_identity_row(text: str) -> dict:
    return {
        "instruction": UNIVERSAL_INSTRUCTION,
        "input": normalize_text(text),
        "output": normalize_text(text),
        "source": "identity_clean_seed",
        "meta": {
            "profile": "identity",
            "difficulty": "clean",
            "error_types": [],
        },
    }


def load_progress(output_dir: str) -> dict:
    path = os.path.join(output_dir, "progress.json")
    if os.path.exists(path):
        return json.load(open(path))
    return {
        "done_seed_keys": [],
        "pairs_written": 0,
        "identity_written": 0,
        "spent_usd": 0.0,
    }


def save_progress(output_dir: str, progress: dict) -> None:
    path = os.path.join(output_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_seed(
    seed: str,
    seed_idx: int,
    profile: tuple[str, str, str, int],
    strict: bool = False,
    min_interval_s: float = 0.0,
) -> tuple[dict | None, str | None]:
    last_reason = "unknown"
    for retry_mode in (False, True):
        raw = call_api(
            build_teacher_messages(seed, profile, retry_mode=retry_mode),
            min_interval_s=min_interval_s,
        )
        parsed = parse_teacher_response(raw)
        if not parsed:
            last_reason = "bad_json"
            continue
        incorrect = parsed.get("incorrect", "")
        correct = parsed.get("correct", "")
        ok, reason = validate_pair(incorrect, correct)
        if not ok:
            last_reason = reason
            continue
        judge_meta = None
        roundtrip_meta = None
        if strict:
            judge_ok, judge_meta, judge_reason = verify_with_judge(
                incorrect, correct, min_interval_s=min_interval_s
            )
            if not judge_ok:
                last_reason = judge_reason
                continue
            rt_ok, repaired, rt_reason = verify_with_roundtrip(
                incorrect, correct, min_interval_s=min_interval_s
            )
            roundtrip_meta = {"repaired": repaired, "status": rt_reason}
            if not rt_ok:
                last_reason = rt_reason
                continue
        row = make_training_row(
            incorrect,
            correct,
            {
                "seed_idx": seed_idx,
                "profile": profile[0],
                "difficulty": parsed.get("difficulty", profile[2]),
                "error_types": parsed.get("error_types", []),
                "correct_seed": normalize_text(seed),
                "retry_mode": retry_mode,
                "judge": judge_meta,
                "roundtrip": roundtrip_meta,
            },
        )
        return row, None
    return None, last_reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./gec_qwen_cloudrift")
    parser.add_argument("--seeds_file", default=None)
    parser.add_argument("--target_pairs", type=int, default=5000)
    parser.add_argument("--identity_ratio", type=float, default=0.30)
    parser.add_argument("--limit", type=int, default=0, help="Limit total seeds for quick tests")
    parser.add_argument("--max_seeds", type=int, default=12000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget_usd", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true", help="Use judge + roundtrip verification")
    parser.add_argument("--min-interval", type=float, default=0.6, help="Global minimum interval between API requests in seconds")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "chunks"), exist_ok=True)

    target_pairs = min(args.target_pairs, 50) if args.dry_run else args.target_pairs
    target_identity = int(target_pairs * args.identity_ratio)
    target_synth = max(1, target_pairs - target_identity)

    seeds = load_seeds(args.seeds_file, args.max_seeds)
    if args.limit:
        seeds = seeds[:args.limit]
    if len(seeds) < 10:
        raise RuntimeError(f"Not enough seeds: {len(seeds)}")

    progress = load_progress(args.output_dir) if args.resume else {
        "done_seed_keys": [],
        "pairs_written": 0,
        "identity_written": 0,
        "spent_usd": 0.0,
    }
    base_spent_usd = float(progress.get("spent_usd", 0.0) or 0.0)
    done_seed_keys = set(progress.get("done_seed_keys", []))

    synth_path = os.path.join(args.output_dir, "chunks", "synthetic.jsonl")
    identity_path = os.path.join(args.output_dir, "chunks", "identity.jsonl")
    samples_path = os.path.join(args.output_dir, "dry_run_samples.json") if args.dry_run else None
    if args.resume and progress.get("pairs_written", 0) and not os.path.exists(synth_path):
        raise RuntimeError(
            "Resume state is inconsistent: progress.json claims synthetic rows exist, "
            "but chunks/synthetic.jsonl is missing. Refusing to continue and silently lose data."
        )

    rng = random.Random(42)
    indexed_seeds = list(enumerate(seeds))
    rng.shuffle(indexed_seeds)

    # Add identity examples first from held clean seeds.
    identity_rows: list[dict] = []
    for seed_idx, seed in indexed_seeds:
        if len(identity_rows) >= target_identity:
            break
        identity_rows.append(make_identity_row(seed))
    if len(identity_rows) < target_identity:
        for i in range(target_identity - len(identity_rows)):
            seed_idx, seed = indexed_seeds[i % len(indexed_seeds)]
            identity_rows.append(make_identity_row(seed))
    if identity_rows:
        append_jsonl(identity_path, identity_rows)
    progress["identity_written"] = len(identity_rows)

    synth_rows: list[dict] = []
    dry_samples: list[dict] = []
    failures: dict[str, int] = {}
    pairs_written_total = int(progress.get("pairs_written", 0) or 0)
    tasks: list[tuple[int, str, tuple[str, str, str, int], str]] = []
    all_pairs: list[tuple[int, str, tuple[str, str, str, int], str]] = []
    for seed_idx, seed in indexed_seeds:
        for profile in ERROR_PROFILES:
            key = f"{seed_idx}:{profile[0]}"
            all_pairs.append((seed_idx, seed, profile, key))
    rng.shuffle(all_pairs)
    for seed_idx, seed, profile, key in all_pairs:
        if len(tasks) >= target_synth:
            break
        if key in done_seed_keys:
            continue
        tasks.append((seed_idx, seed, profile, key))

    print(
        f"Generating synthetic GEC pairs: target={target_pairs} "
        f"(synthetic={target_synth}, identity={target_identity}) "
        f"tasks={len(tasks)}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        future_map = {
            ex.submit(process_seed, seed, seed_idx, profile, args.strict, args.min_interval): (seed_idx, seed, profile, key)
            for seed_idx, seed, profile, key in tasks
        }
        for done_count, future in enumerate(as_completed(future_map), start=1):
            seed_idx, seed, profile, key = future_map[future]
            try:
                row, err = future.result()
            except Exception as exc:
                row, err = None, str(exc)

            done_seed_keys.add(key)
            if row:
                synth_rows.append(row)
                pairs_written_total += 1
                if args.dry_run and len(dry_samples) < 12:
                    dry_samples.append({
                        "seed": seed,
                        "profile": profile[0],
                        "incorrect": row["input"],
                        "correct": row["output"],
                        "meta": row["meta"],
                    })
            else:
                failures[err or "unknown"] = failures.get(err or "unknown", 0) + 1

            if done_count % 10 == 0 or done_count == len(tasks):
                if synth_rows:
                    append_jsonl(synth_path, synth_rows)
                    synth_rows = []
                current_spent = base_spent_usd + TRACKER.cost_usd()
                print(
                    f"  done={done_count}/{len(tasks)} good={pairs_written_total} "
                    f"fail={sum(failures.values())} [{TRACKER.summary()}]",
                    flush=True,
                )
                progress["done_seed_keys"] = sorted(done_seed_keys)
                progress["pairs_written"] = pairs_written_total
                progress["spent_usd"] = current_spent
                save_progress(args.output_dir, progress)
                if args.budget_usd and current_spent >= args.budget_usd:
                    print("Budget reached, stopping early.", flush=True)
                    break

    if synth_rows:
        append_jsonl(synth_path, synth_rows)
    progress["done_seed_keys"] = sorted(done_seed_keys)
    progress["pairs_written"] = pairs_written_total
    progress["spent_usd"] = base_spent_usd + TRACKER.cost_usd()
    save_progress(args.output_dir, progress)

    if args.dry_run and samples_path:
        with open(samples_path, "w") as f:
            json.dump({
                "model": MODEL,
                "tracker": TRACKER.summary(),
                "synthetic_rows": len(synth_rows),
                "identity_rows": len(identity_rows),
                "failures": failures,
                "samples": dry_samples,
            }, f, indent=2, ensure_ascii=False)

    print("\nDone.", flush=True)
    print(f"  synthetic: {len(synth_rows)}", flush=True)
    print(f"  identity:  {len(identity_rows)}", flush=True)
    print(f"  failures:  {failures}", flush=True)
    print(f"  spend:     ${progress['spent_usd']:.4f}", flush=True)
    print(f"  output:    {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
