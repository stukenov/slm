#!/usr/bin/env python3
"""Rule-based synthetic GEC pair generator for Kazakh.

Generates 1-2M (input, target) pairs by corrupting clean Kazakh text
from MDBKD corpus. Covers all 37 error categories from the taxonomy.

Usage:
    python generate_synthetic_rulebased.py --num_pairs 1000000 --output data/synthetic_rulebased.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BACK_VOWELS = "аоұы"
FRONT_VOWELS = "әөүі"
ALL_VOWELS = set(BACK_VOWELS + FRONT_VOWELS)

VOICED_CONS = set("бвгғджз")
VOICELESS_CONS = set("кқпстфхцчшщ")
SONORANTS = set("лмнңр")

# Kazakh-specific letter confusions (common L1 errors and Russian-keyboard errors)
CONFUSABLE = {
    "қ": "к", "к": "қ",
    "ғ": "г", "г": "ғ",
    "ү": "у", "у": "ү",
    "ұ": "у",
    "ө": "о", "о": "ө",
    "ә": "а", "а": "ә",
    "і": "и", "и": "і",
    "ң": "н", "н": "ң",
    "һ": "х", "х": "һ",
}

# Case suffixes: back/front variants
# Format: {case_name: [(back_after_vowel, front_after_vowel, back_after_cons, front_after_cons), ...]}
CASE_SUFFIXES = {
    "genitive": [
        ("ның", "нің", "ның", "нің"),  # after vowel/sonorant
        ("тың", "тің", "тың", "тің"),  # after voiceless
        ("дың", "дің", "дың", "дің"),  # after voiced
    ],
    "dative": [
        ("ға", "ге", "ға", "ге"),      # after vowel/voiced/sonorant
        ("қа", "ке", "қа", "ке"),      # after voiceless
    ],
    "accusative": [
        ("ны", "ні", "ны", "ні"),      # after vowel
        ("ды", "ді", "ды", "ді"),      # after voiced/sonorant
        ("ты", "ті", "ты", "ті"),      # after voiceless
    ],
    "locative": [
        ("да", "де", "да", "де"),      # after vowel/voiced/sonorant
        ("та", "те", "та", "те"),      # after voiceless
    ],
    "ablative": [
        ("дан", "ден", "дан", "ден"),
        ("тан", "тен", "тан", "тен"),
        ("нан", "нен", "нан", "нен"),
    ],
    "instrumental": [
        ("мен", "мен", "мен", "мен"),
        ("бен", "бен", "бен", "бен"),
        ("пен", "пен", "пен", "пен"),
    ],
}

FLAT_CASE_SUFFIXES = []
for variants in CASE_SUFFIXES.values():
    for quad in variants:
        FLAT_CASE_SUFFIXES.extend(quad)
FLAT_CASE_SUFFIXES = list(set(FLAT_CASE_SUFFIXES))

PLURAL_BACK = ["лар", "дар", "тар"]
PLURAL_FRONT = ["лер", "дер", "тер"]
ALL_PLURALS = PLURAL_BACK + PLURAL_FRONT

POSSESSIVE_SUFFIXES = [
    "м", "ым", "ім",        # 1sg
    "ң", "ың", "ің",        # 2sg
    "ңыз", "ңіз",           # 2sg formal
    "сы", "сі", "ы", "і",   # 3sg
    "мыз", "міз",           # 1pl
    "лары", "лері",         # 3pl
]

PERSONAL_ENDINGS = [
    "мын", "мін", "бын", "бін", "пын", "пін",  # 1sg
    "сың", "сің",                                # 2sg
    "сыз", "сіз",                                # 2sg formal
    "ды", "ді", "ты", "ті",                      # 3sg past
    "мыз", "міз", "быз", "біз", "пыз", "піз",  # 1pl
]

TENSE_MARKERS = {
    "past": ["ды", "ді", "ты", "ті"],
    "present": ["а", "е", "й"],
    "future": ["мақ", "мек", "ар", "ер"],
}

POSTPOSITIONS = [
    "туралы", "үшін", "арқылы", "бойынша", "сияқты", "тәрізді",
    "кейін", "бұрын", "дейін", "бері", "сайын", "жайлы", "басқа",
    "қарай", "қарсы", "дейін", "шейін", "салды",
]

CONNECTORS = [
    "бірақ", "алайда", "сондықтан", "себебі", "өйткені", "сонымен",
    "яғни", "демек", "мәселен", "мысалы", "сондай-ақ", "сонда",
    "дегенмен", "сөйтіп", "осылайша", "тіпті", "әсіресе",
]

DERIVATIONAL_SUFFIXES = [
    "шы", "ші", "лық", "лік", "дық", "дік", "тық", "тік",
    "шылық", "шілік", "сыз", "сіз", "лы", "лі", "ды", "ді",
]

COPULAS = ["болды", "еді", "екен", "емес", "бар", "жоқ"]


def is_back(word: str) -> bool:
    for c in reversed(word.lower()):
        if c in BACK_VOWELS:
            return True
        if c in FRONT_VOWELS:
            return False
    return True


def swap_harmony(suffix: str) -> str:
    """Swap vowel harmony: back↔front."""
    table = str.maketrans("аоұыәөүі", "әөүіаоұы")
    return suffix.translate(table)


def corrupt_spelling(word: str) -> str | None:
    """Random character-level typo."""
    if len(word) < 3:
        return None
    ops = ["swap_char", "delete_char", "double_char", "confuse_char"]
    op = random.choice(ops)
    chars = list(word)
    idx = random.randint(0, len(chars) - 1)

    if op == "swap_char" and len(chars) > 1:
        j = min(idx + 1, len(chars) - 1)
        if idx != j:
            chars[idx], chars[j] = chars[j], chars[idx]
    elif op == "delete_char":
        chars.pop(idx)
    elif op == "double_char":
        chars.insert(idx, chars[idx])
    elif op == "confuse_char":
        c = chars[idx].lower()
        if c in CONFUSABLE:
            repl = CONFUSABLE[c]
            chars[idx] = repl.upper() if chars[idx].isupper() else repl
        else:
            return None
    result = "".join(chars)
    return result if result != word else None


def corrupt_vowel_harmony(word: str) -> str | None:
    """Break vowel harmony in a suffix."""
    if len(word) < 4:
        return None
    vowel_positions = [i for i, c in enumerate(word) if c.lower() in ALL_VOWELS]
    if len(vowel_positions) < 2:
        return None
    # Flip a vowel in the latter half
    target_pos = vowel_positions[len(vowel_positions) // 2:]
    pos = random.choice(target_pos)
    c = word[pos]
    swapped = swap_harmony(c)
    if swapped == c:
        return None
    return word[:pos] + swapped + word[pos + 1:]


def corrupt_case(words: list[str]) -> tuple[list[str], str] | None:
    """Swap a case suffix for a wrong one."""
    candidates = []
    for i, w in enumerate(words):
        wl = w.lower()
        for sfx in FLAT_CASE_SUFFIXES:
            if wl.endswith(sfx) and len(wl) > len(sfx) + 1:
                candidates.append((i, sfx))
    if not candidates:
        return None
    idx, old_sfx = random.choice(candidates)
    new_sfx = random.choice(FLAT_CASE_SUFFIXES)
    while new_sfx == old_sfx:
        new_sfx = random.choice(FLAT_CASE_SUFFIXES)
    w = words[idx]
    stem = w[:len(w) - len(old_sfx)]
    new_words = words.copy()
    new_words[idx] = stem + new_sfx
    return new_words, "case"


def corrupt_plural(words: list[str]) -> tuple[list[str], str] | None:
    """Add/remove/swap plural suffix."""
    op = random.choice(["add", "remove", "swap"])
    if op == "remove":
        for i, w in enumerate(words):
            wl = w.lower()
            for sfx in ALL_PLURALS:
                if wl.endswith(sfx) and len(wl) > len(sfx) + 2:
                    new_words = words.copy()
                    new_words[i] = w[:len(w) - len(sfx)]
                    return new_words, "plural/missing_plural"
    elif op == "add":
        candidates = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
        if candidates:
            idx = random.choice(candidates)
            sfx = random.choice(PLURAL_BACK if is_back(words[idx]) else PLURAL_FRONT)
            new_words = words.copy()
            new_words[idx] = words[idx] + sfx
            return new_words, "plural/extra_plural"
    elif op == "swap":
        for i, w in enumerate(words):
            wl = w.lower()
            for sfx in ALL_PLURALS:
                if wl.endswith(sfx) and len(wl) > len(sfx) + 2:
                    wrong = random.choice([s for s in ALL_PLURALS if s != sfx])
                    new_words = words.copy()
                    new_words[i] = w[:len(w) - len(sfx)] + wrong
                    return new_words, "plural/allomorph"
    return None


def corrupt_spacing(text: str) -> str | None:
    """Merge or split words."""
    words = text.split()
    if len(words) < 2:
        return None
    if random.random() < 0.5:
        # merge two adjacent words
        idx = random.randint(0, len(words) - 2)
        merged = words[idx] + words[idx + 1]
        return " ".join(words[:idx] + [merged] + words[idx + 2:])
    else:
        # split a word
        candidates = [i for i, w in enumerate(words) if len(w) > 5]
        if not candidates:
            return None
        idx = random.choice(candidates)
        w = words[idx]
        pos = random.randint(2, len(w) - 2)
        return " ".join(words[:idx] + [w[:pos], w[pos:]] + words[idx + 1:])


def corrupt_punctuation(text: str) -> str | None:
    """Remove, add, or swap punctuation."""
    ops = ["remove", "add", "swap"]
    op = random.choice(ops)
    if op == "remove":
        puncts = [(m.start(), m.group()) for m in re.finditer(r"[,.:;!?]", text)]
        if not puncts:
            return None
        pos, _ = random.choice(puncts)
        return text[:pos] + text[pos + 1:]
    elif op == "add":
        words = text.split()
        if len(words) < 3:
            return None
        idx = random.randint(1, len(words) - 2)
        p = random.choice([",", ".", ";"])
        words[idx] = words[idx] + p
        return " ".join(words)
    elif op == "swap":
        puncts = [(m.start(), m.group()) for m in re.finditer(r"[,.:;!?]", text)]
        if not puncts:
            return None
        pos, old = random.choice(puncts)
        options = [p for p in [",", ".", ";", "!", "?"] if p != old]
        new = random.choice(options)
        return text[:pos] + new + text[pos + 1:]
    return None


def corrupt_word_order(words: list[str]) -> list[str] | None:
    """Swap two adjacent words."""
    if len(words) < 3:
        return None
    idx = random.randint(0, len(words) - 2)
    new_words = words.copy()
    new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
    if new_words == words:
        return None
    return new_words


def corrupt_missing_element(words: list[str]) -> list[str] | None:
    """Drop a word."""
    if len(words) < 4:
        return None
    idx = random.randint(1, len(words) - 2)
    return words[:idx] + words[idx + 1:]


def corrupt_redundant_element(words: list[str]) -> list[str] | None:
    """Duplicate a word."""
    if len(words) < 3:
        return None
    idx = random.randint(0, len(words) - 1)
    return words[:idx] + [words[idx], words[idx]] + words[idx + 1:]


def corrupt_postposition(words: list[str]) -> list[str] | None:
    """Swap a postposition for a wrong one."""
    for i, w in enumerate(words):
        if w.lower() in [p.lower() for p in POSTPOSITIONS]:
            wrong = random.choice([p for p in POSTPOSITIONS if p.lower() != w.lower()])
            new_words = words.copy()
            new_words[i] = wrong
            return new_words
    return None


def corrupt_connector(words: list[str]) -> list[str] | None:
    """Swap a connector for a wrong one."""
    for i, w in enumerate(words):
        if w.lower().rstrip(",.;") in [c.lower() for c in CONNECTORS]:
            wrong = random.choice(CONNECTORS)
            trail = ""
            for c in reversed(w):
                if c in ",.;":
                    trail = c + trail
                else:
                    break
            new_words = words.copy()
            new_words[i] = wrong + trail
            return new_words
    return None


def corrupt_possessive(word: str) -> str | None:
    """Swap possessive suffix."""
    wl = word.lower()
    for sfx in sorted(POSSESSIVE_SUFFIXES, key=len, reverse=True):
        if wl.endswith(sfx) and len(wl) > len(sfx) + 2:
            wrong = random.choice([s for s in POSSESSIVE_SUFFIXES if s != sfx])
            return word[:len(word) - len(sfx)] + wrong
    return None


def corrupt_personal_ending(word: str) -> str | None:
    """Swap personal verb ending."""
    wl = word.lower()
    for sfx in sorted(PERSONAL_ENDINGS, key=len, reverse=True):
        if wl.endswith(sfx) and len(wl) > len(sfx) + 2:
            wrong = random.choice([s for s in PERSONAL_ENDINGS if s != sfx])
            return word[:len(word) - len(sfx)] + wrong
    return None


def corrupt_tense(word: str) -> str | None:
    """Swap tense marker."""
    wl = word.lower()
    for tense, markers in TENSE_MARKERS.items():
        for m in sorted(markers, key=len, reverse=True):
            if wl.endswith(m) and len(wl) > len(m) + 2:
                other_tenses = [t for t in TENSE_MARKERS if t != tense]
                new_tense = random.choice(other_tenses)
                new_marker = random.choice(TENSE_MARKERS[new_tense])
                return word[:len(word) - len(m)] + new_marker
    return None


def corrupt_negation(words: list[str]) -> list[str] | None:
    """Add double negation or remove negation."""
    neg_words = ["емес", "жоқ"]
    neg_indices = [i for i, w in enumerate(words) if w.lower() in neg_words]
    if neg_indices:
        # double negation: add another negation word
        idx = random.choice(neg_indices)
        new_words = words.copy()
        neg = random.choice(neg_words)
        new_words.insert(idx, neg)
        return new_words
    return None


def corrupt_derivational(word: str) -> str | None:
    """Swap derivational suffix."""
    wl = word.lower()
    for sfx in sorted(DERIVATIONAL_SUFFIXES, key=len, reverse=True):
        if wl.endswith(sfx) and len(wl) > len(sfx) + 2:
            wrong = random.choice([s for s in DERIVATIONAL_SUFFIXES if s != sfx])
            return word[:len(word) - len(sfx)] + wrong
    return None


def corrupt_run_on(text: str) -> str | None:
    """Remove sentence boundary (merge two sentences)."""
    for sep in [". ", "! ", "? "]:
        if sep in text:
            pos = text.index(sep)
            if pos > 5 and pos < len(text) - 5:
                after = text[pos + 2:]
                if after and after[0].isupper():
                    return text[:pos + 1] + " " + after[0].lower() + after[1:]
    return None


def corrupt_missing_copula(words: list[str]) -> list[str] | None:
    """Remove copula if present."""
    for i, w in enumerate(words):
        if w.lower() in [c.lower() for c in COPULAS]:
            return words[:i] + words[i + 1:]
    return None


# All corruption functions with their error tags and relative weights
CORRUPTIONS = [
    ("orthography/spelling", 25),
    ("orthography/vowel_harmony", 15),
    ("orthography/spacing", 10),
    ("orthography/punctuation", 8),
    ("morphosyntax/case", 15),
    ("morphosyntax/plural", 8),
    ("morphosyntax/possessive", 5),
    ("morphosyntax/personal_ending", 5),
    ("morphosyntax/tense", 4),
    ("morphosyntax/negation", 2),
    ("morphosyntax/postposition", 3),
    ("morphosyntax/agreement", 3),
    ("morphosyntax/derivation", 3),
    ("syntax_discourse/word_order", 5),
    ("syntax_discourse/missing_element", 4),
    ("syntax_discourse/redundant_element", 3),
    ("syntax_discourse/clause_structure", 2),
    ("syntax_discourse/discourse", 2),
]

TAGS = [t for t, _ in CORRUPTIONS]
WEIGHTS = [w for _, w in CORRUPTIONS]


def apply_corruption(text: str, tag: str) -> str | None:
    """Apply a single corruption to text. Returns corrupted text or None if not applicable."""
    words = text.split()

    if tag == "orthography/spelling":
        candidates = [i for i, w in enumerate(words) if len(w) > 2 and w.isalpha()]
        if not candidates:
            return None
        idx = random.choice(candidates)
        corrupted = corrupt_spelling(words[idx])
        if corrupted is None:
            return None
        new_words = words.copy()
        new_words[idx] = corrupted
        return " ".join(new_words)

    elif tag == "orthography/vowel_harmony":
        candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
        if not candidates:
            return None
        idx = random.choice(candidates)
        corrupted = corrupt_vowel_harmony(words[idx])
        if corrupted is None:
            return None
        new_words = words.copy()
        new_words[idx] = corrupted
        return " ".join(new_words)

    elif tag == "orthography/spacing":
        return corrupt_spacing(text)

    elif tag == "orthography/punctuation":
        return corrupt_punctuation(text)

    elif tag == "morphosyntax/case":
        result = corrupt_case(words)
        if result is None:
            return None
        new_words, _ = result
        return " ".join(new_words)

    elif tag == "morphosyntax/plural":
        result = corrupt_plural(words)
        if result is None:
            return None
        new_words, _ = result
        return " ".join(new_words)

    elif tag == "morphosyntax/possessive":
        candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
        if not candidates:
            return None
        idx = random.choice(candidates)
        corrupted = corrupt_possessive(words[idx])
        if corrupted is None:
            return None
        new_words = words.copy()
        new_words[idx] = corrupted
        return " ".join(new_words)

    elif tag == "morphosyntax/personal_ending":
        candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
        if not candidates:
            return None
        idx = random.choice(candidates)
        corrupted = corrupt_personal_ending(words[idx])
        if corrupted is None:
            return None
        new_words = words.copy()
        new_words[idx] = corrupted
        return " ".join(new_words)

    elif tag == "morphosyntax/tense":
        candidates = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
        if not candidates:
            return None
        idx = random.choice(candidates)
        corrupted = corrupt_tense(words[idx])
        if corrupted is None:
            return None
        new_words = words.copy()
        new_words[idx] = corrupted
        return " ".join(new_words)

    elif tag == "morphosyntax/negation":
        result = corrupt_negation(words)
        if result is None:
            return None
        return " ".join(result)

    elif tag == "morphosyntax/postposition":
        result = corrupt_postposition(words)
        if result is None:
            return None
        return " ".join(result)

    elif tag == "morphosyntax/agreement":
        # Swap suffix on a random word to break agreement
        candidates = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
        if not candidates:
            return None
        idx = random.choice(candidates)
        w = words[idx]
        # Flip harmony on last 2-3 chars
        if len(w) > 3:
            suffix_start = max(len(w) - 3, len(w) // 2)
            prefix = w[:suffix_start]
            suffix = w[suffix_start:]
            flipped = swap_harmony(suffix)
            if flipped != suffix:
                new_words = words.copy()
                new_words[idx] = prefix + flipped
                return " ".join(new_words)
        return None

    elif tag == "morphosyntax/derivation":
        candidates = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
        if not candidates:
            return None
        idx = random.choice(candidates)
        corrupted = corrupt_derivational(words[idx])
        if corrupted is None:
            return None
        new_words = words.copy()
        new_words[idx] = corrupted
        return " ".join(new_words)

    elif tag == "syntax_discourse/word_order":
        result = corrupt_word_order(words)
        if result is None:
            return None
        return " ".join(result)

    elif tag == "syntax_discourse/missing_element":
        result = corrupt_missing_element(words)
        if result is None:
            return None
        return " ".join(result)

    elif tag == "syntax_discourse/redundant_element":
        result = corrupt_redundant_element(words)
        if result is None:
            return None
        return " ".join(result)

    elif tag == "syntax_discourse/clause_structure":
        return corrupt_run_on(text)

    elif tag == "syntax_discourse/discourse":
        result = corrupt_connector(words)
        if result is None:
            return None
        return " ".join(result)

    return None


def generate_pair(text: str, max_corruptions: int = 2) -> tuple[str, str, list[str]] | None:
    """Generate a (corrupted, clean) pair from clean text.

    Returns (corrupted_text, clean_text, error_tags) or None.
    """
    n_corruptions = random.choices([1, 2], weights=[0.7, 0.3])[0]
    n_corruptions = min(n_corruptions, max_corruptions)

    selected_tags = random.choices(TAGS, weights=WEIGHTS, k=n_corruptions)
    # deduplicate
    selected_tags = list(dict.fromkeys(selected_tags))

    corrupted = text
    applied_tags = []
    for tag in selected_tags:
        result = apply_corruption(corrupted, tag)
        if result is not None and result != corrupted:
            corrupted = result
            applied_tags.append(tag)

    if not applied_tags or corrupted == text:
        return None

    return corrupted, text, applied_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=1_500_000)
    parser.add_argument("--output", default="data/synthetic_rulebased.jsonl")
    parser.add_argument("--dataset", default="stukenov/sozkz-corpus-clean-v3")
    parser.add_argument("--max_ppl", type=float, default=40.0,
                        help="Maximum perplexity threshold (lower = higher quality)")
    parser.add_argument("--min_words", type=int, default=4)
    parser.add_argument("--max_words", type=int, default=40)
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument("--max_chars", type=int, default=300)
    parser.add_argument("--identity_ratio", type=float, default=0.1,
                        help="Fraction of pairs where input=target (identity/no-error)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_log", type=int, default=50000)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_identity = int(args.num_pairs * args.identity_ratio)
    n_corrupted = args.num_pairs - n_identity

    logger.info("Loading dataset: %s (streaming), max_ppl=%.1f", args.dataset, args.max_ppl)
    ds = load_dataset(args.dataset, split="train", streaming=True)

    texts = []
    skipped_ppl = 0
    logger.info("Collecting clean Kazakh texts (PPL <= %.1f)...", args.max_ppl)
    needed = n_corrupted * 3
    for row in ds:
        text = row.get("text", "").strip()
        ppl = float(row.get("ppl", 999))
        if ppl > args.max_ppl:
            skipped_ppl += 1
            continue
        n_words = text.count(" ") + 1
        if args.min_chars <= len(text) <= args.max_chars and args.min_words <= n_words <= args.max_words:
            texts.append(text)
            if len(texts) >= needed:
                break
        if len(texts) % 500_000 == 0 and len(texts) > 0:
            logger.info("  collected %d texts (skipped %d high-PPL)...", len(texts), skipped_ppl)

    logger.info("Collected %d clean texts", len(texts))
    random.shuffle(texts)

    pairs = []
    tag_counts: dict[str, int] = {}
    text_idx = 0

    logger.info("Generating %d corrupted pairs...", n_corrupted)
    while len(pairs) < n_corrupted and text_idx < len(texts):
        text = texts[text_idx]
        text_idx += 1
        result = generate_pair(text)
        if result is None:
            continue
        corrupted, clean, tags = result
        pairs.append({"input": corrupted, "target": clean, "error_tags": tags})
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

        if len(pairs) % args.batch_log == 0:
            logger.info("  %d/%d pairs generated (used %d/%d texts)",
                        len(pairs), n_corrupted, text_idx, len(texts))

    logger.info("Generated %d corrupted pairs from %d texts", len(pairs), text_idx)

    # Add identity pairs
    logger.info("Adding %d identity pairs...", n_identity)
    identity_texts = texts[text_idx:text_idx + n_identity]
    if len(identity_texts) < n_identity:
        identity_texts += random.sample(texts[:text_idx], min(n_identity - len(identity_texts), text_idx))
    for text in identity_texts[:n_identity]:
        pairs.append({"input": text, "target": text, "error_tags": ["identity"]})

    random.shuffle(pairs)

    logger.info("Writing %d pairs to %s", len(pairs), output_path)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info("Tag distribution:")
    for tag in sorted(tag_counts, key=tag_counts.get, reverse=True):
        logger.info("  %-40s %6d (%.1f%%)", tag, tag_counts[tag],
                     tag_counts[tag] / len(pairs) * 100)
    logger.info("  %-40s %6d (%.1f%%)", "identity", n_identity, n_identity / len(pairs) * 100)
    logger.info("DONE! Total: %d pairs", len(pairs))


if __name__ == "__main__":
    main()
