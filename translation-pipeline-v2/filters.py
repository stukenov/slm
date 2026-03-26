"""Sentence-level pre-translation and post-translation filters."""

import re
from difflib import SequenceMatcher

from config import (
    MIN_WORDS_PER_SENTENCE,
    MAX_SENTENCE_LENGTH,
    NON_ALPHA_THRESHOLD,
    DUPLICATE_SIMILARITY_THRESHOLD,
    LENGTH_RATIO_MAX,
    LENGTH_RATIO_MIN,
    NGRAM_REPEAT_THRESHOLD,
)

_WORD_RE = re.compile(r'\b\w+\b', re.UNICODE)


def is_noisy_sentence(sentence: str) -> bool:
    """Pre-translation filter. Returns True if sentence should be skipped.

    Checks:
    - Too short (<MIN_WORDS words)
    - Too long (>MAX_SENTENCE_LENGTH chars)
    - Too many non-alpha characters (>NON_ALPHA_THRESHOLD ratio)
    """
    sentence = sentence.strip()
    if not sentence:
        return True

    # Too long
    if len(sentence) > MAX_SENTENCE_LENGTH:
        return True

    # Too short
    words = _WORD_RE.findall(sentence)
    if len(words) < MIN_WORDS_PER_SENTENCE:
        return True

    # Non-alpha ratio (excluding spaces)
    alpha_count = sum(1 for c in sentence if c.isalpha())
    total_count = len(sentence.replace(" ", ""))
    if total_count == 0:
        return True
    non_alpha_ratio = 1.0 - (alpha_count / total_count)
    if non_alpha_ratio > NON_ALPHA_THRESHOLD:
        return True

    return False


def char_similarity(a: str, b: str) -> float:
    """Character-level similarity between two strings (0.0 to 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def has_ngram_repetition(text: str, n: int = 2) -> bool:
    """Check if text contains repeated n-grams (sign of model looping).

    Returns True if any n-gram appears NGRAM_REPEAT_THRESHOLD+ times.
    """
    words = text.split()
    if len(words) < n + NGRAM_REPEAT_THRESHOLD:
        return False

    ngram_counts: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        if ngram_counts[ngram] >= NGRAM_REPEAT_THRESHOLD:
            return True
    return False


def is_translation_bad(original: str, translated: str) -> bool:
    """Post-translation filter. Returns True if translation should be skipped.

    Checks:
    - Translation too similar to original (copy-through)
    - N-gram repetition (model looping)
    - Length ratio too extreme
    - Disproportionate special characters
    """
    if not translated or not translated.strip():
        return True

    # Translation ≈ original (copy-through)
    if char_similarity(original, translated) > DUPLICATE_SIMILARITY_THRESHOLD:
        return True

    # N-gram repetition
    if has_ngram_repetition(translated):
        return True

    # Length ratio
    len_orig = len(original)
    len_trans = len(translated)
    if len_orig > 0:
        ratio = len_trans / len_orig
        if ratio > LENGTH_RATIO_MAX or ratio < LENGTH_RATIO_MIN:
            return True

    # Special char ratio: if translated has way more special chars than original
    orig_special = sum(1 for c in original if not c.isalpha() and not c.isspace())
    trans_special = sum(1 for c in translated if not c.isalpha() and not c.isspace())
    orig_ratio = orig_special / max(len(original), 1)
    trans_ratio = trans_special / max(len(translated), 1)
    if trans_ratio > orig_ratio * 2 + 0.1:
        return True

    return False
