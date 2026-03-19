"""Kazakh text normalization for TTS."""

from __future__ import annotations

import re

# Kazakh digits
_ONES = {
    "0": "нөл", "1": "бір", "2": "екі", "3": "үш", "4": "төрт",
    "5": "бес", "6": "алты", "7": "жеті", "8": "сегіз", "9": "тоғыз",
}
_TENS = {
    "10": "он", "20": "жиырма", "30": "отыз", "40": "қырық",
    "50": "елу", "60": "алпыс", "70": "жетпіс", "80": "сексен", "90": "тоқсан",
}
_HUNDREDS = "жүз"
_THOUSAND = "мың"
_MILLION = "миллион"
_BILLION = "миллиард"


def _number_to_words(n: int) -> str:
    """Convert integer to Kazakh words. Handles 0 to 999_999_999."""
    if n == 0:
        return _ONES["0"]
    if n < 0:
        return "минус " + _number_to_words(-n)

    parts = []

    if n >= 1_000_000_000:
        parts.append(_number_to_words(n // 1_000_000_000) + " " + _BILLION)
        n %= 1_000_000_000
    if n >= 1_000_000:
        parts.append(_number_to_words(n // 1_000_000) + " " + _MILLION)
        n %= 1_000_000
    if n >= 1000:
        q = n // 1000
        if q == 1:
            parts.append(_THOUSAND)
        else:
            parts.append(_number_to_words(q) + " " + _THOUSAND)
        n %= 1000
    if n >= 100:
        q = n // 100
        if q == 1:
            parts.append(_HUNDREDS)
        else:
            parts.append(_ONES[str(q)] + " " + _HUNDREDS)
        n %= 100
    if n >= 10:
        parts.append(_TENS[str(n // 10 * 10)])
        n %= 10
    if n > 0:
        parts.append(_ONES[str(n)])

    return " ".join(parts)


def expand_numbers(text: str) -> str:
    """Replace digit sequences with Kazakh words."""
    def _replace(m: re.Match) -> str:
        return _number_to_words(int(m.group()))
    return re.sub(r"\d+", _replace, text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, strip."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
    """Normalize quotes, dashes, etc."""
    text = re.sub(r"[«»""„]", '"', text)
    text = re.sub(r"[–—]", "-", text)
    return text


def normalize_text(text: str, expand_nums: bool = True) -> str:
    """Full normalization pipeline for Kazakh TTS."""
    text = normalize_punctuation(text)
    if expand_nums:
        text = expand_numbers(text)
    text = normalize_whitespace(text)
    return text
