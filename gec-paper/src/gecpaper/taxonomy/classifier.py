from __future__ import annotations

import re
from difflib import SequenceMatcher

from gecpaper.taxonomy.schema import ErrorAnnotation, Level1, Level2, Level3

BACK_VOWELS = set("аоұы")
FRONT_VOWELS = set("әөүі")
ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS

POSTPOSITIONS = {
    "туралы", "үшін", "арқылы", "бойынша", "сияқты", "тәрізді",
    "кейін", "бұрын", "дейін", "бері", "сайын", "жайлы", "басқа",
}

PLURAL_SUFFIXES = {"лар", "лер", "дар", "дер", "тар", "тер"}


def classify(source: str, target: str) -> ErrorAnnotation:
    src = source.strip()
    tgt = target.strip()

    if src == tgt:
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPELLING)

    src_words = src.split()
    tgt_words = tgt.split()

    src_nospace = src.replace(" ", "")
    tgt_nospace = tgt.replace(" ", "")
    if src_nospace == tgt_nospace:
        l3 = Level3.EXTRA_SPACE if len(src_words) > len(tgt_words) else Level3.MISSING_SPACE
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPACING, l3=l3)

    if len(src_words) == len(tgt_words) and sorted(src_words) == sorted(tgt_words):
        return ErrorAnnotation(l1=Level1.SYNTAX_DISCOURSE, l2=Level2.WORD_ORDER)

    src_letters = re.sub(r"[^\w\s]", "", src)
    tgt_letters = re.sub(r"[^\w\s]", "", tgt)
    if src_letters == tgt_letters:
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.PUNCTUATION)

    if len(src_words) != len(tgt_words):
        diff = abs(len(src_words) - len(tgt_words))
        if diff >= 2:
            if len(src_words) > len(tgt_words):
                return ErrorAnnotation(l1=Level1.SYNTAX_DISCOURSE, l2=Level2.REDUNDANT_ELEMENT)
            return ErrorAnnotation(l1=Level1.SYNTAX_DISCOURSE, l2=Level2.MISSING_ELEMENT)

    changed_pairs = []
    if len(src_words) == len(tgt_words):
        for sw, tw in zip(src_words, tgt_words):
            if sw != tw:
                changed_pairs.append((sw, tw))

    if not changed_pairs and len(src_words) != len(tgt_words):
        return ErrorAnnotation(l1=Level1.SYNTAX_DISCOURSE, l2=Level2.CLAUSE_STRUCTURE)

    if len(changed_pairs) == 1:
        sw, tw = changed_pairs[0]
        ann = _classify_word_pair(sw, tw)
        if ann is not None:
            return ann

    if changed_pairs:
        all_suffix = True
        for sw, tw in changed_pairs:
            ratio = SequenceMatcher(None, sw, tw).ratio()
            if ratio < 0.5:
                all_suffix = False
                break
        if all_suffix:
            return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.AGREEMENT)

    overall = SequenceMatcher(None, src, tgt).ratio()
    if overall > 0.85:
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPELLING)

    return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.CASE)


def _classify_word_pair(src_word: str, tgt_word: str) -> ErrorAnnotation | None:
    sw_lower = src_word.lower()
    tw_lower = tgt_word.lower()

    if sw_lower in POSTPOSITIONS or tw_lower in POSTPOSITIONS:
        return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.POSTPOSITION)

    prefix_len = 0
    for a, b in zip(sw_lower, tw_lower):
        if a == b:
            prefix_len += 1
        else:
            break

    src_suffix = sw_lower[prefix_len:]
    tgt_suffix = tw_lower[prefix_len:]

    src_vowels = [c for c in src_suffix if c in ALL_VOWELS]
    tgt_vowels = [c for c in tgt_suffix if c in ALL_VOWELS]
    if src_vowels and tgt_vowels:
        src_back = any(v in BACK_VOWELS for v in src_vowels)
        tgt_back = any(v in BACK_VOWELS for v in tgt_vowels)
        if src_back != tgt_back:
            return ErrorAnnotation(
                l1=Level1.ORTHOGRAPHY, l2=Level2.VOWEL_HARMONY,
                l3=Level3.FRONT_BACK_MISMATCH,
            )

    if src_suffix in PLURAL_SUFFIXES or tgt_suffix in PLURAL_SUFFIXES:
        return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.PLURAL, l3=Level3.ALLOMORPH)

    if prefix_len >= 2 and len(src_suffix) <= 4 and len(tgt_suffix) <= 4:
        return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.CASE)

    if SequenceMatcher(None, sw_lower, tw_lower).ratio() > 0.8:
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPELLING)

    return None
