from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Level1(Enum):
    ORTHOGRAPHY = "orthography"
    MORPHOSYNTAX = "morphosyntax"
    SYNTAX_DISCOURSE = "syntax_discourse"


class Level2(Enum):
    SPELLING = "spelling"
    VOWEL_HARMONY = "vowel_harmony"
    SPACING = "spacing"
    PUNCTUATION = "punctuation"
    CASE = "case"
    POSSESSIVE = "possessive"
    PERSONAL_ENDING = "personal_ending"
    PLURAL = "plural"
    NEGATION = "negation"
    TENSE = "tense"
    POSTPOSITION = "postposition"
    AGREEMENT = "agreement"
    DERIVATION = "derivation"
    WORD_ORDER = "word_order"
    CLAUSE_STRUCTURE = "clause_structure"
    MISSING_ELEMENT = "missing_element"
    REDUNDANT_ELEMENT = "redundant_element"
    DISCOURSE = "discourse"


class Level3(Enum):
    FRONT_BACK_MISMATCH = "front_back_mismatch"
    ROUNDING_HARMONY = "rounding_harmony"
    BOUNDARY_HARMONY = "boundary_harmony"
    MISSING_SPACE = "missing_space"
    EXTRA_SPACE = "extra_space"
    NOMINATIVE = "nominative"
    GENITIVE = "genitive"
    DATIVE = "dative"
    ACCUSATIVE = "accusative"
    LOCATIVE = "locative"
    ABLATIVE = "ablative"
    INSTRUMENTAL = "instrumental"
    PERSON_MISMATCH = "person_mismatch"
    NUMBER_MISMATCH = "number_mismatch"
    PERSON = "person"
    NUMBER = "number"
    TENSE_AGREEMENT = "tense_agreement"
    EXTRA_PLURAL = "extra_plural"
    MISSING_PLURAL = "missing_plural"
    ALLOMORPH = "allomorph"
    DOUBLE_NEGATION = "double_negation"
    WRONG_FORM = "wrong_form"
    PAST_PRESENT = "past_present"
    PRESENT_FUTURE = "present_future"
    PAST_FUTURE = "past_future"
    WRONG_POSTPOSITION = "wrong_postposition"
    CASE_GOVERNMENT = "case_government"
    SUBJECT_VERB = "subject_verb"
    MODIFIER_HEAD = "modifier_head"
    WRONG_DERIVATIONAL_SUFFIX = "wrong_derivational_suffix"
    VERB_POSITION = "verb_position"
    MODIFIER_POSITION = "modifier_position"
    FRAGMENTED = "fragmented"
    RUN_ON = "run_on"
    DROPPED_ARGUMENT = "dropped_argument"
    MISSING_COPULA = "missing_copula"
    REPEATED_WORD = "repeated_word"
    PLEONASM = "pleonasm"
    CONNECTOR_MISUSE = "connector_misuse"


ALL_L2_FOR_L1: dict[Level1, list[Level2]] = {
    Level1.ORTHOGRAPHY: [
        Level2.SPELLING, Level2.VOWEL_HARMONY, Level2.SPACING, Level2.PUNCTUATION,
    ],
    Level1.MORPHOSYNTAX: [
        Level2.CASE, Level2.POSSESSIVE, Level2.PERSONAL_ENDING, Level2.PLURAL,
        Level2.NEGATION, Level2.TENSE, Level2.POSTPOSITION, Level2.AGREEMENT,
        Level2.DERIVATION,
    ],
    Level1.SYNTAX_DISCOURSE: [
        Level2.WORD_ORDER, Level2.CLAUSE_STRUCTURE, Level2.MISSING_ELEMENT,
        Level2.REDUNDANT_ELEMENT, Level2.DISCOURSE,
    ],
}

ALL_L3_FOR_L2: dict[Level2, list[Level3]] = {
    Level2.VOWEL_HARMONY: [Level3.FRONT_BACK_MISMATCH, Level3.ROUNDING_HARMONY, Level3.BOUNDARY_HARMONY],
    Level2.SPACING: [Level3.MISSING_SPACE, Level3.EXTRA_SPACE],
    Level2.CASE: [
        Level3.NOMINATIVE, Level3.GENITIVE, Level3.DATIVE, Level3.ACCUSATIVE,
        Level3.LOCATIVE, Level3.ABLATIVE, Level3.INSTRUMENTAL,
    ],
    Level2.POSSESSIVE: [Level3.PERSON_MISMATCH, Level3.NUMBER_MISMATCH],
    Level2.PERSONAL_ENDING: [Level3.PERSON, Level3.NUMBER, Level3.TENSE_AGREEMENT],
    Level2.PLURAL: [Level3.EXTRA_PLURAL, Level3.MISSING_PLURAL, Level3.ALLOMORPH],
    Level2.NEGATION: [Level3.DOUBLE_NEGATION, Level3.WRONG_FORM],
    Level2.TENSE: [Level3.PAST_PRESENT, Level3.PRESENT_FUTURE, Level3.PAST_FUTURE],
    Level2.POSTPOSITION: [Level3.WRONG_POSTPOSITION, Level3.CASE_GOVERNMENT],
    Level2.AGREEMENT: [Level3.SUBJECT_VERB, Level3.MODIFIER_HEAD],
    Level2.DERIVATION: [Level3.WRONG_DERIVATIONAL_SUFFIX],
    Level2.WORD_ORDER: [Level3.VERB_POSITION, Level3.MODIFIER_POSITION],
    Level2.CLAUSE_STRUCTURE: [Level3.FRAGMENTED, Level3.RUN_ON],
    Level2.MISSING_ELEMENT: [Level3.DROPPED_ARGUMENT, Level3.MISSING_COPULA],
    Level2.REDUNDANT_ELEMENT: [Level3.REPEATED_WORD, Level3.PLEONASM],
    Level2.DISCOURSE: [Level3.CONNECTOR_MISUSE],
}

_L2_TO_L1: dict[Level2, Level1] = {}
for _l1, _l2s in ALL_L2_FOR_L1.items():
    for _l2 in _l2s:
        _L2_TO_L1[_l2] = _l1

_L3_TO_L2: dict[Level3, Level2] = {}
for _l2, _l3s in ALL_L3_FOR_L2.items():
    for _l3 in _l3s:
        _L3_TO_L2[_l3] = _l2


@dataclass(frozen=True)
class ErrorAnnotation:
    l1: Level1
    l2: Level2
    l3: Level3 | None = None

    @property
    def tag(self) -> str:
        parts = [self.l1.value, self.l2.value]
        if self.l3 is not None:
            parts.append(self.l3.value)
        return "/".join(parts)

    def to_dict(self) -> dict:
        d = {"l1": self.l1.value, "l2": self.l2.value}
        if self.l3 is not None:
            d["l3"] = self.l3.value
        return d


def parse_annotation(tag: str) -> ErrorAnnotation:
    parts = tag.split("/")
    l1 = Level1(parts[0])
    l2 = Level2(parts[1])
    l3 = Level3(parts[2]) if len(parts) > 2 else None
    return ErrorAnnotation(l1=l1, l2=l2, l3=l3)


def get_l1_for_l2(l2: Level2) -> Level1:
    return _L2_TO_L1[l2]


def get_l2_for_l3(l3: Level3) -> Level2:
    return _L3_TO_L2[l3]
