"""
exp_016/router.py — Lingua language detector

lingua: 100% точность на KK/RU/EN, 0.1-60ms
"""

from lingua import Language, LanguageDetectorBuilder

_DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.RUSSIAN, Language.KAZAKH
).build()

_LANG_MAP = {
    Language.ENGLISH: "en",
    Language.RUSSIAN: "ru",
    Language.KAZAKH: "kk",
}


def detect_lang(text: str) -> str:
    """Определяет язык: kk, ru, en."""
    result = _DETECTOR.detect_language_of(text)
    return _LANG_MAP.get(result, "en")
