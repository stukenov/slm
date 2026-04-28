"""Detect language of the extracted text using lingua-language-detector."""
from .base import BaseAgent

# Lazy-load to avoid slow import at module level
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from lingua import Language, LanguageDetectorBuilder
        # Focus on languages relevant to Kaznet: kk, ru, en + common others
        languages = [
            Language.KAZAKH,
            Language.RUSSIAN,
            Language.ENGLISH,
            Language.GERMAN,
            Language.FRENCH,
            Language.TURKISH,
            Language.ARABIC,
            Language.CHINESE,
        ]
        _detector = LanguageDetectorBuilder.from_languages(*languages).build()
    return _detector


LINGUA_CODE = {
    "KAZAKH": "kk",
    "RUSSIAN": "ru",
    "ENGLISH": "en",
    "GERMAN": "de",
    "FRENCH": "fr",
    "TURKISH": "tr",
    "ARABIC": "ar",
    "CHINESE": "zh",
}


class LanguageAgent(BaseAgent):
    columns = ["language", "lang_score"]

    def process(self, row: dict) -> dict:
        text = row.get("text", "")
        if not text or len(text) < 20:
            return {"language": None, "lang_score": None}

        detector = get_detector()
        # Use first 2000 chars for speed
        sample = text[:2000]

        result = detector.compute_language_confidence_values(sample)
        if not result:
            return {"language": None, "lang_score": None}

        top = result[0]
        lang_name = top.language.name  # e.g. "KAZAKH"
        lang_code = LINGUA_CODE.get(lang_name, lang_name.lower()[:2])
        score = round(top.value, 4)

        return {"language": lang_code, "lang_score": score}
