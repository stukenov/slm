"""
exp_012/router.py — Lang detect + rule-based router

Определяет язык через langdetect, домен через regex/keywords.
"""

import re
from langdetect import detect, DetectorFactory

# Фиксируем seed для воспроизводимости
DetectorFactory.seed = 0

# Паттерны кода
CODE_PY = re.compile(r"^print\s*\(")
CODE_JS = re.compile(r"^console\.log\s*\(")

# Математические ключевые слова по языкам
MATH_KW_KK = {"қосу", "алу", "нешеге", "тең"}
MATH_KW_RU = {"сколько", "будет", "плюс", "минус", "равно"}
MATH_KW_EN = {"what", "plus", "minus"}

# Уникальные казахские буквы (не встречаются в русском)
KK_CHARS = set("қңғүұіөәһ")
# Русские буквы (кириллица без казахских уникальных)
RU_CHARS = set("абвгдежзийклмнопрстуфхцчшщъыьэюя")


def detect_lang(text: str) -> str:
    """Определяет язык: kk, ru, en. Приоритет: символьный анализ > langdetect."""
    lower = text.lower()

    # Казахские уникальные символы — точный маркер
    if any(c in KK_CHARS for c in lower):
        return "kk"

    # Кириллица без казахских — русский
    if any(c in RU_CHARS for c in lower):
        return "ru"

    # Fallback на langdetect для латиницы
    try:
        lang = detect(text)
    except Exception:
        return "en"
    if lang in ("kk", "ky"):
        return "kk"
    if lang in ("ru", "uk"):
        return "ru"
    return "en"


def route(text: str) -> tuple[str, str]:
    """Возвращает (domain, lang).

    domain: 'math', 'code_py', 'code_js', 'error'
    lang: 'kk', 'ru', 'en'
    """
    text_clean = text.strip()

    # Code detection (до lang detect — язык программирования не нужно определять)
    if CODE_PY.match(text_clean):
        return "code_py", "py"
    if CODE_JS.match(text_clean):
        return "code_js", "js"

    # Lang detection
    lang = detect_lang(text_clean)

    # Math detection по ключевым словам
    words = set(text_clean.lower().replace("?", "").split())
    if lang == "kk" and words & MATH_KW_KK:
        return "math", "kk"
    if lang == "ru" and words & MATH_KW_RU:
        return "math", "ru"
    if lang == "en" and words & MATH_KW_EN:
        return "math", "en"

    # Всё остальное — ошибка
    return "error", lang


if __name__ == "__main__":
    tests = [
        "бір қосу екі нешеге тең ?",
        "он алу бес нешеге тең ?",
        "сколько будет два плюс три ?",
        "what is one plus two ?",
        "print(3 + 4)",
        "console.log(5 + 5)",
        "сәлем қалайсың ?",
        "привет как дела ?",
        "hello how are you ?",
    ]
    for t in tests:
        d, l = route(t)
        print(f"  [{d:8s}] [{l:2s}] {t}")
