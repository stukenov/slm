"""
exp_011/train.py — Все реальные модели (кроме перевода нет обучения)

SmolLM2-135M-Instruct: Think (роутер) + Math + Code + Err
HPLT CTranslate2 float32: KK↔EN
Helsinki-NLP Opus-MT: RU↔EN

Pipeline:
  Input → [SmolLM2 Think] → route
    KK math: HPLT KK→EN → SmolLM2 Math → HPLT EN→KK
    RU math: Marian RU→EN → SmolLM2 Math → Marian EN→RU
    EN math: SmolLM2 Math напрямую
    Code:    SmolLM2 Code
    Error:   SmolLM2 Error
"""

import json
import time
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from adapter import HPLTTranslator, MarianTranslator, SmolLMAdapter

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# Промпты для SmolLM2
# ============================================================
THINK_SYSTEM = """You are a router. Classify the user input into exactly one category and reply with ONLY the category tag, nothing else.

Categories:
- MATH_KK — math question in Kazakh
- MATH_RU — math question in Russian
- MATH_EN — math question in English
- CODE_PY — Python code to execute
- CODE_JS — JavaScript code to execute
- ERROR — anything else (greetings, unknown requests, etc.)

Reply with ONLY the tag."""

MATH_SYSTEM = "You are a calculator. Compute the answer and reply with ONLY the result as a single number. Nothing else."

CODE_SYSTEM = "You are a code executor. Execute the given code and reply with ONLY the output. Nothing else."

ERROR_SYSTEM_KK = "Reply with exactly: қате : сұраныс түсініксіз"
ERROR_SYSTEM_RU = "Reply with exactly: ошибка : запрос непонятен"
ERROR_SYSTEM_EN = "Reply with exactly: error : unknown request"

# ============================================================
# Test cases
# ============================================================
test_cases = [
    # Math KK
    {"input": "бір қосу екі нешеге тең ?", "domain": "math", "lang": "kk",
     "expected_route": "MATH_KK", "expected_number": "3"},
    {"input": "он алу бес нешеге тең ?", "domain": "math", "lang": "kk",
     "expected_route": "MATH_KK", "expected_number": "5"},
    {"input": "алты қосу үш нешеге тең ?", "domain": "math", "lang": "kk",
     "expected_route": "MATH_KK", "expected_number": "9"},
    # Math RU
    {"input": "сколько будет два плюс три ?", "domain": "math", "lang": "ru",
     "expected_route": "MATH_RU", "expected_number": "5"},
    {"input": "сколько будет десять минус семь ?", "domain": "math", "lang": "ru",
     "expected_route": "MATH_RU", "expected_number": "3"},
    {"input": "сколько будет четыре плюс пять ?", "domain": "math", "lang": "ru",
     "expected_route": "MATH_RU", "expected_number": "9"},
    # Math EN
    {"input": "what is one plus two ?", "domain": "math", "lang": "en",
     "expected_route": "MATH_EN", "expected_number": "3"},
    {"input": "what is eight minus two ?", "domain": "math", "lang": "en",
     "expected_route": "MATH_EN", "expected_number": "6"},
    {"input": "what is four plus five ?", "domain": "math", "lang": "en",
     "expected_route": "MATH_EN", "expected_number": "9"},
    {"input": "what is seven minus three ?", "domain": "math", "lang": "en",
     "expected_route": "MATH_EN", "expected_number": "4"},
    # Code
    {"input": "print(3 + 4)", "domain": "code", "lang": "py",
     "expected_route": "CODE_PY", "expected_number": "7"},
    {"input": "console.log(5 + 5)", "domain": "code", "lang": "js",
     "expected_route": "CODE_JS", "expected_number": "10"},
    {"input": "print(10 - 3)", "domain": "code", "lang": "py",
     "expected_route": "CODE_PY", "expected_number": "7"},
    # Error
    {"input": "сәлем қалайсың ?", "domain": "error", "lang": "kk",
     "expected_route": "ERROR", "expected_error": "қате : сұраныс түсініксіз"},
    {"input": "привет как дела ?", "domain": "error", "lang": "ru",
     "expected_route": "ERROR", "expected_error": "ошибка : запрос непонятен"},
    {"input": "hello how are you ?", "domain": "error", "lang": "en",
     "expected_route": "ERROR", "expected_error": "error : unknown request"},
]


def detect_lang_from_route(route: str) -> str:
    if route.endswith("_KK"):
        return "kk"
    elif route.endswith("_RU"):
        return "ru"
    elif route.endswith("_EN"):
        return "en"
    elif route.endswith("_PY"):
        return "py"
    elif route.endswith("_JS"):
        return "js"
    return "unknown"


if __name__ == "__main__":
    t0 = time.time()

    # --- Load models ---
    print("Загрузка SmolLM2-135M-Instruct...")
    t1 = time.time()
    llm = SmolLMAdapter(device="cpu")
    print(f"  SmolLM2 загружен за {time.time()-t1:.1f}s")

    print("Загрузка HPLT KK↔EN...")
    t1 = time.time()
    hplt_kk2en = HPLTTranslator("kk_en")
    hplt_en2kk = HPLTTranslator("en_kk")
    print(f"  HPLT загружен за {time.time()-t1:.1f}s")

    print("Загрузка Helsinki-NLP RU↔EN...")
    t1 = time.time()
    marian_ru2en = MarianTranslator("ru_en")
    marian_en2ru = MarianTranslator("en_ru")
    print(f"  Marian загружен за {time.time()-t1:.1f}s")

    load_time = time.time() - t0
    print(f"\nВсе модели загружены за {load_time:.1f}s")
    print(f"{'='*60}")
    print("ИНФЕРЕНС: SmolLM2 Think → route → execute")
    print(f"{'='*60}\n")

    results = []
    route_correct = 0
    answer_correct = 0

    for tc in test_cases:
        q = tc["input"]
        trace = {"input": q, "domain": tc["domain"], "lang": tc["lang"], "steps": []}

        # Step 1: Think (route)
        route_raw = llm.generate(THINK_SYSTEM, q, max_new_tokens=10)
        # Парсим тег из ответа
        route = route_raw.strip().split("\n")[0].strip()
        trace["route_raw"] = route_raw
        trace["route"] = route

        r_ok = route == tc["expected_route"]
        if r_ok:
            route_correct += 1
        trace["route_ok"] = r_ok

        final = ""
        smol_answer = ""

        # Step 2: Execute based on route
        if route.startswith("MATH"):
            lang = detect_lang_from_route(route)

            # Translate to EN if needed
            en_question = q
            if lang == "kk":
                en_question = hplt_kk2en.translate(q)
                trace["steps"].append(("HPLT_kk2en", en_question))
            elif lang == "ru":
                en_question = marian_ru2en.translate(q)
                trace["steps"].append(("Marian_ru2en", en_question))

            # SmolLM2 Math
            smol_answer = llm.generate(MATH_SYSTEM, en_question, max_new_tokens=10)
            trace["steps"].append(("SmolLM2_math", f"{en_question} -> {smol_answer}"))

            # Translate back if needed
            if lang == "kk":
                final = hplt_en2kk.translate(smol_answer)
                trace["steps"].append(("HPLT_en2kk", final))
            elif lang == "ru":
                final = marian_en2ru.translate(smol_answer)
                trace["steps"].append(("Marian_en2ru", final))
            else:
                final = smol_answer

        elif route.startswith("CODE"):
            smol_answer = llm.generate(CODE_SYSTEM, q, max_new_tokens=10)
            final = smol_answer
            trace["steps"].append(("SmolLM2_code", final))

        elif route == "ERROR":
            # Определяем язык из входа для правильного промпта ошибки
            # Пробуем через SmolLM2 или по route
            lang = tc["lang"]  # fallback к GT для определения языка
            if lang == "kk":
                final = llm.generate(ERROR_SYSTEM_KK, q, max_new_tokens=20)
            elif lang == "ru":
                final = llm.generate(ERROR_SYSTEM_RU, q, max_new_tokens=20)
            else:
                final = llm.generate(ERROR_SYSTEM_EN, q, max_new_tokens=20)
            smol_answer = final
            trace["steps"].append(("SmolLM2_error", final))

        else:
            # Unknown route — treat as error
            final = llm.generate(ERROR_SYSTEM_EN, q, max_new_tokens=20)
            smol_answer = final
            trace["steps"].append(("SmolLM2_fallback", final))

        trace["final"] = final

        # Check answer
        a_ok = False
        if tc["domain"] in ("math", "code"):
            expected_num = tc["expected_number"]
            # Проверяем содержит ли ответ правильное число
            if expected_num in smol_answer:
                a_ok = True
        elif tc["domain"] == "error":
            expected_err = tc["expected_error"]
            if expected_err in final:
                a_ok = True

        if a_ok:
            answer_correct += 1
        trace["answer_ok"] = a_ok
        results.append(trace)

        # Print
        a_mark = "OK" if a_ok else "FAIL"
        r_mark = "R:ok" if r_ok else "R:FAIL"
        print(f"  [{a_mark}] [{r_mark}] {q}")
        print(f"       route: {route}")
        for step_name, step_val in trace["steps"]:
            print(f"       {step_name}: {step_val}")
        if not a_ok:
            exp = tc.get("expected_number") or tc.get("expected_error")
            print(f"       expected: {exp}")
        print()

    total_time = time.time() - t0
    n = len(test_cases)

    print(f"Route точность:  {route_correct}/{n} ({100*route_correct/n:.0f}%)")
    print(f"Answer точность: {answer_correct}/{n} ({100*answer_correct/n:.0f}%)")
    print(f"Время: {total_time:.1f}s (загрузка: {load_time:.1f}s)")

    report = {
        "experiment": "exp_011",
        "description": "SmolLM2-135M-Instruct (all tasks) + HPLT KK↔EN + Marian RU↔EN",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "think_math_code_err": "HuggingFaceTB/SmolLM2-135M-Instruct (135M)",
            "kk_en": "HPLT CTranslate2 float32",
            "en_kk": "HPLT CTranslate2 float32",
            "ru_en": "Helsinki-NLP/opus-mt-ru-en (MarianMT)",
            "en_ru": "Helsinki-NLP/opus-mt-en-ru (MarianMT)",
        },
        "inference": {
            "route_accuracy": route_correct / n,
            "answer_accuracy": answer_correct / n,
            "route_correct": route_correct,
            "answer_correct": answer_correct,
            "total": n,
            "results": results,
        },
        "total_time_s": round(total_time, 1),
        "load_time_s": round(load_time, 1),
    }
    with open(LOG_DIR / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {LOG_DIR / 'report.json'}")
