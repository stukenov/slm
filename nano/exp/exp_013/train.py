"""
exp_013/train.py — langdetect + SmolLM2-1.7B + HPLT + Marian

Router: langdetect + keywords (rule-based)
SmolLM2-1.7B-Instruct: Math + Code + Err
HPLT CTranslate2 float32: KK↔EN
Helsinki-NLP Opus-MT: RU↔EN

Pipeline:
  Input → [langdetect router] → route
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
from router import route

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# Промпты для SmolLM2
# ============================================================
MATH_SYSTEM = "You are a calculator. Compute the answer and reply with ONLY the result as a single number. Nothing else."

CODE_SYSTEM = "You are a code executor. Execute the given code and reply with ONLY the printed output. Nothing else."

ERROR_SYSTEM_KK = "Reply with exactly: қате : сұраныс түсініксіз"
ERROR_SYSTEM_RU = "Reply with exactly: ошибка : запрос непонятен"
ERROR_SYSTEM_EN = "Reply with exactly: error : unknown request"

ERROR_SYSTEMS = {"kk": ERROR_SYSTEM_KK, "ru": ERROR_SYSTEM_RU, "en": ERROR_SYSTEM_EN}

# ============================================================
# Test cases
# ============================================================
test_cases = [
    # Math KK
    {"input": "бір қосу екі нешеге тең ?", "domain": "math", "lang": "kk", "expected_number": "3"},
    {"input": "он алу бес нешеге тең ?", "domain": "math", "lang": "kk", "expected_number": "5"},
    {"input": "алты қосу үш нешеге тең ?", "domain": "math", "lang": "kk", "expected_number": "9"},
    # Math RU
    {"input": "сколько будет два плюс три ?", "domain": "math", "lang": "ru", "expected_number": "5"},
    {"input": "сколько будет десять минус семь ?", "domain": "math", "lang": "ru", "expected_number": "3"},
    {"input": "сколько будет четыре плюс пять ?", "domain": "math", "lang": "ru", "expected_number": "9"},
    # Math EN
    {"input": "what is one plus two ?", "domain": "math", "lang": "en", "expected_number": "3"},
    {"input": "what is eight minus two ?", "domain": "math", "lang": "en", "expected_number": "6"},
    {"input": "what is four plus five ?", "domain": "math", "lang": "en", "expected_number": "9"},
    {"input": "what is seven minus three ?", "domain": "math", "lang": "en", "expected_number": "4"},
    # Code
    {"input": "print(3 + 4)", "domain": "code", "lang": "py", "expected_number": "7"},
    {"input": "console.log(5 + 5)", "domain": "code", "lang": "js", "expected_number": "10"},
    {"input": "print(10 - 3)", "domain": "code", "lang": "py", "expected_number": "7"},
    # Error
    {"input": "сәлем қалайсың ?", "domain": "error", "lang": "kk",
     "expected_error": "қате : сұраныс түсініксіз"},
    {"input": "привет как дела ?", "domain": "error", "lang": "ru",
     "expected_error": "ошибка : запрос непонятен"},
    {"input": "hello how are you ?", "domain": "error", "lang": "en",
     "expected_error": "error : unknown request"},
]


if __name__ == "__main__":
    t0 = time.time()

    # --- Load models ---
    print("Загрузка SmolLM2-1.7B-Instruct...")
    t1 = time.time()
    llm = SmolLMAdapter()
    print(f"  SmolLM2 MLX загружен за {time.time()-t1:.1f}s")

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
    print("ИНФЕРЕНС: langdetect router → SmolLM2-1.7B tasks")
    print(f"{'='*60}\n")

    results = []
    route_correct = 0
    answer_correct = 0

    for tc in test_cases:
        q = tc["input"]
        trace = {"input": q, "domain": tc["domain"], "lang": tc["lang"], "steps": []}

        # Step 1: Route
        domain, lang = route(q)
        trace["route_domain"] = domain
        trace["route_lang"] = lang

        # Check route
        expected_domain = tc["domain"]
        if expected_domain == "code":
            expected_domain = f"code_{tc['lang']}"
        r_ok = domain == expected_domain and lang == tc["lang"]
        if r_ok:
            route_correct += 1
        trace["route_ok"] = r_ok

        final = ""
        smol_answer = ""

        # Step 2: Execute
        if domain.startswith("math"):
            # Translate to EN if needed
            en_question = q
            if lang == "kk":
                en_question = hplt_kk2en.translate(q)
                trace["steps"].append(("HPLT_kk2en", en_question))
            elif lang == "ru":
                en_question = marian_ru2en.translate(q)
                trace["steps"].append(("Marian_ru2en", en_question))

            # SmolLM2 Math
            smol_answer = llm.generate_response(MATH_SYSTEM, en_question, max_tokens=10)
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

        elif domain.startswith("code"):
            smol_answer = llm.generate_response(CODE_SYSTEM, q, max_tokens=10)
            final = smol_answer
            trace["steps"].append(("SmolLM2_code", final))

        elif domain == "error":
            err_system = ERROR_SYSTEMS.get(lang, ERROR_SYSTEM_EN)
            smol_answer = llm.generate_response(err_system, q, max_tokens=20)
            final = smol_answer
            trace["steps"].append(("SmolLM2_error", final))

        trace["final"] = final

        # Check answer
        a_ok = False
        if tc["domain"] in ("math", "code"):
            expected_num = tc["expected_number"]
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
        print(f"       route: {domain}/{lang}")
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
        "experiment": "exp_013",
        "description": "langdetect router + SmolLM2-1.7B + HPLT KK↔EN + Marian RU↔EN",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "router": "langdetect + keywords (rule-based)",
            "tasks": "mlx-community/SmolLM2-1.7B-Instruct-4bit (MLX Metal)",
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
