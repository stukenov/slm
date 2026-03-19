"""
exp_014/run.py — Cascading LLM Pipeline (только инференс)

Qwen2.5-0.5B получает запрос первым:
  - Если уверен и EN → отвечает напрямую
  - Если не EN или не понимает → CONTINUE:LANG_DETECT → перевод → Qwen2.5 повторно
  - Если нужен умный ответ → CONTINUE:THINK → Qwen3-1.7B отвечает

Тест-кейсы: простые вопросы, math, code, мультиязычные, general QA.
"""

import json
import time
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from adapter import HPLTTranslator, MarianTranslator, QwenAdapter
from router import detect_lang

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# System prompt для Qwen2.5-0.5B (primary)
# ============================================================
PRIMARY_SYSTEM = """You are a helpful assistant. Answer the user's question directly if you can.

Rules:
1. If the question is in English and you are confident in your answer, respond normally.
2. If the question is NOT in English and you cannot understand it well, respond with exactly: CONTINUE:LANG_DETECT
3. If the question requires deep reasoning, complex analysis, or you are not confident, respond with exactly: CONTINUE:THINK

Important: Only use CONTINUE: prefix when you truly cannot handle it. Try to answer first."""

# После перевода на EN — более простой промпт
PRIMARY_TRANSLATED_SYSTEM = """You are a helpful assistant. The user's question was translated to English for you. Answer it directly and concisely."""

# ============================================================
# Test cases — разнообразные задачи
# ============================================================
test_cases = [
    # Simple EN (Qwen2.5 должен ответить сам)
    {"input": "What is 2 + 3?", "category": "simple_en", "check": "5"},
    {"input": "What is the capital of France?", "category": "simple_en", "check": "Paris"},
    {"input": "How many days are in a week?", "category": "simple_en", "check": "7"},
    {"input": "What color is the sky?", "category": "simple_en", "check": "blue"},

    # Code (Qwen2.5 должен справиться)
    {"input": "What does print(3 + 4) output in Python?", "category": "code_en", "check": "7"},
    {"input": "What is 10 % 3 in Python?", "category": "code_en", "check": "1"},

    # RU (должен отправить в LANG_DETECT → перевод → ответ)
    {"input": "Какая столица Франции?", "category": "translate_ru", "check": "Пари"},
    {"input": "Сколько дней в неделе?", "category": "translate_ru", "check": "7"},
    {"input": "Какой цвет неба?", "category": "translate_ru", "check": "голуб"},

    # KK (должен отправить в LANG_DETECT → перевод → ответ)
    {"input": "Францияның астанасы қай қала?", "category": "translate_kk", "check": "Пари"},
    {"input": "Аптада неше күн бар?", "category": "translate_kk", "check": "7"},

    # Complex EN (должен эскалировать в THINK → Qwen3-1.7B)
    {"input": "Explain the difference between TCP and UDP protocols in networking.", "category": "think_en",
     "check": "TCP"},
    {"input": "What are the main causes of climate change and how can we address them?",
     "category": "think_en", "check": "carbon"},
    {"input": "Write a Python function that checks if a number is prime.",
     "category": "think_en", "check": "def"},
]


if __name__ == "__main__":
    t0 = time.time()

    # --- Load models ---
    print("Загрузка Qwen2.5-0.5B-Instruct (MLX 4bit)...")
    t1 = time.time()
    qwen_small = QwenAdapter("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print(f"  Qwen2.5-0.5B загружен за {time.time()-t1:.1f}s")

    print("Загрузка Qwen3-1.7B (MLX 4bit)...")
    t1 = time.time()
    qwen_big = QwenAdapter("mlx-community/Qwen3-1.7B-4bit")
    print(f"  Qwen3-1.7B загружен за {time.time()-t1:.1f}s")

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
    print(f"{'='*70}")
    print("CASCADING PIPELINE: Qwen2.5-0.5B → [lang_detect | think] → response")
    print(f"{'='*70}\n")

    results = []
    correct = 0
    paths = {"direct": 0, "lang_detect": 0, "think": 0}

    for tc in test_cases:
        q = tc["input"]
        t_start = time.perf_counter()
        trace = {"input": q, "category": tc["category"], "steps": []}

        # Step 1: Qwen2.5-0.5B
        response = qwen_small.generate_response(PRIMARY_SYSTEM, q, max_tokens=150)
        trace["steps"].append(("Qwen2.5-0.5B", response[:200]))

        final = response
        path = "direct"

        # Check for CONTINUE signals
        first_line = response.strip().split("\n")[0].strip()

        if first_line == "CONTINUE:LANG_DETECT":
            path = "lang_detect"
            # Detect language
            lang = detect_lang(q)
            trace["steps"].append(("lang_detect", lang))

            # Translate to EN
            if lang == "kk":
                en_q = hplt_kk2en.translate(q)
                trace["steps"].append(("HPLT_kk2en", en_q))
            elif lang == "ru":
                en_q = marian_ru2en.translate(q)
                trace["steps"].append(("Marian_ru2en", en_q))
            else:
                en_q = q

            # Re-ask Qwen2.5 in English
            en_answer = qwen_small.generate_response(PRIMARY_TRANSLATED_SYSTEM, en_q, max_tokens=150)
            trace["steps"].append(("Qwen2.5-0.5B (EN)", en_answer[:200]))

            # Translate back
            if lang == "kk":
                final = hplt_en2kk.translate(en_answer)
                trace["steps"].append(("HPLT_en2kk", final[:200]))
            elif lang == "ru":
                final = marian_en2ru.translate(en_answer)
                trace["steps"].append(("Marian_en2ru", final[:200]))
            else:
                final = en_answer

        elif first_line == "CONTINUE:THINK":
            path = "think"
            # Escalate to Qwen3-1.7B
            final = qwen_big.generate_response(
                "You are a knowledgeable assistant. Give a thorough but concise answer.",
                q, max_tokens=200
            )
            trace["steps"].append(("Qwen3-1.7B", final[:300]))

        elapsed = time.perf_counter() - t_start
        trace["path"] = path
        trace["final"] = final[:500]
        trace["time_ms"] = round(elapsed * 1000)
        paths[path] += 1

        # Check answer
        check = tc["check"].lower()
        is_ok = check in final.lower()
        if is_ok:
            correct += 1
        trace["correct"] = is_ok
        results.append(trace)

        # Print
        mark = "OK" if is_ok else "FAIL"
        path_icon = {"direct": "→", "lang_detect": "→🌐→", "think": "→🧠→"}[path]
        print(f"  [{mark}] [{path:12s}] [{elapsed*1000:6.0f}ms] {q[:50]}")
        for step_name, step_val in trace["steps"]:
            val_short = str(step_val)[:80].replace("\n", " ")
            print(f"       {step_name}: {val_short}")
        if not is_ok:
            print(f"       expected to contain: {tc['check']}")
        print()

    total_time = time.time() - t0
    n = len(test_cases)

    print(f"{'='*70}")
    print(f"Answer точность: {correct}/{n} ({100*correct/n:.0f}%)")
    print(f"Пути: direct={paths['direct']}, lang_detect={paths['lang_detect']}, think={paths['think']}")
    print(f"Время: {total_time:.1f}s (загрузка: {load_time:.1f}s)")

    report = {
        "experiment": "exp_014",
        "description": "Cascading: Qwen2.5-0.5B → lang_detect/think → Qwen3-1.7B",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "primary": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "escalation": "mlx-community/Qwen3-1.7B-4bit",
            "kk_en": "HPLT CTranslate2 float32",
            "ru_en": "Helsinki-NLP/opus-mt-ru-en",
        },
        "paths": paths,
        "inference": {
            "accuracy": correct / n, "correct": correct, "total": n,
            "results": results,
        },
        "total_time_s": round(total_time, 1),
        "load_time_s": round(load_time, 1),
    }
    with open(LOG_DIR / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {LOG_DIR / 'report.json'}")
