"""
exp_015/run.py — Cascading LLM + NLLB universal translator

Architecture (from exp_014 product review):
  1. Charset detector (0ms) — if non-Latin → LANG_DETECT path
  2. NLLB-200 translates any language → EN
  3. Qwen2.5-0.5B answers in EN
  4. Self-check: Qwen2.5 verifies own answer
  5. If not confident → escalate to Qwen3-1.7B
  6. NLLB translates answer back to original language

Pipeline:
  Запрос → [charset check]
    → if non-EN: [NLLB → EN] → [Qwen2.5-0.5B] → [self-check] → [NLLB ← EN] → ответ
    → if EN: [Qwen2.5-0.5B] → [self-check]
      → if confident → ответ
      → if not → [Qwen3-1.7B] → ответ
"""

import json
import time
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from adapter import NLLBTranslator, HPLTTranslator, QwenAdapter
from router import detect_lang

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# System prompts
# ============================================================
PRIMARY_SYSTEM = """You are a helpful assistant. Answer the user's question directly and concisely.
If the question requires deep reasoning, complex analysis, coding, or you are not confident in your answer, respond with exactly: ESCALATE
Otherwise, give a direct answer."""

SELFCHECK_SYSTEM = """You are a fact-checker. The user will show you a question and an answer.
Reply with exactly YES if the answer is correct, or NO if it might be wrong.
Only reply YES or NO, nothing else."""

ESCALATION_SYSTEM = """You are a knowledgeable assistant. Give a thorough but concise answer."""

# ============================================================
# Test cases
# ============================================================
test_cases = [
    # === EN — simple ===
    {"input": "What is 2 + 3?", "lang": "en", "check": "5"},
    {"input": "What is the capital of France?", "lang": "en", "check": "Paris"},
    {"input": "How many days are in a week?", "lang": "en", "check": "7"},
    {"input": "What color is the sky?", "lang": "en", "check": "blue"},
    {"input": "Who wrote Romeo and Juliet?", "lang": "en", "check": "Shakespeare"},
    {"input": "What is the largest planet in our solar system?", "lang": "en", "check": "Jupiter"},
    {"input": "What does print(3 + 4) output in Python?", "lang": "en", "check": "7"},
    {"input": "What is the boiling point of water in Celsius?", "lang": "en", "check": "100"},
    {"input": "How many legs does a spider have?", "lang": "en", "check": "8"},

    # === RU — русский ===
    {"input": "Какая столица Франции?", "lang": "ru", "check": "Париж"},
    {"input": "Сколько дней в неделе?", "lang": "ru", "check": "7"},
    {"input": "Кто написал «Война и мир»?", "lang": "ru", "check": "Толстой"},
    {"input": "Сколько планет в солнечной системе?", "lang": "ru", "check": "8"},
    {"input": "Какой химический символ у воды?", "lang": "ru", "check": "H2O"},
    {"input": "В каком году человек впервые полетел в космос?", "lang": "ru", "check": "1961"},
    {"input": "Сколько часов в сутках?", "lang": "ru", "check": "24"},

    # === KK — қазақша ===
    {"input": "Францияның астанасы қай қала?", "lang": "kk", "check": "Париж"},
    {"input": "Аптада неше күн бар?", "lang": "kk", "check": "7"},
    {"input": "Қазақстанның астанасы қай қала?", "lang": "kk", "check": "Астана"},
    {"input": "Жердегі ең биік тау қайсы?", "lang": "kk", "check": "Эверест"},
    {"input": "Бір жылда неше ай бар?", "lang": "kk", "check": "12"},
    {"input": "Судың химиялық формуласы қандай?", "lang": "kk", "check": "H2O"},
    {"input": "Тәулікте неше сағат бар?", "lang": "kk", "check": "24"},
    {"input": "Абай Құнанбаев кім?", "lang": "kk", "check": "поэт"},

    # === Complex EN (may escalate) ===
    {"input": "Explain the difference between TCP and UDP protocols.", "lang": "en", "check": "TCP"},
    {"input": "Write a Python function that checks if a number is prime.", "lang": "en", "check": "def"},
]


if __name__ == "__main__":
    t0 = time.time()

    # --- Load models ---
    print("Загрузка HPLT KK↔EN (float32)...")
    t1 = time.time()
    hplt_kk2en = HPLTTranslator("kk_en")
    hplt_en2kk = HPLTTranslator("en_kk")
    print(f"  HPLT загружен за {time.time()-t1:.1f}s")

    print("Загрузка NLLB-200-distilled-600M (CT2, для не-KK)...")
    t1 = time.time()
    nllb = NLLBTranslator()
    print(f"  NLLB загружен за {time.time()-t1:.1f}s")

    print("Загрузка Qwen2.5-0.5B-Instruct (MLX 4bit)...")
    t1 = time.time()
    qwen_small = QwenAdapter("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print(f"  Qwen2.5-0.5B загружен за {time.time()-t1:.1f}s")

    print("Загрузка Qwen3-1.7B (MLX 4bit)...")
    t1 = time.time()
    qwen_big = QwenAdapter("mlx-community/Qwen3-1.7B-4bit")
    print(f"  Qwen3-1.7B загружен за {time.time()-t1:.1f}s")

    load_time = time.time() - t0
    print(f"\nВсе модели загружены за {load_time:.1f}s")
    print(f"{'='*70}")
    print("CASCADING: charset→[HPLT|NLLB]→Qwen2.5→self-check→[Qwen3]→[HPLT|NLLB] back")
    print(f"{'='*70}\n")

    results = []
    correct = 0
    paths = {"direct": 0, "translate": 0, "escalate": 0, "translate+escalate": 0}

    for tc in test_cases:
        q = tc["input"]
        t_start = time.perf_counter()
        trace = {"input": q, "lang": tc["lang"], "steps": []}

        # Step 1: Charset-based language detection
        lang = detect_lang(q)
        trace["detected_lang"] = lang
        need_translate = (lang != "en")

        # Step 2: Translate to EN if needed (KK→HPLT, rest→NLLB)
        en_question = q
        if need_translate:
            if lang == "kk":
                en_question = hplt_kk2en.translate(q)
                trace["steps"].append(("HPLT_kk2en", f"{q} → {en_question}"))
            else:
                en_question = nllb.translate(q, src_lang=lang, tgt_lang="en")
                trace["steps"].append(("NLLB_to_en", f"{q} → {en_question}"))

        # Step 3: Qwen2.5-0.5B answers
        response = qwen_small.generate_response(PRIMARY_SYSTEM, en_question, max_tokens=150)
        trace["steps"].append(("Qwen2.5-0.5B", response[:200]))

        escalated = False
        first_line = response.strip().split("\n")[0].strip()

        if first_line == "ESCALATE":
            # Escalate to Qwen3-1.7B
            escalated = True
            response = qwen_big.generate_response(ESCALATION_SYSTEM, en_question, max_tokens=200)
            trace["steps"].append(("Qwen3-1.7B", response[:300]))
        else:
            # Self-check
            check_prompt = f"Question: {en_question}\nAnswer: {response}\nIs this answer correct?"
            check = qwen_small.generate_response(SELFCHECK_SYSTEM, check_prompt, max_tokens=5)
            trace["steps"].append(("self-check", check))

            if "NO" in check.upper():
                escalated = True
                response = qwen_big.generate_response(ESCALATION_SYSTEM, en_question, max_tokens=200)
                trace["steps"].append(("Qwen3-1.7B (after NO)", response[:300]))

        # Step 4: Translate back if needed (KK→HPLT, rest→NLLB)
        final = response
        if need_translate:
            if lang == "kk":
                final = hplt_en2kk.translate(response)
                trace["steps"].append(("HPLT_en2kk", final[:200]))
            else:
                final = nllb.translate(response, src_lang="en", tgt_lang=lang)
                trace["steps"].append(("NLLB_back", final[:200]))

        elapsed = time.perf_counter() - t_start

        # Determine path
        if need_translate and escalated:
            path = "translate+escalate"
        elif need_translate:
            path = "translate"
        elif escalated:
            path = "escalate"
        else:
            path = "direct"
        paths[path] += 1

        trace["path"] = path
        trace["final"] = final[:500]
        trace["time_ms"] = round(elapsed * 1000)

        # Check answer
        check_str = tc["check"].lower()
        is_ok = check_str in final.lower()
        # Also check in the EN response (before translation back)
        if not is_ok:
            is_ok = check_str in response.lower()
        if is_ok:
            correct += 1
        trace["correct"] = is_ok
        results.append(trace)

        # Print
        mark = "OK" if is_ok else "FAIL"
        print(f"  [{mark}] [{path:20s}] [{elapsed*1000:6.0f}ms] {q[:55]}")
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
    print(f"Пути: {paths}")
    print(f"Время: {total_time:.1f}s (загрузка: {load_time:.1f}s)")

    report = {
        "experiment": "exp_015",
        "description": "Cascading Qwen2.5→Qwen3 + hybrid translator (HPLT KK, NLLB rest) + self-check",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "primary": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "escalation": "mlx-community/Qwen3-1.7B-4bit",
            "translator_kk": "HPLT CTranslate2 float32 (KK↔EN)",
            "translator_other": "facebook/nllb-200-distilled-600M CT2 (200+ languages)",
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
