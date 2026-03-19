"""
exp_013/bench.py — Замеры скорости каждого компонента
"""

import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

def bench(name, fn, n=5):
    """Прогоняет fn n раз, возвращает среднее время."""
    # Warmup
    fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    print(f"  {name:30s}  avg={avg*1000:7.1f}ms  min={mn*1000:.1f}ms  max={mx*1000:.1f}ms  (n={n})")
    return avg

print("=" * 70)
print("ЗАГРУЗКА МОДЕЛЕЙ")
print("=" * 70)

# SmolLM2 MLX
from adapter import SmolLMAdapter
t = time.perf_counter()
llm = SmolLMAdapter()
print(f"  SmolLM2-1.7B MLX:  {time.perf_counter()-t:.1f}s")

# HPLT
from adapter import HPLTTranslator
t = time.perf_counter()
hplt_kk2en = HPLTTranslator("kk_en")
hplt_en2kk = HPLTTranslator("en_kk")
print(f"  HPLT KK↔EN:        {time.perf_counter()-t:.1f}s")

# Marian
from adapter import MarianTranslator
t = time.perf_counter()
marian_ru2en = MarianTranslator("ru_en")
marian_en2ru = MarianTranslator("en_ru")
print(f"  Marian RU↔EN:      {time.perf_counter()-t:.1f}s")

# Router
from router import route
t = time.perf_counter()
for _ in range(100):
    route("бір қосу екі нешеге тең ?")
print(f"  Router (100 calls): {time.perf_counter()-t:.3f}s")

print(f"\n{'=' * 70}")
print("ИНФЕРЕНС (5 прогонов, среднее)")
print("=" * 70)

# Router
print("\n--- Router (langdetect + keywords) ---")
bench("route KK math", lambda: route("бір қосу екі нешеге тең ?"))
bench("route RU math", lambda: route("сколько будет два плюс три ?"))
bench("route EN math", lambda: route("what is one plus two ?"))
bench("route code py", lambda: route("print(3 + 4)"))
bench("route error", lambda: route("hello how are you ?"))

# HPLT
print("\n--- HPLT CTranslate2 float32 ---")
bench("HPLT KK→EN short", lambda: hplt_kk2en.translate("бір қосу екі нешеге тең?"))
bench("HPLT EN→KK short", lambda: hplt_en2kk.translate("one plus two is three."))
bench("HPLT KK→EN long", lambda: hplt_kk2en.translate("Астана Қазақстанның астанасы, ол елдің жүрегі."))
bench("HPLT EN→KK long", lambda: hplt_en2kk.translate("Astana is the capital of Kazakhstan, it is the heart of the country."))

# Marian
print("\n--- Helsinki-NLP MarianMT ---")
bench("Marian RU→EN short", lambda: marian_ru2en.translate("сколько будет два плюс три?"))
bench("Marian EN→RU short", lambda: marian_en2ru.translate("two plus three is five."))
bench("Marian RU→EN long", lambda: marian_ru2en.translate("Москва — столица России, один из крупнейших городов мира."))
bench("Marian EN→RU long", lambda: marian_en2ru.translate("Moscow is the capital of Russia, one of the largest cities in the world."))

# SmolLM2 MLX
print("\n--- SmolLM2-1.7B MLX Metal ---")
MATH_SYS = "You are a calculator. Compute the answer and reply with ONLY the result as a single number. Nothing else."
CODE_SYS = "You are a code executor. Execute the given code and reply with ONLY the printed output. Nothing else."
ERR_SYS = "Reply with exactly: error : unknown request"

bench("SmolLM2 math (1+2)", lambda: llm.generate_response(MATH_SYS, "what is one plus two?", max_tokens=10))
bench("SmolLM2 math (10-7)", lambda: llm.generate_response(MATH_SYS, "what is ten minus seven?", max_tokens=10))
bench("SmolLM2 code py", lambda: llm.generate_response(CODE_SYS, "print(3 + 4)", max_tokens=10))
bench("SmolLM2 code js", lambda: llm.generate_response(CODE_SYS, "console.log(5 + 5)", max_tokens=10))
bench("SmolLM2 error", lambda: llm.generate_response(ERR_SYS, "hello how are you?", max_tokens=20))

# Full pipeline
print(f"\n{'=' * 70}")
print("FULL PIPELINE (end-to-end)")
print("=" * 70)

def pipeline_en_math():
    d, l = route("what is one plus two ?")
    return llm.generate_response(MATH_SYS, "what is one plus two ?", max_tokens=10)

def pipeline_ru_math():
    d, l = route("сколько будет два плюс три ?")
    en = marian_ru2en.translate("сколько будет два плюс три ?")
    ans = llm.generate_response(MATH_SYS, en, max_tokens=10)
    return marian_en2ru.translate(ans)

def pipeline_kk_math():
    d, l = route("бір қосу екі нешеге тең ?")
    en = hplt_kk2en.translate("бір қосу екі нешеге тең?")
    ans = llm.generate_response(MATH_SYS, en, max_tokens=10)
    return hplt_en2kk.translate(ans)

def pipeline_code():
    d, l = route("print(3 + 4)")
    return llm.generate_response(CODE_SYS, "print(3 + 4)", max_tokens=10)

def pipeline_error():
    d, l = route("hello how are you ?")
    return llm.generate_response(ERR_SYS, "hello how are you?", max_tokens=20)

bench("pipeline EN math", pipeline_en_math)
bench("pipeline RU math (3 models)", pipeline_ru_math)
bench("pipeline KK math (3 models)", pipeline_kk_math)
bench("pipeline Code", pipeline_code)
bench("pipeline Error", pipeline_error)

print(f"\n{'=' * 70}")
print("ПАМЯТЬ")
print("=" * 70)
import subprocess
pid = os.getpid()
result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
rss_kb = int(result.stdout.strip())
print(f"  RSS: {rss_kb / 1024:.0f} MB")
