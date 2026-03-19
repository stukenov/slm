#!/usr/bin/env python3
"""Test different generation params for Kazakh/Russian quality"""
import json, urllib.request, time

URL = "http://localhost:15127/v1/chat/completions"

def ask(prompt, system=None, temp=0.7, top_p=0.95, top_k=40, min_p=0.05,
        repeat_penalty=1.0, label=""):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    data = json.dumps({
        "model": "GPT-OSS-120B",
        "messages": msgs,
        "max_tokens": 400,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
    }).encode()
    req = urllib.request.Request(URL, data=data,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=180)
    result = json.loads(resp.read())
    elapsed = time.time() - t0
    ch = result["choices"][0]["message"]
    tg = result["timings"]["predicted_per_second"]
    content = ch["content"]
    reasoning = ch.get("reasoning_content", "")
    print(f"\n{'='*70}")
    print(f"[{label}] temp={temp} top_p={top_p} top_k={top_k} min_p={min_p} rep={repeat_penalty}")
    print(f"TG: {tg:.1f} t/s | Time: {elapsed:.1f}s")
    if system:
        print(f"System: {system[:80]}...")
    print(f"Q: {prompt}")
    print(f"-"*70)
    print(content[:600])
    print(f"{'='*70}")
    return content

# ============================================================
# KAZAKH LANGUAGE TESTS
# ============================================================
print("\n" + "#"*70)
print("# KAZAKH LANGUAGE PARAMETER TUNING")
print("#"*70)

KK_PROMPT = "Abai Qunanbaev kim? Onyn shygarmashylygy turaly qysqasha aityp ber."

# --- System prompt variants ---
print("\n>>> SYSTEM PROMPT VARIANTS (Kazakh)")

SYS_V1 = "You are a helpful assistant. Always respond in Kazakh language using Cyrillic script."
SYS_V2 = "Сен көмекші ботсың. Барлық жауаптарыңды тек қазақ тілінде, кирилл жазуымен бер. Жауаптарың нақты, сауатты және мәдени тұрғыдан дұрыс болсын."
SYS_V3 = "Ты — помощник, отвечающий исключительно на казахском языке кириллицей. Пиши грамотно, развёрнуто, используй правильную казахскую грамматику и орфографию."
SYS_V4 = "Sen qazaq tilinde jauap beretyn komekcysin. Tek qazaq tilinde, kirill jazyuymyn jauap ber. Gramatiqa men orfografiyany saqta."

ask(KK_PROMPT, SYS_V1, label="EN system prompt")
ask(KK_PROMPT, SYS_V2, label="KK system prompt")
ask(KK_PROMPT, SYS_V3, label="RU system prompt")
ask(KK_PROMPT, SYS_V4, label="KK-latin system prompt")

# --- Temperature ---
print("\n>>> TEMPERATURE VARIANTS (Kazakh)")
for t in [0.3, 0.5, 0.7, 1.0]:
    ask(KK_PROMPT, SYS_V2, temp=t, label=f"temp={t}")

# --- Top-p ---
print("\n>>> TOP_P VARIANTS (Kazakh)")
for tp in [0.8, 0.9, 0.95, 1.0]:
    ask(KK_PROMPT, SYS_V2, top_p=tp, label=f"top_p={tp}")

# --- Min-p ---
print("\n>>> MIN_P VARIANTS (Kazakh)")
for mp in [0.0, 0.02, 0.05, 0.1]:
    ask(KK_PROMPT, SYS_V2, min_p=mp, label=f"min_p={mp}")

# --- Repeat penalty ---
print("\n>>> REPEAT PENALTY (Kazakh)")
for rp in [1.0, 1.05, 1.1, 1.15]:
    ask(KK_PROMPT, SYS_V2, repeat_penalty=rp, label=f"rep={rp}")

# ============================================================
# RUSSIAN LANGUAGE TESTS
# ============================================================
print("\n" + "#"*70)
print("# RUSSIAN LANGUAGE PARAMETER TUNING")
print("#"*70)

RU_PROMPT = "Расскажи подробно о Шёлковом пути и его значении для Казахстана."

SYS_RU1 = "You are a helpful assistant. Always respond in Russian."
SYS_RU2 = "Ты — умный и полезный помощник. Отвечай всегда на русском языке. Пиши грамотно, развёрнуто и структурированно."
SYS_RU3 = "Ты — эксперт по истории Центральной Азии. Отвечай на русском языке, грамотно, с фактами и датами."

print("\n>>> SYSTEM PROMPT VARIANTS (Russian)")
ask(RU_PROMPT, SYS_RU1, label="RU simple system")
ask(RU_PROMPT, SYS_RU2, label="RU detailed system")
ask(RU_PROMPT, SYS_RU3, label="RU expert system")

# --- Temperature ---
print("\n>>> TEMPERATURE VARIANTS (Russian)")
for t in [0.3, 0.5, 0.7, 1.0]:
    ask(RU_PROMPT, SYS_RU2, temp=t, label=f"temp={t}")

# --- Best combo test ---
print("\n" + "#"*70)
print("# FINAL COMPARISON: DEFAULT vs OPTIMIZED")
print("#"*70)

# Default
ask("Қазіргі Қазақстандағы білім беру жүйесі туралы не айта аласың?",
    None, temp=0.7, label="KK DEFAULT (no system)")

# Optimized (will fill in best params after seeing results)
ask("Қазіргі Қазақстандағы білім беру жүйесі туралы не айта аласың?",
    SYS_V2, temp=0.5, top_p=0.9, min_p=0.05, repeat_penalty=1.1,
    label="KK OPTIMIZED")

ask("Объясни, почему Казахстан важен в мировой геополитике.",
    None, temp=0.7, label="RU DEFAULT (no system)")

ask("Объясни, почему Казахстан важен в мировой геополитике.",
    SYS_RU2, temp=0.5, top_p=0.9, min_p=0.05, repeat_penalty=1.1,
    label="RU OPTIMIZED")

print("\n\nDONE!")
