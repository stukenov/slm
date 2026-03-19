# exp_015 — Cascading LLM + Hybrid Translator (HPLT KK + NLLB rest)

## Что привело к этому
- exp_012 (38%): langdetect + SmolLM2-360M — слишком слабая модель
- exp_013 (75%): langdetect + SmolLM2-1.7B MLX — лучше, но SmolLM2 хуже Qwen
- exp_014 (79%): Cascading Qwen2.5-0.5B → Qwen3-1.7B + HPLT/Marian — каскад работает, но отдельные переводчики для каждой пары
- exp_015: заменили отдельные переводчики на NLLB-200 (универсальный), но KK качество упало → гибрид HPLT(KK) + NLLB(rest)

## Архитектура

```
Запрос → [charset detector (0ms)]
  ├─ KK: [HPLT KK→EN] → [Qwen2.5-0.5B] → [self-check] → [HPLT EN→KK] → ответ
  ├─ RU: [NLLB RU→EN] → [Qwen2.5-0.5B] → [self-check] → [NLLB EN→RU] → ответ
  └─ EN: [Qwen2.5-0.5B] → [self-check]
           ├─ confident → ответ
           └─ not confident / ESCALATE → [Qwen3-1.7B] → ответ
```

| Компонент | Модель | Формат |
|-----------|--------|--------|
| Charset detector | regex (қңғүұіөәһ/кириллица) | — |
| KK↔EN перевод | HPLT CTranslate2 | float32 |
| Остальные языки | NLLB-200-distilled-600M CTranslate2 | float32 |
| Primary LLM | Qwen2.5-0.5B-Instruct | MLX 4bit |
| Escalation LLM | Qwen3-1.7B | MLX 4bit |

## Результаты

**Точность: 24/26 (92%)**

### По языкам

| Язык | Результат | Переводчик | Скорость |
|------|-----------|------------|----------|
| EN (9) | 9/9 (100%) | — | 300-500ms |
| RU (7) | 6/7 (86%) | NLLB CT2 | 3-14s |
| KK (8) | 7/8 (88%) | HPLT float32 | 700-1400ms |
| Complex EN (2) | 2/2 (100%) | — | 700-1500ms |

### Ошибки
- "Сколько часов в сутках?" — NLLB перевёл "How many hours a day?" → Qwen запутался
- "Абай Құнанбаев кім?" — HPLT перевёл "Who is Abysmal Bhagavan?" → не знает имя

### Скорость
- Загрузка: ~9s (HPLT 0.3s, NLLB 5s, Qwen2.5 1s, Qwen3 2s)
- EN direct: 300-500ms
- KK translate: 700-1400ms
- RU translate: 3-14s (NLLB медленнее HPLT)
- RAM: ~2.5GB

## Ключевые решения
1. **HPLT для KK** — качество перевода выше чем NLLB (92% vs 79% с NLLB-only)
2. **NLLB для остальных** — универсальный, 200+ языков, один переводчик
3. **CTranslate2** — 60x быстрее transformers для NLLB
4. **Self-check** — по факту бесполезен, всегда YES. Нужен лучший механизм
5. **Qwen2.5-0.5B** — не знает Астану Казахстана (отвечает Алматы), нужна модель побольше

## Сравнение экспериментов

| Exp | Accuracy | Architecture |
|-----|----------|-------------|
| exp_008 | 93% | tinygrad 125K × 6 (toy, узкие задачи) |
| exp_012 | 38% | langdetect + SmolLM2-360M |
| exp_013 | 75% | langdetect + SmolLM2-1.7B MLX |
| exp_014 | 79% | Cascading Qwen2.5→Qwen3 + HPLT + Marian |
| **exp_015** | **92%** | **Cascading Qwen2.5→Qwen3 + HPLT(KK) + NLLB(rest)** |

## Запуск
```bash
cd nano && .venv/bin/python exp/exp_015/run.py
```
