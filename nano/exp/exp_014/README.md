# exp_014 — Cascading LLM: Qwen2.5-0.5B → Qwen3-1.7B

## Что просили
Без роутера. Qwen2.5-0.5B принимает запрос первой. Если уверена — отвечает. Если не на EN → CONTINUE:LANG_DETECT → перевод → повтор. Если нужен умный ответ → CONTINUE:THINK → Qwen3-1.7B.

## Архитектура

```
Запрос → [Qwen2.5-0.5B]
  ├─ уверен → ответ (206-1385ms)
  ├─ CONTINUE:LANG_DETECT → langdetect → перевод → [Qwen2.5-0.5B] → перевод → ответ (427-2370ms)
  └─ CONTINUE:THINK → [Qwen3-1.7B] → ответ (4351-5137ms)
```

| Компонент | Модель | Формат |
|-----------|--------|--------|
| Primary | Qwen2.5-0.5B-Instruct | MLX 4bit |
| Escalation | Qwen3-1.7B | MLX 4bit |
| KK↔EN | HPLT CTranslate2 float32 | — |
| RU↔EN | Helsinki-NLP Opus-MT | — |

## Тест-кейсы
14 разнообразных: simple EN, code, RU перевод, KK перевод, complex EN (TCP/UDP, climate, prime).

## Результаты

**Answer: 11/14 (79%)**

| Путь | Кол-во | Примеры |
|------|--------|---------|
| direct | 7 | code, complex EN, 1 RU |
| lang_detect | 4 | RU вопросы, 1 EN |
| think | 3 | KK вопросы, 1 EN |

### Ошибки
- "How many days in a week?" → "14 days" (Qwen2.5 ошиблась)
- "Какой цвет неба?" → "Белый" (не эскалировала, ответила сама)
- KK "Аптада неше күн бар?" → Qwen3 рассуждает но не выдаёт "7"

### Скорость
- Загрузка: 96с (Qwen2.5: 17с, Qwen3: 73с, HPLT: 0.4с, Marian: 6с)
- direct: 200-1400ms
- lang_detect: 400-2400ms
- think: 4300-5100ms

## Выводы
- Каскад работает! Модель сама решает когда эскалировать
- Qwen2.5-0.5B справляется с complex EN (TCP/UDP, prime) — не нужна большая модель
- RU корректно уходит в LANG_DETECT
- KK уходит в THINK вместо LANG_DETECT (Qwen не знает казахский как язык)
- Qwen3 включает `<think>` reasoning chain

## Сравнение

| Exp | Accuracy | Architecture |
|-----|----------|-------------|
| exp_008 | 93% | tinygrad 125K × 6 (toy) |
| exp_012 | 38% | langdetect + SmolLM2-360M |
| exp_013 | 75% | langdetect + SmolLM2-1.7B MLX |
| **exp_014** | **79%** | **Cascading Qwen2.5-0.5B → Qwen3-1.7B** |

## Запуск
```bash
cd nano && .venv/bin/python exp/exp_014/run.py
```
