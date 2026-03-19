# exp_012 — langdetect Router + SmolLM2-360M + HPLT + Marian

## Что просили
langdetect + rule-based router вместо SmolLM2 для роутинга. SmolLM2-360M только для задач (Math/Code/Err).

## Что привело к этому
- exp_011: SmolLM2-135M на всё — провал (Route 0%, Answer 6%)
- Проблема: 135M не умеет следовать инструкциям
- Решение: rule-based router + более крупная модель (360M)

## Архитектура

| Компонент | Модель |
|-----------|--------|
| Router | langdetect + keywords (rule-based) |
| M1_kk (KK→EN) | HPLT CTranslate2 float32 |
| M1_ru (RU→EN) | Helsinki-NLP/opus-mt-ru-en |
| Math/Code/Err | SmolLM2-360M-Instruct |
| M3_kk (EN→KK) | HPLT CTranslate2 float32 |
| M3_ru (EN→RU) | Helsinki-NLP/opus-mt-en-ru |

### Router
- Казахские уникальные символы (қңғүұіөәһ) → kk
- Кириллица → ru
- Латиница → langdetect fallback
- Keywords: қосу/алу → math_kk, плюс/минус → math_ru, plus/minus → math_en
- print() → code_py, console.log() → code_js
- Всё остальное → error

## Результаты

| Задача | Результат |
|--------|-----------|
| **Route** | **16/16 (100%)** |
| Math KK (HPLT+SmolLM2) | 1/3 |
| Math RU (Marian+SmolLM2) | 2/3 |
| Math EN (SmolLM2) | 3/4 |
| Code (SmolLM2) | 1/3 |
| Error (SmolLM2) | 0/3 |
| **Answer total** | **6/16 (38%)** |

## Выводы
- **langdetect router: 100%** — символьный анализ + keywords надёжнее любого LLM
- SmolLM2-360M лучше 135M в арифметике (считает 6+3=9, 8-2=6)
- Но нестабильна: 1+2=1, 10-7=10, не выполняет код надёжно
- Error: не следует "reply exactly" — генерирует свободный текст
- Marian RU↔EN переводит значительно лучше чем HPLT KK↔EN

## Сравнение экспериментов

| Exp | Route | Answer | Router | Task model |
|-----|-------|--------|--------|------------|
| exp_008 | 100% | 93% | tinygrad Think 126K | tinygrad 125K×5 |
| exp_011 | 0% | 6% | SmolLM2-135M | SmolLM2-135M |
| **exp_012** | **100%** | **38%** | **langdetect+keywords** | **SmolLM2-360M** |

## Запуск
```bash
cd nano && .venv/bin/python exp/exp_012/train.py
```
