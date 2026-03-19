# exp_011 — All Real Models (SmolLM2-135M + HPLT + Marian)

## Что просили
Заменить все тойшные tinygrad модели на реальные из экосистемы HuggingFaceTB. Одна SmolLM2-135M-Instruct на все задачи (Think/Math/Code/Err). Перевод: HPLT KK↔EN + Helsinki-NLP Opus-MT RU↔EN.

## Что привело к этому
- exp_008: 6 тойшных tinygrad моделей, 93%
- exp_009: HPLT CTranslate2 INT8 адаптер для KK↔EN, 92%
- exp_010: SmolLM2-135M вместо Math, 36% (SmolLM2 плохо считает)
- exp_011: SmolLM2-135M на ВСЁ

## Архитектура

| Компонент | Модель |
|-----------|--------|
| Think (роутер) | SmolLM2-135M-Instruct (system prompt) |
| M1_kk (KK→EN) | HPLT CTranslate2 float32 |
| M1_ru (RU→EN) | Helsinki-NLP/opus-mt-ru-en (MarianMT) |
| Math | SmolLM2-135M-Instruct |
| Code | SmolLM2-135M-Instruct |
| M3_kk (EN→KK) | HPLT CTranslate2 float32 |
| M3_ru (EN→RU) | Helsinki-NLP/opus-mt-en-ru (MarianMT) |
| Err | SmolLM2-135M-Instruct |

## Результаты

**Route: 0/16 (0%), Answer: 1/16 (6%)**

Полный провал. SmolLM2-135M-Instruct:
- Не следует system prompt
- Не генерирует теги классификации
- Отвечает свободным текстом вместо следования инструкциям
- Плохо считает арифметику (7-3=2, 1+2=4)
- Не умеет выполнять код

Загрузка: 47с, инференс: 63с (5 моделей в RAM).

## Выводы
- SmolLM2-135M слишком маленькая для instruction following
- 135M параметров недостаточно для: роутинга, арифметики, eval кода
- Наши тойшные tinygrad модели (125K params) на узкой задаче работают ЛУЧШЕ чем 135M general LLM
- Специализация > размер для узких задач

## Запуск
```bash
cd nano && .venv/bin/python exp/exp_011/train.py
```
