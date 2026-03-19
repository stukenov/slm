# exp_013 — MLX SmolLM2-1.7B + langdetect + HPLT + Marian

## Что просили
SmolLM2-1.7B через быстрый MLX движок (Metal GPU). Только инференс, без обучения.

## Что привело к этому
- exp_011: SmolLM2-135M на всё — 6%
- exp_012: langdetect router + SmolLM2-360M — 38%
- exp_013: langdetect router + SmolLM2-1.7B MLX — **75%**

## Архитектура

| Компонент | Модель |
|-----------|--------|
| Router | langdetect + keywords (rule-based) |
| M1_kk (KK→EN) | HPLT CTranslate2 float32 |
| M1_ru (RU→EN) | Helsinki-NLP/opus-mt-ru-en |
| Math/Code/Err | **mlx-community/SmolLM2-1.7B-Instruct (MLX Metal)** |
| M3_kk (EN→KK) | HPLT CTranslate2 float32 |
| M3_ru (EN→RU) | Helsinki-NLP/opus-mt-en-ru |

## Результаты

| Задача | Результат |
|--------|-----------|
| **Route** | **16/16 (100%)** |
| Math RU | 3/3 (100%) |
| Math EN | 4/4 (100%) |
| Math KK | 1/3 (HPLT перевод плохой) |
| Code | 3/3 (100%) |
| Error EN | 1/1 |
| Error KK/RU | 0/2 |
| **Answer total** | **12/16 (75%)** |

Загрузка: 156с (SmolLM2 скачивание + MLX), инференс: 37с.

## Сравнение экспериментов

| Exp | Route | Answer | Task model | Engine |
|-----|-------|--------|------------|--------|
| exp_008 | 100% | 93% | tinygrad 125K×5 | tinygrad Metal |
| exp_011 | 0% | 6% | SmolLM2-135M | transformers CPU |
| exp_012 | 100% | 38% | SmolLM2-360M | transformers CPU |
| **exp_013** | **100%** | **75%** | **SmolLM2-1.7B** | **MLX Metal** |

## Выводы
- SmolLM2-1.7B — значительный скачок: Math и Code почти безупречно
- MLX Metal — быстрый инференс на M1
- Слабые места: HPLT KK↔EN перевод (не наш формат) и error на кириллице
- Marian RU↔EN переводит отлично — 100% на RU math

## Запуск
```bash
cd nano && .venv/bin/python exp/exp_013/train.py
```
