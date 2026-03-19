# exp_007 — Encoder-Decoder версия Think-оркестратора

## Что просили
Тот же пайплайн что exp_006 (Think + M1 + Math + M3), но все 4 модели encoder-decoder вместо decoder-only. Сравнение архитектур.

## Что привело к этому
exp_006 показал 100% точность с decoder-only. Вопрос: будет ли encoder-decoder лучше или хуже?

## Архитектура (вариант A)

4 модели (все encoder-decoder, dim=64, 4 heads, 2 layers, ~249K params каждая):

| Модель | Encoder input | Decoder output |
|--------|--------------|----------------|
| Think | [user input text] | [plan actions] |
| M1 | `<kk>`/`<ru>` [src_q] | [en_q] |
| Math | [en_q] | [en_a] |
| M3 | `<kk>`/`<ru>` [en_a] | [src_a] |

M3 получает тег языка в encoder чтобы знать на какой язык переводить.

## Данные
396 примеров (132 KK + 132 RU + 132 EN), числа 0-10, +/-

## Баги и фиксы
1. **M3 выводил `<kk>` в тексте** → фильтрация спецтокенов в `ids_to_text`
2. **M3 всегда переводил на казахский** → тег языка передаётся в encoder (не в decoder)
3. **Math недообучен при 200 шагах** → увеличено до 300

## Результаты (300 шагов, Metal M1)

| Модель | Loss start → end | Время |
|--------|-----------------|-------|
| Think | 4.320 → 0.001 | 110s |
| M1 | 4.369 → 0.001 | 111s |
| Math | 4.242 → 0.034 | 107s |
| M3 | 4.291 → 0.002 | 99s |

### Инференс: Plan 10/10 (100%), Answer 8/10 (80%)
- 2 ошибки: Math считает `10-5=3` (недообучен)

## Сравнение с exp_006 (decoder-only)

| | exp_006 (dec-only) | exp_007 (enc-dec) |
|---|---|---|
| Answer | **100%** | 80% |
| Math loss | **0.002** | 0.034 |
| Params/модель | **110K** | 249K |
| Время | **241s** | 442s |

## Выводы
- Encoder-decoder в 2.3x больше параметров, в 1.8x медленнее
- Math (enc-dec) сходится хуже чем Math (dec-only) — cross-attention overhead для простой seq2seq задачи
- Think/M1/M3 работают хорошо в обоих вариантах
- **Decoder-only лучше для этой задачи** по всем метрикам

## Запуск
```bash
cd nano && METAL=1 .venv/bin/python exp/exp_007/train.py
```
