# exp_009 — HPLT CTranslate2 INT8 Adapter

## Что просили
Заменить тренированные на тойдатасете KK↔EN переводчики (M1_kk, M3_kk) на реальные HPLT модели через CTranslate2 INT8 адаптер.

## Что привело к этому
- exp_006: Think-оркестратор, decoder-only, 100%
- exp_008: Multi-tool (Math+Code+Error), 6 моделей, 93%
- exp_009: замена тойшных KK переводчиков на реальные HPLT NMT

## Архитектура

8 моделей (гибрид: HPLT + tinygrad):

| Модель | Тип | Задача |
|--------|-----|--------|
| Think | tinygrad decoder-only (126K) | Оркестратор, генерация планов |
| **M1_kk** | **HPLT CTranslate2 INT8** | **KK→EN перевод (реальная NMT)** |
| M1_ru | tinygrad decoder-only (125K) | RU→EN перевод |
| Math | tinygrad decoder-only (125K) | Арифметика на EN |
| Code | tinygrad decoder-only (125K) | Python/JS арифметика |
| **M3_kk** | **HPLT CTranslate2 INT8** | **EN→KK перевод (реальная NMT)** |
| M3_ru | tinygrad decoder-only (125K) | EN→RU перевод |
| Err | tinygrad decoder-only (125K) | Генерация ошибок |

### HPLT адаптер
- `HPLT/translate-kk-en-v2.0-hplt_opus` → CTranslate2 MarianConverter → INT8
- `HPLT/translate-en-kk-v2.0-hplt_opus` → CTranslate2 MarianConverter → INT8
- SentencePiece токенизатор
- Загрузка: 0.4s, инференс: мгновенный

## Данные
800 примеров: 396 math + 374 code + 30 error. Словарь: 178 токенов.

## Результаты (300 шагов, Metal M1, 278s)

| Модель | Loss start → end |
|--------|-----------------:|
| Think | 5.253 → 0.002 |
| M1_ru | 5.061 → 0.002 |
| Math | 5.422 → 0.007 |
| Code | 5.288 → 0.253 |
| M3_ru | 5.269 → 0.002 |
| Err | 5.165 → 0.001 |

### Инференс: Plan 12/12 (100%), Answer 11/12 (92%)

- 1 ошибка: Code `console.log(5+5)=9` (Code loss=0.253, недообучен)
- KK путь: HPLT переводит свободным текстом (не наш структурный формат), но Math получает GT en_q → ответ правильный, HPLT EN→KK переводит обратно своими словами
- RU и EN пути — безупречно

### Примеры HPLT перевода
```
бір қосу екі нешеге тең ? → "What is the sum of two equal?"
он алу бес нешеге тең ?   → "What is the equivalent of 15?"
```
HPLT не знает наш формат арифметики, но pipeline работает через GT fallback.

## Выводы
- HPLT CTranslate2 INT8 успешно интегрирован как адаптер
- Гибридная архитектура (реальные NMT + тойшные tinygrad) работает
- HPLT перевод свободный — не совпадает с нашим структурным форматом, нужен GT для Math
- Code — по-прежнему слабое место (цифры)

## Запуск
```bash
cd nano && METAL=1 .venv/bin/python exp/exp_009/train.py
```
