# exp_008 — Multi-tool Think: Math + Code + Error

## Что просили
Добавить к Math языки программирования (Python, JS) и обработку ошибок. Think должен определить домен (math/code/error) и язык, сгенерировать план, выполнить.

## Что привело к этому
- exp_006: Think-оркестратор (decoder-only), Math, KK+RU+EN, 100%
- exp_007: то же на encoder-decoder — хуже (80%), decoder-only побеждает
- exp_008: расширяем decoder-only на новые инструменты

## Архитектура

6 моделей (все decoder-only, dim=64, 4 heads, 2 layers):

| Модель | Params | Задача |
|--------|--------|--------|
| Think | 127K | `<think> [input] <plan> [actions]` — оркестратор |
| M1 | 125K | KK/RU → EN перевод |
| Math | 125K | Арифметика на EN (слова) |
| Code | 125K | Python/JS арифметика (цифры) |
| M3 | 125K | EN → KK/RU перевод |
| Err | 125K | Генерация ошибок на языке запроса |

### Планы Think
```
Math KK:  <lang_kk> <translate> <math> <translate_back>
Math RU:  <lang_ru> <translate> <math> <translate_back>
Math EN:  <lang_en> <math>
Python:   <lang_py> <code>
JS:       <lang_js> <code>
Мусор:    <error>
```

## Данные
800 примеров: 396 math + 374 code + 30 error. Словарь: 181 токен.
- Math: числа 0-10 словами, +/-, 3 языка
- Code: `print(a+b)` / `console.log(a+b)`, числа цифрами, результат цифрой
- Error: 10 фраз × 3 языка (приветствия, вопросы, не-math/code)

## Результаты (300 шагов, Metal M1, 279s)

| Модель | Loss start → end |
|--------|-----------------|
| Think | 4.97 → 0.002 |
| M1 | 5.32 → 0.002 |
| Math | 5.32 → 0.002 |
| Code | 5.89 → 0.246 |
| M3 | 5.47 → 0.090 |
| Err | 5.31 → 0.001 |

### Инференс: Plan 14/14 (100%), Answer 13/14 (93%)
- 1 ошибка: Code `print(3+4)=6` (Code loss=0.246, недообучен на цифрах)

## Выводы
- Think безошибочно маршрутизирует по 6 разным планам
- Error model корректно генерирует ошибки на 3 языках
- Code — слабое место (цифры + большой словарь); нужно больше шагов
- Архитектура масштабируется: добавление новых tools = новая модель + новые планы

## Запуск
```bash
cd nano && METAL=1 .venv/bin/python exp/exp_008/train.py
```
