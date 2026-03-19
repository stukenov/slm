# exp_016 — Development branch от exp_015

## База
Точная копия exp_015 (92%, Cascading Qwen2.5→Qwen3 + HPLT(KK) + NLLB(rest)).
exp_015 зафиксирован как рабочий baseline.

## Что можно улучшать
- Поднять Qwen до 3B/7B (больше знаний, Астана, Абай)
- Убрать/заменить self-check (сейчас бесполезен — всегда YES)
- Ускорить NLLB для RU (3-14s → нужно быстрее)
- Добавить кэш переводов
- Streaming ответов

## Запуск
```bash
cd nano && .venv/bin/python exp/exp_016/run.py
```
