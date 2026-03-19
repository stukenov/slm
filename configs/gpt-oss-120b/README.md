# GPT-OSS-120B Deployment on kaznu

## Подключение

| Параметр | Значение |
|----------|----------|
| **GUI** | `http://164.138.46.36:15127` |
| **API Base URL** | `http://164.138.46.36:15127/v1` |
| **Model name** | `GPT-OSS-120B` |
| **API Key** | не требуется (любой, например `"none"`) |
| **SSH** | `ssh -p 15126 root@164.138.46.36` (alias: `kaznu`) |

## API пример

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://164.138.46.36:15127/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="GPT-OSS-120B",
    messages=[{"role": "user", "content": "Салем!"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Сервер

| Параметр | Значение |
|----------|----------|
| Хост | 164.138.46.36 (kaznu) |
| GPU | 2x NVIDIA A10 (23 GB каждая) |
| RAM | 1 TB |
| CPU | 56-core Xeon Gold 6330N |
| OS | Ubuntu 22.04 |
| Порты | 15126 (SSH), 15127 (API/GUI) |

## Модель

| Параметр | Значение |
|----------|----------|
| Модель | GPT-OSS-120B (MoE, 120B total / 5.1B active) |
| Формат | GGUF mxfp4 (~60 GB, 3 файла) |
| Путь | `/root/models/gpt-oss-120b/` |
| Контекст | **131,072 токенов (128K)** |
| Скорость | ~37 t/s generation, ~23 t/s prompt |

## Параметры запуска

```bash
/root/llama.cpp/build/bin/llama-server \
  -m /root/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  --alias "GPT-OSS-120B" \
  -fa 1 -ncmoe 30 -ngl 99 --tensor-split 0.5,0.5 -fit off \
  -t 44 -ctk q4_0 -ctv q4_0 \
  -ub 2048 -b 2048 -c 131072 \
  --jinja --chat-template-file /root/chat_template.jinja \
  --temp 0.5 --top-p 0.9 --min-p 0.05 --top-k 40 \
  --repeat-penalty 1.1 \
  --path /root/webui \
  --host 0.0.0.0 --port 15127
```

### Ключевые флаги

| Флаг | Значение | Описание |
|------|----------|----------|
| `-ncmoe 30` | 30 | MoE слоёв на CPU (из 36). Освобождает GPU VRAM для контекста |
| `-ngl 99` | 99 | Все attention слои на GPU |
| `-fa 1` | — | Flash attention |
| `-ctk q4_0 -ctv q4_0` | — | KV cache в q4 (минимальный расход VRAM) |
| `-c 131072` | 128K | Контекстное окно |
| `-fit off` | — | Отключает auto-fit (иначе игнорирует tensor-split) |
| `--repeat-penalty 1.1` | — | Убирает повторения в казахском |
| `--jinja` | — | Jinja chat template |

## Файлы на сервере

| Файл | Описание |
|------|----------|
| `/root/llama.cpp/` | Собранный llama.cpp с CUDA (sm_86) |
| `/root/models/gpt-oss-120b/` | GGUF файлы модели (3 шт, ~60 GB) |
| `/root/chat_template.jinja` | Модифицированный шаблон: reasoning_effort="medium", системный промпт для казахского |
| `/root/system_prompt.txt` | Системный промпт (казахский натуральный язык) |
| `/root/webui/index.html` | Кастомный UI (Liquid Glass дизайн) |
| `/root/gpt_infer.log` | Лог сервера |
| Screen сессия | `gpt_infer` |

## Системный промпт

Встроен в `chat_template.jinja`. Если пользователь не отправляет system message, автоматически добавляется промпт, оптимизирующий модель для натурального казахского языка:
- Приоритет натуральности над буквальным переводом
- Запрет калек с русского/английского
- Примеры хорошего/плохого казахского
- Полный текст: `/root/system_prompt.txt`

## Reasoning

`reasoning_effort = "medium"` (в chat_template.jinja). Модель думает внутренне, но reasoning не показывается в UI.

Варианты: `"low"`, `"medium"`, `"high"` — чем выше, тем качественнее ответ, но медленнее.

## Параметры генерации (по умолчанию)

| Параметр | Значение |
|----------|----------|
| temperature | 0.5 |
| top_p | 0.9 |
| min_p | 0.05 |
| top_k | 40 |
| repeat_penalty | 1.1 |

## Бенчмарки контекста

| ncmoe | Контекст | tg (t/s) | Потеря |
|-------|----------|----------|--------|
| 24 | 8K | 42.0 | baseline |
| 25 | 16K | 41.1 | -2% |
| 26 | 24K | 40.5 | -4% |
| 27 | 32K | 39.5 | -6% |
| 28 | 48K | 37.8 | -10% |
| 30 | 64K | 37.2 | -11% |
| 30 | 128K | 37.1 | -12% |

## Перезапуск

```bash
ssh -p 15126 root@164.138.46.36
screen -X -S gpt_infer quit
# Скопировать команду запуска выше
screen -dmS gpt_infer bash -c '...'
```

## UI

Кастомный Liquid Glass чат-интерфейс. Без настроек, без thinking, без таймеров. Поддержка markdown: таблицы, заголовки, списки, код, ссылки, bold/italic.

Исходник: `/Users/sakentukenov/slm/configs/gpt-oss-120b/webui/index.html`
