# Postmortem: exp001 (Mimi latents) + exp002 (Mel spectrograms)

**Дата:** 2026-02-21
**Статус:** ПРОВАЛ (документированный)
**Инстанс:** vast.ai RTX 4090, $0.31/hr

---

## exp001_v4: Flow matching на Mimi 512-dim латентах

### Конфигурация
- **Подход:** audio → Mimi encode → 512-dim latents → flow matching → Mimi decode → waveform
- **Модель:** TTSBackbone, 10 layers, d_model=1024, 16 heads, ~60M params
- **Данные:** `stukenov/kzcalm-tts-kk-v1` (232K samples, 439h, 24kHz) — предварительно экстрагированные Mimi латенты
- **Тренировка:** batch_size=64, lr=1e-4, 200K steps, huber loss

### Результат
- **104K шагов**, loss застрял на **~21.5** с ~50K шагов
- Генерация — **неразборчивый шум**, голос не распознаётся
- Скорость: 5.8 steps/s

### Причины провала
1. **Размерность 512 слишком высока** для flow matching — модель не может выучить distribution в таком пространстве
2. **Mimi латенты плохо структурированы** для generative modeling — они оптимизированы для reconstruction, не для generation
3. **Нет alignment** между текстом и аудио — cross-attention должна сама выучить mapping

---

## exp002_mel: Flow matching на mel спектрограммах (80→100 dim)

### Конфигурация
- **Подход:** audio → mel spectrogram (100 bins) → flow matching → Vocos vocoder → waveform
- **Модель:** TTSBackbone, 10 layers, d_model=512, 8 heads, d_ff=2048
- **Данные:** тот же датасет, mel extraction на лету (streaming)
- **Тренировка:** batch_size=32, grad_accum=2, lr=1e-4, warmup=2000, huber loss
- **Vocoder:** Vocos `charactr/vocos-mel-24khz` (100 mel bins, 24kHz)

### Хронология
1. Первый запуск с n_mels=80 — Vocos ожидает 100 bins, пришлось паддить
2. Перезапуск с n_mels=100 — нормально
3. Первый batch с max_audio_frames=3000 вызвал OOM — уменьшили до 1500 (но чекпоинты сохранились с 3000 в весах)
4. Обнаружили неправильную нормализацию: hardcoded mean=-6.0/std=3.0 vs реальные mean=-1.42/std=3.80
5. Пересчитали статистику на 5000 семплов — стабильно mean=-1.42, std=3.80

### Результат
- **~100K шагов**, loss застрял на **~22-24** с ~50K шагов
- Генерация: **не шум, слышен казахский акцент, проскакивают отдельные слоги**, но **полностью неразборчиво**
- Скорость: 2.8-3.2 steps/s
- Время тренировки: ~10 часов

### Loss trajectory
```
step=0       loss=1.52
step=100     loss=130.75
step=500     loss=64.27
step=1000    loss=56.13
step=5000    loss=~35
step=25000   loss=~24
step=50000   loss=~23
step=70000   loss=~22.5
step=95000   loss=~22.5   ← плато
```

### Прослушанные чекпоинты
| Step | Результат |
|------|-----------|
| 5000 | Шум с намёками на голос |
| 25000 | Неразборчиво |
| 70000 | Неразборчиво, но не шум |
| 80000 | Слышен акцент, проскакивают слоги |
| 95000 | Чуть лучше, слова пытаются проявиться, но неразборчиво |

---

## Ошибки при реализации

1. **Vocos mel bins mismatch:** `charactr/vocos-mel-24khz` требует 100 mel bins, мы начали с 80. Пришлось перезапускать.
2. **datasets 4.5 + torch 2.4.1 несовместимость:** datasets 4.5 требует torchcodec, а torchcodec не работает с torch 2.4.1. Даунгрейд до datasets 3.6.
3. **Неправильная нормализация:** hardcoded mean=-6.0/std=3.0 vs реальные -1.42/3.80. Однако после исправления качество не улучшилось — это не было root cause.
4. **max_audio_frames путаница:** первый запуск с 3000 вызвал OOM, изменили на 1500 в конфиге, но чекпоинты уже сохранились с latent_pos.weight shape [3000, 512]. Пришлось вручную указывать 3000 при инференсе.
5. **torchcodec/ffmpeg:** на vast.ai docker image нет ffmpeg, torchcodec не линкуется. На локальной машине torchaudio 2.10 требует torchcodec для save. Workaround: soundfile.

---

## Корневая причина провала

**Отсутствие alignment между текстом и mel.**

Наша архитектура (TTSBackbone) — vanilla transformer decoder с cross-attention к тексту. Модель должна сама выучить:
1. Какая фонема соответствует какому участку mel
2. Длительность каждой фонемы
3. Генерацию mel-спектрограммы через flow matching

Это **слишком много** для одной модели без explicit alignment. Все успешные flow-matching TTS системы используют:

- **Monotonic Alignment Search (MAS)** — Grad-TTS, Matcha-TTS
- **Duration predictor** — предсказывает длительность каждой фонемы
- **Phoneme-level conditioning** — text расширяется до длины mel через duration

Без этого cross-attention не может выучить правильный mapping text→mel, и loss застревает.

---

## Выводы и рекомендации

1. **Flow matching на mel работает лучше чем на Mimi латентах** (loss падает, есть намёки на речь vs чистый шум)
2. **Но без alignment/duration predictor модель не может генерировать разборчивую речь**
3. **Рекомендуемый следующий шаг:** взять архитектуру Matcha-TTS (flow matching + MAS + duration predictor) и адаптировать для казахского
4. **Vocos vocoder работает** — при правильном mel он должен давать хороший waveform
5. **Данные достаточные** — 232K samples, 439h — более чем достаточно для TTS

---

## Затраты

| Ресурс | Значение |
|--------|----------|
| vast.ai RTX 4090 | ~2400 часов * $0.31 = ~$744 (включая все эксперименты) |
| exp002_mel тренировка | ~10 часов * $0.31 = ~$3.1 |
| Инженерное время | ~1 день |
