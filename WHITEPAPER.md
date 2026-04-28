# SLM: Small Language Models for Kazakh — Experiment Log

## Introduction

This document tracks experiments in adapting small English language models to Kazakh through Domain-Adaptive Pre-Training (DAPT).

**Goal**: Determine the most effective approach for creating a small (&lt;50M parameters) language model capable of generating coherent Kazakh text.

**Methodology**: Start with pretrained Pythia models (14m, 31m) and continue training on a Kazakh multidomain dataset. Compare DAPT with original tokenizer vs. custom Kazakh tokenizer vs. training from scratch.

## Dataset

**Source**: `kz-transformers/multidomain-kazakh-dataset`

| Property | Value |
|----------|-------|
| Language | Kazakh (kk) |
| Domains | Multi-domain |
| Format | Text |

## Experiments

| # | Name | Model | Tokenizer | Steps | LR | Loss (final) | Status |
|---|------|-------|-----------|-------|-----|--------------|--------|
| 001 | DAPT Pythia-14m | pythia-14m | Original | 13,000 | 1e-4 | ~5.0 | Приостановлен |
| 002 | DAPT Pythia-31m | pythia-31m | Original | — | 5e-5 | — | Запланирован |
| 003 | Custom Tok 14m | pythia-14m | Kazakh BPE (+5000) | — | 1e-4 | — | В процессе |
| 004 | Scratch 14m | Random init | Kazakh BPE | — | — | — | Будущее |
| **013** | **Llama 50M scratch** | **Llama 50.3M** | **BPE 50K** | **36,616** | **6e-4** | **3.185** | **Завершён** |
| **014** | **Llama 150M scratch** | **Llama 151.9M** | **BPE 50K** | **36,616** | **3e-4** | **2.985** | **Завершён** |
| **015** | **Llama 150M SFT instruct** | **Llama 151.9M** | **BPE 50K** | **1,152** | **2e-5** | **2.918** | **Завершён** |
| **016** | **Llama 150M SFT ChatML** | **Llama 151.9M** | **BPE 50K** | **714** | **2e-5** | **~1.5** | **Завершён** |
| 018 | Llama 500M 200K vocab | Llama ~608M | BPE 200K | — | 3e-4 | — | Подготовка |
| **019** | **Llama 900M Chinchilla** | **Llama ~897M** | **BPE 50K** | **~67,800** | **2e-4** | **—** | **Запланирован** |
| **025** | **Llama 600M SFT sentiment** | **Llama 587M** | **BPE 50K** | **2,688** | **2e-5** | **~0.10** | **Завершён** |
| 026 | Llama 150M SFT sentiment | Llama 151.9M | BPE 50K | — | 2e-5 | — | Подготовлен |
| **028** | **Llama 1.08B GQA+QKNorm** | **Llama 1.08B** | **BPE 50K** | **~7,700** | **2e-4** | **2.15** | **Деградирован** |
| 038 | YouTube recent KK audio bootstrap | Audio corpus mining | recent-only speech | — | — | — | В процессе |

### EXP-038: Recent-only Kazakh YouTube Audio Bootstrap

**Config**: `configs/experiments/exp038_youtube_recent_kk_audio.yaml`

**Статус**: Спроектирован, inventory stage проверен локально

**Цель**: собрать `5K+` нетегированных казахских speech chunks для pretraining аудиоэнкодера из свежих YouTube-выпусков, не делая полный crawl каналов.

#### Ключевые правила

- Источники: `QazaqstanTV` и `aqparatkz`
- Полный historical crawl запрещён
- Отбор только из recent window через metadata inventory
- `aqparatkz` ограничен как вспомогательный источник и не должен доминировать в корпусе
- После каждого шага прогресс синхронизируется в Hugging Face dataset repo вместе с manifest/state файлами

#### Resumability

На Hugging Face сохраняются:

- `state/status.json`
- `state/selected_videos.jsonl`
- `state/processed_videos.jsonl`
- `state/rejected_videos.jsonl`

Новый запуск сначала подтягивает этот state и пропускает уже обработанные `video_id`, чтобы продолжение шло по уникальному контенту.

#### Inventory snapshot (2026-04-12)

- Candidate pool after metadata triage:
  - `qazaqstan_tv`: `102 / 800`
  - `aqparatkz`: `40 / 450`
- Selected manifest:
  - total: `113`
  - `qazaqstan_tv`: `102`
  - `aqparatkz`: `11`
- Current HF state:
  - `selected_videos = 113`
  - `processed_videos = 0`
  - `uploaded_chunks = 0`

## Experiment Details

### EXP-001: DAPT Pythia-14m

**Config**: `configs/experiments/exp001_dapt_pythia14m.yaml`

**Статус**: Приостановлен на 13,000 шагах

**HuggingFace**: [saken-tukenov/sozkz-core-pythia-14m-kk-dapt-v1](https://huggingface.co/saken-tukenov/sozkz-core-pythia-14m-kk-dapt-v1)

#### Прогресс обучения

| Шаг | Train Loss |
|-----|------------|
| 500 | 17.2 |
| 2,000 | 9.8 |
| 5,000 | 6.4 |
| 7,500 | 5.8 |
| 10,000 | 5.3 |
| 13,000 | 5.0 |

#### Наблюдения

- Loss снизился с ~17 до ~5 за 13K шагов
- Модель начала показывать казахские слова в генерации
- Оригинальный токенизатор GPT-NeoX неоптимален для казахского — многие символы разбиваются на отдельные токены
- Генерация на шаге 5K: почти только `⁇` (unknown tokens)
- Генерация на шаге 12.5K: появляются казахские слова

#### Причина приостановки

Обучение требует значительного времени (~52 часа для 5 эпох). На текущем этапе модель показывает признаки обучения (казахские слова появляются), но для полноценной оценки нужно сравнить с кастомным токенизатором (exp003).

### EXP-002: DAPT Pythia-31m

**Config**: `configs/experiments/exp002_dapt_pythia31m.yaml`

*Запланирован после завершения exp003.*

### EXP-003: Custom Tokenizer + DAPT Pythia-14m

**Config**: `configs/experiments/exp003_custom_tok_14m.yaml`

**Статус**: В процессе (обучение токенизатора)

Расширение оригинального токенизатора Pythia на 5000 казахских токенов. После создания токенизатора — обучение модели на 10,000 шагов для сравнения с exp001.

Гипотеза: кастомный токенизатор должен улучшить качество генерации, так как казахские слова будут представлены целыми токенами, а не разбиты на отдельные символы.

## Data Collection: sozkz-corpus-dedup-kk-web-v1

**Дата**: 2026-02-13

**HuggingFace**: [saken-tukenov/sozkz-corpus-dedup-kk-web-v1](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-dedup-kk-web-v1)

**Код**: `src/slm/collect/` (модуль сбора), `scripts/merge_and_push.py` (дедупликация и загрузка)

### Цель

Собрать казахский текст из всех крупных публичных датасетов на HuggingFace для расширения обучающей выборки. Результат — корпус, дополняющий `kz-transformers/multidomain-kazakh-dataset`.

### Источники и результаты

| Источник | HF Dataset | Собрано | Уникальных новых | Дубли с existing | Дубли cross-source |
|----------|-----------|---------|-----------------|------------------|--------------------|
| culturax | uonlp/CulturaX (kk) | 2,731,934 | 2,705,991 | 25,943 | 0 |
| hplt | HPLT/HPLT2.0_cleaned (kaz_Cyrl) | 2,637,330 | 2,246,264 | 4,229 | 386,837 |
| mc4 | allenai/c4 (kk) | 2,371,528 | 2,230,795 | 3,563 | 137,170 |
| madlad400 | allenai/MADLAD-400 (kk) | 1,807,996 | 1,807,827 | 136 | 33 |
| moscar | oscar-corpus/mOSCAR (kaz_Cyrl) | 245,869 | 245,869 | 0 | 0 |
| wikipedia | wikimedia/wikipedia (20231101.kk) | 238,356 | 238,343 | 12 | 1 |
| **Итого** | | **10,033,013** | **9,475,089** | **33,883** | **524,041** |

### Дедупликация

- Метод: exact dedup по MD5 хешу текста
- Референс: kz-transformers/multidomain-kazakh-dataset (12,441,904 уникальных хешей, streaming)
- Cross-source: дополнительная дедупликация между 6 собранными источниками
- Dedup rate: 5.6% (557,924 из 10,033,013 отсеяно)

### Очистка текста

- Unicode NFC нормализация
- Удаление управляющих символов
- Схлопывание пробелов и переводов строк
- Фильтр по длине (min 50 символов)
- Фильтр по плотности URL (max 5 на 1000 символов)
- Фильтр по HTML тегам (max 5)

### Итоговый корпус

| Метрика | Значение |
|---------|----------|
| Уникальных текстов | 9,475,089 |
| Формат | Parquet (142 шарда) |
| Колонки | `text`, `source` |
| Совместно с multidomain | ~21.9M уникальных текстов |

### Технические заметки

- `streaming=True` для экономии RAM при сборе
- `data_files` параметр для MADLAD-400 и HPLT (обход медленного листинга файлов)
- mOSCAR вместо OSCAR-2301 (последний gated/suspended)
- allenai/c4 вместо mc4 (dataset scripts deprecated в datasets 5.x)
- CC-100 исключён (нет альтернативы без dataset scripts)
- HPLT parquet повреждён при первом сборе — пересобран
- Хеширование existing dataset (12.4M) заняло ~2ч, upload на HF ~1.5ч

### EXP-013: Llama 50M from Scratch — 9B Kazakh Tokens

**Config**: `configs/experiments/exp013_llama_50m_9b.yaml`

**Статус**: Завершён

**HuggingFace**: [saken-tukenov/sozkz-core-llama-50m-kk-base-v4](https://huggingface.co/saken-tukenov/sozkz-core-llama-50m-kk-base-v4)

**Датасет**: [saken-tukenov/sozkz-corpus-tokenized-kk-llama50k-v3](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-tokenized-kk-llama50k-v3) (pre-tokenized, ~8.8B tokens)

**Токенизатор**: [saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1) (BPE, 50K vocab)

#### Архитектура

| Параметр | Значение |
|----------|----------|
| Архитектура | LlamaForCausalLM (from scratch) |
| Параметры | 50.3M |
| Слои | 8 |
| Hidden size | 576 |
| Attention heads | 8 |
| Intermediate (SwiGLU) | 1,536 |
| Vocab size | 50,000 |
| Context length | 2,048 |

#### Гиперпараметры обучения

| Параметр | Значение |
|----------|----------|
| Hardware | 2× NVIDIA RTX 4090 (vast.ai, DDP) |
| Batch size | 24 per GPU × 5 grad accum × 2 GPUs = 240 |
| Max LR | 6e-4 |
| Schedule | Cosine с 2% warmup |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight decay | 0.1 |
| Precision | bf16 |
| Шагов | 36,616 |
| Время | ~12.5 часов |
| Скорость | 1.20 s/step |

#### Прогресс обучения

| Шаг | Train Loss | Perplexity |
|-----|------------|------------|
| 500 | 5.069 | 159.0 |
| 1,000 | 4.282 | 72.4 |
| 2,000 | 3.865 | 47.7 |
| 4,000 | 3.627 | 37.6 |
| 8,000 | 3.468 | 32.1 |
| 16,000 | 3.330 | 27.9 |
| 24,000 | 3.254 | 25.9 |
| 32,000 | 3.190 | 24.3 |
| 36,000 | 3.178 | 24.0 |
| **36,616 (final)** | **3.184** | **24.2** |
| **Eval loss** | **3.185** | **24.2** |

#### Примеры генерации (checkpoint-34000)

**Промпт:** «Бүгінгі күні білім беру»
> Бүгінгі күні білім беру – барлық дамыған елдердің білім беру жүйесі мен сапасына тікелей байланысты. Білім беру жүйесін жетілдіру үшін мемлекет, жеке бизнес және қоғамдық сектордың белсенді өзара іс-қимылдары қажет.

**Промпт:** «Жасанды интеллект технологиясы»
> Жасанды интеллект технологиясы», «Робототехника негіздері» пәндері бойынша семинар-тренингтер өткізілді. Ағылшын тілін үйренудегі жетістіктер, шетелдік және отандық авторлардың заманауи оқу және ғылыми жетістіктерін қолдана отырып, интерактивті оқытудың жаңа әдістерін қолдану...

**Промпт:** «Алматы қаласында»
> Алматы қаласында Бішкек қаласында «Тойота» автокөлігінің қатысуымен жол-көлік оқиғасы орын алды. Автобус жүргізушісі «Lada» маркалы көлікті басқара алмай, жолдың қарсы бағытына шығып кетіп...

#### Наблюдения

- Train/eval loss практически совпадают (3.184 vs 3.185) — модель не переобучена
- Грамматически правильный казахский, связные абзацы, реалистичный новостной стиль
- Типичные артефакты маленькой модели: смешение городов/контекстов, обрыв кавычек
- Loss стабилизировался после ~30K шагов, cosine schedule правильно затухает к нулю
- 50M параметров достаточно для связного казахского текста при обучении на 9B токенов

### EXP-014: Llama 150M from Scratch — 9B Kazakh Tokens

**Config**: `configs/experiments/exp014_llama_150m_9b.yaml`

**Статус**: Завершён

**HuggingFace**: [saken-tukenov/sozkz-core-llama-150m-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-llama-150m-kk-base-v1)

**Датасет**: [saken-tukenov/sozkz-corpus-tokenized-kk-llama50k-v3](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-tokenized-kk-llama50k-v3) (pre-tokenized, ~9.0B tokens)

**Токенизатор**: [saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1) (BPE, 50K vocab)

#### Архитектура

| Параметр | Значение |
|----------|----------|
| Архитектура | LlamaForCausalLM (from scratch) |
| Параметры | 151.87M |
| Слои | 16 |
| Hidden size | 768 |
| Attention heads | 12 (MHA) |
| Intermediate (SwiGLU) | 2,048 |
| Vocab size | 50,257 |
| Context length | 1,024 |
| Tie embeddings | Yes |

#### Гиперпараметры обучения

| Параметр | Значение |
|----------|----------|
| Hardware | 2× NVIDIA RTX 4090 (vast.ai, DDP) |
| Batch size | 12 per GPU × 10 grad accum × 2 GPUs = 240 |
| Max LR | 3e-4 |
| Schedule | Cosine с 750 warmup steps |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight decay | 0.1 |
| Max grad norm | 1.0 |
| Precision | bf16 |
| Шагов | 36,616 |
| Время | ~26 часов |
| Скорость | 2.37 s/step |
| Стоимость | ~$16.7 ($0.64/hr × 26h) |

#### Данные обучения

| Метрика | Значение |
|---------|----------|
| Исходный корпус | sozkz-corpus-clean-v3 (13.7M документов) |
| Источники | 18 (CulturaX, HPLT, mC4, CC-100, MADLAD-400, Leipzig, Wikipedia и др.) |
| Казахских слов | ~5.4 млрд |
| Казахских предложений | ~501 млн |
| Символов | ~42.7 млрд |
| Очистка | 9-этапный пайплайн (NFC, мин. длина, каз. символы, скрипт, LID, dedup) |
| Pass rate | 48.2% (28.4M → 13.7M) |
| Токенов (BPE 50K) | ~9.0B (8,787,709 blocks × 1024) |

#### Прогресс обучения

| Шаг | Train Loss | Perplexity |
|-----|------------|------------|
| 50 | 8.800 | 6,634 |
| 100 | 8.103 | 3,303 |
| 150 | 7.360 | 1,572 |
| 200 | 6.782 | 879 |
| 250 | 6.370 | 584 |
| 300 | 6.020 | 411 |
| 350 | 5.731 | 308 |
| 400 | 5.504 | 246 |
| **Final** | **~2.98** | **~19.8** |
| **Eval loss (200 samples)** | **2.985** | **19.78** |

#### Примеры генерации

**Политика / Новости:**

Промпт: «Қазақстан Президенті»
> Қазақстан Президенті Нұрсұлтан Назарбаев «Болашаққа бағдар: рухани жаңғыру» атты бағдарламалық мақаласында «Мен еліміз мықты, әрі жауапкершілігі жоғары Біртұтас Ұлт болу үшін болашаққа қалай қадам басатынымыз және бұқаралық сананы қалай өзгертетініміз туралы көзқарастарымды ортаға салуды жөн көрдім...»

**Экономика:**

Промпт: «Экономика министрлігі»
> Экономика министрлігі хабарлағандай, Қазақстанда жалпы құны 96 миллиард теңгені құрайтын 350-ден астам инвестициялық жоба жүзеге асырылып жатыр. «Алынған мәліметтерге сәйкес, бүгінгі таңда 258 мың жұмыс орны ашылды...»

**Білім / Образование:**

Промпт: «Білім беру жүйесі»
> Білім беру жүйесі үшін ең маңызды нәрсе - мұғалімдер мен оқушыларға білім берудің сапасы. Нені және қалай оқыту керек? Мұғалімнің кәсіби құзіреттілігі - сапалы білімге қол жеткізудегі негізгі фактор...

**Ауа райы / Погода:**

Промпт: «Ауа райы болжамы бойынша»
> Ауа райы болжамы бойынша, еліміздің кей жерлерінде жаңбыр жауып, найзағай ойнайды. Кей жерлерде тұман түседі. Оңтүстікте шаңды дауыл тұрады деп күтіледі. Жел оңтүстік-шығыстан соғады, күндіз екпіні 15-20 м/с жетеді...

**Тарих / История:**

Промпт: «Қазақ халқының тарихы»
> Қазақ халқының тарихы мен мәдениеті туралы сөз қозғағанда, ең алдымен еске алатындарыңыз осы. Қазақ халқы-тарихтың терең қойнауынан бастау алып, өз ұрпағының санасына сіңген ұлттық коды. Сондықтан да оның тарихы мен мәдениетін келер ұрпаққа аманаттау бүгінгі ұрпақтың парызы болмақ...

**Технологии:**

Промпт: «Жасанды интеллект технологиясы»
> Жасанды интеллект технологиясы бойынша жасалған. Бұл құрылғы адам мен робот арасындағы байланысты жүзеге асырады, нәтижесінде жасанды интеллект адамды қоршаған орта жайлы ақпаратты өңдеуге және өңдеуге мүмкіндік береді...

**Спорт:**

Промпт: «Футбол чемпионаты»
> Футбол чемпионаты мен Кубогында "Қайрат" сырт алаңда "Ақтөбеден" басым түсті. "Қайрат" футболшылары үш ұпаймен екінші орынға жайғасты...

**Денсаулық / Здоровье:**

Промпт: «Денсаулық сақтау министрлігі»
> Денсаулық сақтау министрлігі Қазақстан Республикасының Денсаулық сақтау министрі міндетін атқарушысының бұйрығымен бекітілген «Стационарлық деңгейде емдеумен және медициналық оңалтумен қамтамасыз ету қағидаларына» сәйкес медициналық көмек көрсету үшін іс-шараларды жүзеге асыру қажет...

**Күнделікті өмір / Быт:**

Промпт: «Алматы қаласында бүгін»
> Алматы қаласында бүгіннен бастап қалааралық автобустар уақытша жүрмейді. Бұл туралы қалалық жолаушылар көлігі және автомобиль жолдары бөлімінің басшысы Арман Әбдешов хабарлады...

#### Наблюдения

- **Perplexity 19.78 vs 24.2** (exp013): масштабирование 50M → 150M дало снижение PPL на 18%
- Грамматически правильный казахский, длинные связные абзацы
- Модель уверенно генерирует в разных доменах: политика, экономика, образование, спорт, погода, здоровье
- Реалистичный стиль казахских СМИ — цитаты, числа, ссылки на источники
- Типичные артефакты: галлюцинация конкретных цифр и имён, иногда обрыв контекста
- Train/eval loss совпадают — модель не переобучена
- Стоимость обучения ~$17 за полный прогон (26 часов на 2× RTX 4090)

#### Сравнение с exp013 (50M)

| Метрика | exp013 (50M) | exp014 (150M) | Δ |
|---------|-------------|---------------|---|
| Параметры | 50.3M | 151.9M | ×3.0 |
| Eval Loss | 3.185 | 2.985 | −0.20 |
| Perplexity | 24.2 | 19.8 | −18% |
| Время | 12.5 ч | 26 ч | ×2.1 |
| Стоимость | ~$7 | ~$17 | ×2.4 |
| Скорость | 1.20 s/step | 2.37 s/step | ×2.0 |

---

## Instruct Dataset: sozkz-instruct-chatml-kk-v1

**Дата**: 2026-02-18

**HuggingFace (EN)**: [stukenov/sozkz-instruct-chatml-en-v1](https://huggingface.co/datasets/stukenov/sozkz-instruct-chatml-en-v1)

**HuggingFace (KK)**: [stukenov/sozkz-instruct-chatml-kk-v1](https://huggingface.co/datasets/stukenov/sozkz-instruct-chatml-kk-v1)

### Цель

Собрать крупный instruct-датасет в формате ChatML (multi-turn messages) и перевести его EN→KK для масштабного SFT.

### Сбор (EN)

9 публичных англоязычных instruct-источников → **1,806,030** строк → дедупликация по MD5(первый user message) → **1,306,422** уникальных строк.

### Перевод EN→KK

| Параметр | Значение |
|----------|----------|
| Метод | CTranslate2, HPLT Marian (opus-mt-en-kk) |
| Hardware | 4× NVIDIA RTX 5090 (vast.ai) |
| Batch size | 16,384 |
| Время перевода | ~33 минуты |
| Стоимость | ~$0.50 |

### Итоговый датасет

| Split | Строк |
|-------|-------|
| Train | 1,293,357 |
| Validation | 13,065 |
| **Итого** | **1,306,422** |

**Формат**: ChatML — колонка `messages` содержит JSON массив `[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`

---

## Conclusions

### Промежуточные выводы (2026-02-06)

1. **DAPT работает**: Loss снижается, модель начинает генерировать казахские слова
2. **Токенизатор критичен**: Оригинальный GPT-NeoX токенизатор неоптимален для казахского
3. **Требуется время**: Полное обучение на 23.6M сэмплов занимает десятки часов
4. **Следующий шаг**: Сравнить результаты с кастомным токенизатором (exp003)

### Выводы по exp013 (2026-02-16)

1. **From scratch работает лучше DAPT**: 50M Llama с нуля (loss 3.18, ppl 24) значительно превосходит DAPT Pythia-14m (loss ~5.0) — кастомная архитектура + токенизатор дают кардинальное улучшение
2. **50M параметров достаточны** для связного казахского текста — модель генерирует грамматически правильные предложения, абзацы в новостном стиле
3. **9B токенов — оптимальный объём**: loss стабилизировался после ~30K шагов, train/eval loss совпадают (нет переобучения)
4. **Казахский BPE 50K** — правильный выбор: в отличие от оригинального GPT-NeoX, казахские слова представлены целыми токенами
5. **Cloud training на vast.ai** экономически эффективен: 2× RTX 4090, 12.5 часов, ~$5-7 за полный прогон

### Выводы по exp014 (2026-02-17)

1. **Масштабирование 50M → 150M даёт ощутимое улучшение**: PPL 24.2 → 19.8 (−18%), при этом стоимость выросла всего в 2.4 раза
2. **150M модель генерирует более связный и разнообразный текст**: уверенно работает в 10+ доменах (политика, экономика, образование, спорт, погода, здоровье, технологии, история)
3. **Данные**: 13.7M документов, ~5.4B казахских слов, ~501M предложений, ~9.0B токенов из 18 источников
4. **Архитектура Llama с SwiGLU оптимальна**: intermediate=2048 (8/3 × hidden) правильно учитывает 3 матрицы SwiGLU
5. **DDP на 2× RTX 4090** требует осторожного управления дисковым пространством: file-based barrier для координации загрузки датасета между ранками
6. **Следующие шаги**: fine-tuning для instruction following, evaluation на казахских бенчмарках, масштабирование до 300M+

### EXP-015: Llama 150M SFT Instruct

**Config**: `configs/experiments/exp015_sft_llama_150m.yaml`

**Статус**: Завершён

**HuggingFace Model**: [saken-tukenov/sozkz-core-llama-150m-kk-instruct-v1](https://huggingface.co/saken-tukenov/sozkz-core-llama-150m-kk-instruct-v1)

**HuggingFace Dataset**: [saken-tukenov/sozkz-corpus-synthetic-kk-instruct-v1](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-synthetic-kk-instruct-v1)

**Base Model**: [saken-tukenov/sozkz-core-llama-150m-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-llama-150m-kk-base-v1) (exp014)

**Токенизатор**: [saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1) (BPE, 50K vocab)

#### Цель

Supervised Fine-Tuning (SFT) base модели exp014 для instruction-following на казахском языке. Alpaca-казахский chat template с loss masking на prompt-токенах.

#### Chat Template

```
### Нұсқаулық:
{instruction}

### Кіріс:
{input}

### Жауап:
{output}<eos>
```

Секция «Кіріс» опускается если input пустой. Loss считается только на токенах после «### Жауап:\n».

#### SFT Датасет

| Источник | Строк | Лицензия |
|----------|-------|----------|
| AmanMussa/kazakh-instruction-v2 | 52,201 | MIT |

После дедупликации (MD5) и фильтрации (min 5 символов instruction, min 10 символов output):

| Split | Строк |
|-------|-------|
| Train | 49,047 |
| Validation | 496 |

#### Гиперпараметры обучения

| Параметр | Значение |
|----------|----------|
| Hardware | 2× NVIDIA RTX A5000 (vast.ai, DDP) |
| Batch size | 16 per GPU × 4 grad accum × 2 GPUs = 128 |
| Max LR | 2e-5 |
| Schedule | Cosine с 3% warmup |
| Weight decay | 0.01 |
| Max length | 512 |
| Precision | bf16 |
| Epochs | 3 |
| Шагов | 1,152 |
| Время | ~12 минут |
| Стоимость | ~$0.05 |

#### Прогресс обучения

| Шаг | Train Loss | Epoch |
|-----|------------|-------|
| 10 | 3.486 | 0.03 |
| 50 | 3.197 | 0.13 |
| 100 | 3.174 | 0.26 |
| 200 | 3.085 | 0.52 |
| 300 | 2.998 | 0.78 |
| 400 | 2.854 | 1.04 |
| 500 | 2.860 | 1.30 |
| 700 | 2.794 | 1.83 |
| 900 | 2.751 | 2.35 |
| 1000 | 2.708 | 2.61 |
| **1,152 (final)** | **~2.70** | **3.0** |
| **Eval loss** | **2.918** | |

#### Наблюдения

- Eval loss снижается: 2.949 (epoch 1.3) → **2.918** (epoch 3.0)
- Train loss упал с 3.486 → ~2.70 за 3 эпохи — модель успешно адаптировалась к instruction формату
- Обучение заняло всего ~12 минут на 2× A5000 (~$0.05) — SFT на маленьких моделях крайне дешёвый
- Loss masking работает корректно — модель обучается только генерировать ответы, не запоминать промпты
- Датасет AmanMussa/kazakh-instruction-v2 (52K) — основной источник казахских инструкций с MIT лицензией

### Выводы по exp015 (2026-02-17)

1. **SFT на 150M модели занимает минуты**: 3 эпохи на 49K примерах = 12 минут, $0.05
2. **Eval loss 2.918** — улучшение относительно base модели (2.985) указывает на успешную адаптацию к instruction формату
3. **Cloud pipeline (vast.ai) с --pre-cmd** позволяет готовить датасет прямо на обучающем сервере — удобно для small datasets
4. **Проблемы с датасетами**: MBZUAI/Bactrian-X и Vikhrmodels/OpenHermes-2.5-kz несовместимы с datasets 5.x (deprecated scripts, schema errors) — пришлось исключить
5. **Следующие шаги**: тестирование instruction following, добавление DPO/RLHF, сбор более качественного казахского SFT датасета

#### Benchmark: kk-socio-cultural-bench-mc

Оценка на [kz-transformers/kk-socio-cultural-bench-mc](https://huggingface.co/datasets/kz-transformers/kk-socio-cultural-bench-mc) — 7,111 multiple-choice вопросов о казахской культуре, истории, традициях.

**Overall accuracy: 10.4%** (742/7111) | Random baseline: 25.0%

| Категория | Accuracy |
|-----------|----------|
| Cinema | 13.4% |
| Literature: Poetry and Prose | 13.4% |
| Traditional Clothing | 12.5% |
| Cuisine and Beverages | 12.2% |
| Proverbs, Sayings, Mythology | 11.6% |
| History | 10.1% |
| Traditions | 7.1% |

Результат ниже random baseline — ожидаемо для 152M модели на задачах, требующих глубоких культурных знаний.

#### Benchmark: Kaz-Offline-Arena

Оценка на [Kaz-Offline-Arena](https://github.com/horde-research/Kaz-Offline-Arena) — open-ended QA с LLM-judge (GPT). 500 вопросов, 5 типов, оценка 0-10.

**Overall: 0.48 / 10** | Avg tokens: 24.7

| Тип вопроса | Балл |
|-------------|------|
| WHAT_QS | 0.67 |
| HOW_QS | 0.59 |
| DESCRIBE_QS | 0.42 |
| WHY_QS | 0.38 |
| ANALYZE_QS | 0.36 |

Крайне низкие баллы отражают ограничения 152M модели для open-ended генерации, требующей рассуждений и фактических знаний. Средняя длина ответа (24.7 токенов) тоже указывает на поверхностные ответы.

### EXP-016: Llama 150M SFT ChatML (Instruct v2)

**Config**: `configs/experiments/exp016_sft_chatml_150m.yaml`

**Дата**: 2026-02-18

**HuggingFace Model**: [stukenov/sozkz-core-llama-150m-kk-instruct-v2](https://huggingface.co/stukenov/sozkz-core-llama-150m-kk-instruct-v2)

**HuggingFace Dataset**: [stukenov/sozkz-corpus-chatml-kk-instruct-mix-v1](https://huggingface.co/datasets/stukenov/sozkz-corpus-chatml-kk-instruct-mix-v1)

#### Цель

Масштабное SFT базовой модели exp014 на 26× большем instruct-датасете (368K vs 49K) с переходом на ChatML multi-turn формат. Основной прогресс: от Alpaca-шаблона (single-turn) к ChatML (multi-turn).

#### Chat Template (ChatML)

```
<|user|>
Қазақстанның астанасы қандай?
<|end|>
<|assistant|>
Қазақстанның астанасы — Астана.
<|end|>
```

Специальные токены: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` — добавлены в токенизатор, эмбеддинги модели расширены.

#### SFT Датасет (Mix)

| Источник | Формат | Строк | Лицензия |
|----------|--------|-------|----------|
| stukenov/sozkz-instruct-chatml-kk-v1 | ChatML | 1,280,423 | Apache 2.0 |
| saillab/alpaca_kazakh_taco | Alpaca→ChatML | 62,308 | MIT |
| AmanMussa/kazakh-instruction-v2 | Alpaca→ChatML | 52,201 | MIT |

После дедупликации по MD5(первый user message) и объединения:

| Split | Строк |
|-------|-------|
| Train | 365,168 |
| Validation | 3,689 |
| **Итого** | **368,857** |

> Значительная дедупликация (1.39M → 369K, ~73% дубликатов) объясняется пересечением между переведённым датасетом и казахскими Alpaca-источниками, а также внутренней дупликацией.

#### Гиперпараметры

| Параметр | Значение |
|----------|----------|
| Base model | stukenov/sozkz-core-llama-150m-kk-base-v1 (152M) |
| Tokenizer | stukenov/sozkz-core-gpt2-50k-kk-base-v1 |
| SFT format | ChatML (multi-turn) |
| Max length | 1024 |
| Batch size | 4 × 16 GPU × 8 grad_accum = **512** |
| Learning rate | 2e-5 (cosine, warmup 3%) |
| Epochs | 1 |
| Steps | 714 |
| Loss masking | Prompt-токены маскированы, train только на assistant-ответах |

#### Обучение

| Параметр | Значение |
|----------|----------|
| Hardware | 16× NVIDIA RTX 4090 (vast.ai) |
| Время обучения | ~13 минут |
| Стоимость | ~$1.40 |
| Framework | PyTorch 2.4.1 + transformers 5.x, torchrun DDP |

#### Loss

| Шаг | Train Loss |
|-----|-----------|
| 10 | 3.302 |
| 20 | 2.984 |
| 30 | 2.658 |
| 40 | 2.374 |
| 50 | 2.131 |
| 60 | 1.892 |

Loss стабильно снижался на протяжении всего обучения. Финальный loss ~1.5 — существенное улучшение относительно exp015 (2.918).

#### Выводы по exp016 (2026-02-18)

1. **26× больше данных = значительно ниже loss**: 368K ChatML примеров vs 49K Alpaca → loss 1.5 vs 2.918
2. **ChatML multi-turn** позволяет моделировать диалоги, а не только single-turn Q&A
3. **Масштабирование compute**: 16× RTX 4090 на vast.ai, эффективный batch 512, 13 минут, $1.40
4. **Дедупликация критична**: 73% дубликатов между источниками — без дедупа модель бы переобучилась на повторяющихся примерах
5. **Следующие шаги**: бенчмарки (kk-socio-cultural-bench, Kaz-Offline-Arena), сравнение с instruct-v1, DPO/RLHF

#### Примеры инференса

Параметры генерации: `temperature=0.3`, `top_p=0.9`, `repetition_penalty=1.2`, `max_new_tokens=150`

**Q: Қазақстанның астанасы қандай?**
> Астана - бұл әлемдегі ең әдемі қалалардың бірі. Оның сұлулығы мен сұлулығына қарамастан, ол әрдайым керемет және тартымды болып қала береді. Ол өзінің ерекше атмосферасымен танымал...

**Q: Абай Құнанбайұлы кім?**
> Ақын, ақын және ойшыл. 1. Абай - ұлы ақын және философ. 2. Ол адам мен табиғат арасындағы қарым-қатынасты зерттейді. 3. Оның шығармалары көбінесе философиялық мәселелерге бағытталған. 4. Оның өлеңдері адамның табиғатын түсінуге көмектеседі...

**Q: Наурыз мейрамы туралы қысқаша айтып бер.**
> 1-наурыз - бұл күн мен түннің теңелуі, көктемнің басталуы және табиғаттың оянуы. Бұл күні адамдар бір-біріне сыйлықтар беріп, жаңа жылды қарсы алу үшін жиналады...

**Q: Бесбармақ қалай дайындалады?**
> Дәміне қарай, бір тілім нан немесе тоқаш. Тұздалған және пісірілген тағамдарға арналған рецептке сәйкес, ол жақсы қуырылған табада қуырылады. Содан кейін олар кесілгенше пісіріледі...

Модель генерирует связный казахский текст и частично верную фактическую информацию (Астана — столица, Абай — ақын/философ, Наурыз — көктем мерекесі). Однако наблюдаются галлюцинации и неточности — ожидаемо для 152M модели.

#### Бенчмарки EXP-016

**kk-socio-cultural-bench-mc** (7111 вопросов, 4 варианта ответа):

| Модель | Accuracy | Correct / Total |
|--------|----------|-----------------|
| instruct-v1 (exp015) | 10.4% | 740 / 7111 |
| **instruct-v2 (exp016)** | **10.9%** | **777 / 7111** |
| Random baseline | 25.0% | — |

Небольшое улучшение (+0.5 п.п.) при переходе на ChatML и 26× больше данных. Оба результата ниже random baseline — MC benchmark требует глубоких культурных знаний, которых у 152M модели недостаточно.

**Kaz-Offline-Arena** (500 вопросов, GPT-4o judge, 0-10 баллов):

| Модель | Avg Score | Avg Tokens |
|--------|-----------|------------|
| instruct-v1 (exp015) | 0.48/10 | 24.7 |
| **instruct-v2 (exp016)** | **0.75/10** | **299.0** |

Instruct-v2 показывает улучшение над v1 (0.75 vs 0.48) благодаря обрезке контекста до 720 токенов (модель обучена на max_length=1024). По типам вопросов: ANALYZE_QS=0.92, DESCRIBE_QS=0.91, HOW_QS=0.78, WHAT_QS=0.64, WHY_QS=0.53.

**Overall Score** (среднее нормализованных оценок по 10-балльной шкале):

| Модель | MC Bench (norm/10) | Arena (/10) | **Overall** |
|--------|--------------------|-------------|-------------|
| instruct-v1 (exp015) | 1.04 | 0.48 | **0.76** |
| **instruct-v2 (exp016)** | **1.09** | **0.75** | **0.92** |

**Анализ**: 152M параметров — критически мало для open-ended QA с контекстом. Модель генерирует казахский текст по формату, но ответы слабо соответствуют контексту. Для существенного улучшения необходимо увеличение размера модели (>1B параметров).

---

## Tokenizer: sozkz-core-gpt2-200k-kk-base-v1

**Дата**: 2026-02-19

**HuggingFace**: [stukenov/sozkz-core-gpt2-200k-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-gpt2-200k-kk-base-v1)

### Мотивация

Увеличение словаря с 50K до 200K для улучшения сжатия казахского текста. Казахский — агглютинативный язык, больший словарь позволяет кодировать длинные морфологические формы целиком.

### Архитектура

| Параметр | Значение |
|----------|----------|
| Тип | ByteLevel BPE (GPT-2 style) |
| Vocab size | 200,019 |
| Byte tokens | 256 |
| Special tokens | `<\|endoftext\|>`, `<\|padding\|>`, `<\|startoftext\|>` |
| Unicode digits | Все Nd категории (кроме ASCII 0-9) |
| model_max_length | 2048 |

### Данные обучения

| Источник | Сэмплов |
|----------|---------|
| saken-tukenov/sozkz-corpus-clean-kk-text-v2 | 18,768,063 |
| kz-transformers/multidomain-kazakh-dataset | 24,883,808 |
| **Итого** | **43,651,871** |

### Сжатие (vs 50K tokenizer)

Примеры на казахских предложениях:

| Текст | 50K | 200K | Экономия |
|-------|-----|------|----------|
| Қазақстан — Орталық Азиядағы мемлекет. | 8 | 6 | 25% |
| Бүгін ауа райы жақсы болады. | 8 | 7 | 12% |
| 2024 жылы халықаралық конференция өтеді. | 9 | 7 | 22% |
| Мектепте оқушылар математика сабағына дайындалуда. | 8 | 6 | 25% |
| Алматы қаласында жаңа метро стансасы ашылды. | 9 | 7 | 22% |

Средняя экономия: **~20%** на коротких предложениях.

### Обучение

- Hardware: RTX 3090 (vast.ai, $0.16/hr)
- Время: 29.7 минут
- Стоимость: ~$0.08

---

## Pre-tokenized Datasets (200K)

**Дата**: 2026-02-19

Три датасета токенизированы для обучения Llama 500M:

| Датасет | HuggingFace repo | block_size |
|---------|-----------------|------------|
| Clean Kazakh corpus | `stukenov/sozkz-corpus-tokenized-kk-200k-v1` | 2048 |
| Multidomain Kazakh | `stukenov/sozkz-corpus-tokenized-kk-multidomain-200k-v1` | 2048 |
| FineWeb-Edu EN+KK | `stukenov/sozkz-corpus-tokenized-enkk-200k-v1` | 2048 |

### Методология токенизации

Длинные тексты разбиваются на чанки по 20K символов (на границах абзацев/слов) перед токенизацией — это предотвращает straggler workers без потери данных. Затем все токены конкатенируются и нарезаются на блоки по 2048 токенов.

Скрипт: `scripts/tokenize_dataset.py`

---

### EXP-018: Llama 500M Dense from Scratch — 200K vocab (в процессе)

**Config**: `configs/experiments/exp018_llama_500m_200k.yaml`

**Статус**: Подготовка данных

#### Мотивация

По Chinchilla scaling laws, для ~10B токенов оптимальный размер модели ~500M параметров. Переход на 200K vocab для лучшего сжатия казахского текста.

#### Архитектура

| Параметр | Значение |
|----------|----------|
| Архитектура | LlamaForCausalLM (from scratch) |
| Vocab size | 200,019 |
| Hidden size | 1024 |
| Intermediate (SwiGLU) | 4096 |
| Layers | 24 |
| Attention heads | 16 (MHA) |
| Context length | 2048 |
| Tie embeddings | Yes |
| Ожидаемые параметры | ~500M |

#### Данные

Три pre-tokenized датасета (200K tokenizer, block_size=2048). Ожидаемый объём: ~10B+ токенов.

#### Обучение

| Параметр | Значение |
|----------|----------|
| Hardware | vast.ai (TBD) |
| LR | 3e-4 |
| Schedule | Cosine, 1000 warmup steps |
| Epochs | 1 |
| Weight decay | 0.1 |
| Precision | bf16 |

#### Проблемы exp018

- **Реальный размер ~608M** (не 500M) из-за гигантского embedding: 200019 × 1024 = 205M (34% всех параметров)
- **Не Chinchilla-оптимален**: ratio 10B/608M = 16.4:1 (нужно 20:1)
- 200K vocab оправдан только для моделей 3B+, где embedding overhead < 10%

---

### EXP-019: Llama 900M Chinchilla-optimal — 17.8B tokens (запланирован)

**Config**: `configs/experiments/exp019_llama_900m_chinchilla.yaml`

**Статус**: Запланирован

#### Мотивация

Максимальная Chinchilla-оптимальная модель на всех доступных данных с проверенным 50K токенизатором. В отличие от exp018 (200K vocab, 34% overhead на embedding), используем GPT2-50K — embedding overhead всего 8.6%.

Ключевое открытие: на HuggingFace есть **два** крупных датасета, токенизированных с 50K vocab:
- `sozkz-corpus-tokenized-kk-llama50k-v3`: 8.79M blocks = **9.0B токенов** (казахский)
- `sozkz-corpus-tokenized-enkk-fineweb-edu-v1`: 8.57M blocks = **8.8B токенов** (EN→KK FineWeb-Edu)
- **Итого: 17.8B токенов**

Chinchilla-оптимум: 17.8B / 20 = **890M параметров**.

#### Архитектура

| Параметр | Значение |
|----------|----------|
| Архитектура | LlamaForCausalLM (from scratch) |
| Параметры | **~897M** |
| Vocab size | 50,257 |
| Hidden size | 1536 |
| Intermediate (SwiGLU) | 5376 (3.5 × hidden) |
| Layers | 24 |
| Attention heads | 24 (MHA) |
| Context length | 2048 |
| Tie embeddings | Yes |
| Embedding overhead | 77M / 897M = **8.6%** |

##### Расчёт параметров

| Компонент | Формула | Параметры |
|-----------|---------|-----------|
| Embedding (tied) | 50257 × 1536 | 77M |
| Attention/layer (QKV+O) | 4 × 1536² | 9.4M |
| MLP/layer (SwiGLU: gate+up+down) | 3 × 1536 × 5376 | 24.8M |
| × 24 layers | 24 × 34.2M | 820M |
| **Итого** | | **~897M** |

#### Данные

| Датасет | Rows (train) | Токенов |
|---------|-------------|---------|
| stukenov/sozkz-corpus-tokenized-kk-llama50k-v3 | 8,787,709 | 9.0B |
| stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1 | 8,565,679 | 8.8B |
| **Итого** | **17,353,388** | **17.8B** |

Chinchilla ratio: 17.8B / 897M = **19.8:1** (target: 20:1)

#### Гиперпараметры обучения

| Параметр | Значение |
|----------|----------|
| Hardware | vast.ai (TBD — минимум A100 80GB или 2× A100 40GB) |
| Effective batch | 4 × 64 = 256 blocks = 262K tokens/step |
| Шагов (оценка) | ~67,800 |
| Max LR | 2e-4 |
| Schedule | Cosine, 2000 warmup steps |
| Weight decay | 0.1 |
| Max grad norm | 1.0 |
| Precision | bf16 |
| Epochs | 1 |

#### Сравнение с exp018

| Метрика | exp018 (500M/200K) | exp019 (900M/50K) |
|---------|-------------------|-------------------|
| Реальные params | 608M | 897M |
| Vocab | 200,019 | 50,257 |
| Embedding overhead | 34% | 8.6% |
| Данные | ~10B | 17.8B |
| Chinchilla ratio | 16.4:1 | **19.8:1** |
| Transformer params | 403M | 820M |

**exp019 имеет в 2× больше "рабочих" параметров** (820M vs 403M transformer params) при идеальном Chinchilla scaling.

### EXP-025: Llama 600M SFT Sentiment Classification

**Config**: `configs/experiments/exp025_sft_sentiment_600m.yaml`

**Статус**: Завершён

**HuggingFace**: [stukenov/sozkz-core-llama-600m-kk-sentiment-v1](https://huggingface.co/stukenov/sozkz-core-llama-600m-kk-sentiment-v1) (gated)

#### Задача

Бинарная классификация тональности казахского текста с использованием специального тега `<sentiment>`.

Формат ввода/вывода:
```
<sentiment>Тамақтары өте дәмді, қызмет көрсету керемет!</sentiment>
positive
```

#### Датасет

Источник: [issai/kazsandra](https://huggingface.co/datasets/issai/kazsandra) (KazSAnDRA, LREC 2024).

| Параметр | Значение |
|----------|----------|
| Маппинг | scores 1-2 → negative, 4-5 → positive, 3 → excluded |
| Балансировка | Undersample majority (30K per class) |
| Train | 57,312 |
| Validation | 3,016 |
| HF dataset | stukenov/sozkz-corpus-kazsandra-sentiment-v1 |

#### Гиперпараметры

| Параметр | Значение |
|----------|----------|
| Init model | sozkz-core-llama-600m-kk-base-v1 (587M) |
| Tokenizer | sozkz-core-gpt2-50k-kk-base-v1 |
| Batch size | 64 (8 × 4 GPU × 2 accum) |
| Learning rate | 2e-5 (cosine) |
| Epochs | 3 |
| Steps | 2,688 |
| max_length | 512 |
| Hardware | 4× RTX 4090 (vast.ai) |
| Training time | ~1.9h |
| Cost | ~$1.03 |

#### Прогресс обучения

| Step | Loss | Epoch |
|------|------|-------|
| 50 | 1.883 | 0.06 |
| 100 | 0.115 | 0.11 |
| 150 | 0.109 | 0.17 |
| 200 | 0.105 | 0.22 |
| 250 | 0.100 | 0.28 |

#### Результаты

10/10 на ручных тестах (4 positive, 4 negative, 2 ambiguous — все корректно).

| Текст | Предсказание | Верно |
|-------|-------------|-------|
| Тамақтары өте дәмді, қызмет көрсету керемет! | positive | ✓ |
| Бұл қосымша өте ыңғайлы, маған ұнады | positive | ✓ |
| Қызмет көрсету нашар, тамақ суық | negative | ✓ |
| Бұл қосымша жұмыс істемейді, ақша ысырап | negative | ✓ |
| Тамақ жаман емес, бірақ баға қымбат | negative | ✓ |
| Қалыпты, ерекше ештеңе жоқ | negative | ✓ |

#### Выводы

- Decoder-only модель отлично справляется с классификацией через prompt-based подход
- Loss быстро сходится до ~0.10 — задача простая для 600M модели
- Специальный тег `<sentiment>` работает как надёжный интерфейс классификации
- Следующий шаг: exp026 — тот же подход на 150M модели для сравнения

### EXP-026: Llama 150M SFT Sentiment Classification (подготовлен)

**Config**: `configs/experiments/exp026_sft_sentiment_150m.yaml`

**Статус**: Подготовлен, ожидает запуска

Аналогичный exp025, но на модели 150M. Цель — сравнить качество классификации маленькой и большой модели на одной задаче.

### EXP-028: Llama 1.08B GQA — 16.2B Kazakh+FineWeb-Edu Tokens (деградирован)

**Config**: `configs/experiments/exp028_llama_1b.yaml`

**Статус**: Деградирован — QK-Norm веса потеряны при конвертации в HF формат

**HuggingFace**: [stukenov/sozkz-core-llama-1b-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-1b-kk-base-v1)

**Документация**: [docs/exp028_llama_1b.md](docs/exp028_llama_1b.md)

#### Архитектура

Крупнейшая dense казахская модель в серии SozKZ. Архитектурные улучшения по сравнению с exp023 (600M):

- **GQA** (4 KV heads vs 16) — 4× сжатие KV-кэша, HF-совместимо
- **QK-Norm** (RMSNorm на Q,K) — стабилизация attention, **НЕ** HF-совместимо
- **Z-loss** (1e-4) — регуляризация логитов (PaLM/Gemma), HF-совместимо
- **Embedding scaling** (×sqrt(2048)) — Gemma трюк, **НЕ** HF-совместимо

```
Параметры:       1.08B
Скрытый размер:  2048
Слои:            22
Головы:          16 (4 KV heads — GQA)
Intermediate:    5504 (SwiGLU)
Vocab:           50,257
```

#### Данные

~16.2B токенов (70% KK / 30% ENKK):
- KK: `stukenov/sozkz-corpus-tokenized-kk-llama50k-v4` (~8.4B токенов, очищено от CJK, мусора, encoding artifacts)
- ENKK: `stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v2` (~7.8B токенов)
- Chinchilla ratio: 15:1

#### Обучение

| Метрика | Значение |
|---------|----------|
| Hardware | 8×H100 SXM 80GB (RunPod) |
| Время | 9.53 часа |
| Стоимость | ~$205 |
| Final loss | 2.15 |
| Val BPB | 0.8127 |
| Throughput | ~577K tok/s |

#### Критическая проблема: потеря QK-Norm весов

При конвертации в HF формат `load_state_dict(strict=False)` тихо отбросил QK-Norm веса, т.к. `LlamaForCausalLM` не поддерживает QK-Norm. Pod был уничтожен после загрузки — оригинальный чекпоинт потерян навсегда.

**Попытки восстановления:**
1. Embedding scaling fix (baked into weights) — частичное улучшение
2. QK-Norm weight=ones — модель сколлапсировала
3. Fine-tune 5,632 QK-Norm параметров — loss не сходился (~60)

**Вывод**: QK-Norm веса невосстановимы. Для перетренировки (~$205) нужно убрать QK-Norm и embedding scaling, оставив только GQA + Z-loss.

**Урок**: Всегда тестировать полный round-trip (train → save → HF convert → load → inference) перед дорогими запусками. Никогда не уничтожать pod до верификации модели с HF Hub.

### EXP-033: Kazakh Language Model Benchmark — Base Model Selection

**Цель**: Определить лучшую базовую модель для адаптации к казахскому языку (tokenizer extension + continual pretraining). Протестировать максимальное количество open-source моделей по трём языкам (kk, ru, en) и выбрать оптимальное соотношение качества к размеру.

**Методология**:
- **Perplexity**: 10 казахских, 5 русских, 4 английских предложения (Wikipedia-style factual текст)
- **Tokenizer fertility**: tokens/word для казахского vs английского (ниже = лучше)
- **Kazakh character coverage**: наличие специфичных символов (ә, ғ, қ, ң, ө, ү, ұ, і, һ) в vocab
- **Generation quality**: 10 казахских промптов, max_new_tokens=80, temperature=0.7
- **Hardware**: 2× NVIDIA A10 (казну-сервер) + RunPod H100 80GB SXM (retry)
- **Software**: transformers 5.1–5.5, torch 2.10, bf16

**Скоринг** (100 баллов):
- PPL казахский: 35 баллов (ниже PPL = больше баллов)
- PPL русский: 10 баллов
- PPL английский: 5 баллов
- Tokenizer efficiency: 10 баллов (fertility ratio kk/en ближе к 1.0)
- Generation quality: 10 баллов (экспертная оценка связности)
- License: 10 баллов (Apache/MIT = 10, Llama = 8, ограниченная = 5)
- Efficiency: 20 баллов (лучший PPL при меньшем размере)

#### Полная таблица результатов (28 моделей)

| # | Model | Params | PPL kk | PPL ru | PPL en | Fert kk | kk/en |
|---|-------|--------|--------|--------|--------|---------|-------|
| 1 | **mGPT-1.3B-kazakh** | 1.4B | **2.0** | 21.6 | 18.2 | 4.73 | 4.26x |
| 2 | mGPT-13B | 13.1B | 2.4 | 8.8 | 10.1 | 4.73 | 4.26x |
| 3 | mGPT-1.3B | 1.4B | 2.7 | 9.3 | 13.1 | 4.73 | 4.26x |
| 4 | LLaMA-3.1-8B | 8.0B | 3.2 | 6.0 | 7.0 | 4.00 | 3.39x |
| 5 | LLaMA-3.2-3B | 3.2B | 3.5 | 6.4 | 6.8 | 4.17 | — |
| 6 | Qwen2.5-14B | 14.8B | 4.3 | 3.7 | 6.5 | 4.82 | 4.34x |
| 7 | LLaMA-3.2-1B | 1.2B | 4.4 | 7.5 | 6.8 | 4.17 | — |
| 8 | Gemma-2-9B | 9.2B | 4.8 | 9.6 | 8.2 | 3.00 | 2.48x |
| 9 | Qwen2.5-7B | 7.6B | 4.9 | 3.2 | 6.6 | 4.27 | 3.53x |
| 10 | Qwen2.5-3B | 3.1B | 6.7 | 3.4 | 5.9 | 4.33 | — |
| 11 | SmolLM2-1.7B | 1.7B | 7.0 | 2.6 | 6.2 | 7.50 | — |
| 12 | Gemma-2-2B | 2.6B | 7.2 | 11.4 | 8.2 | 3.00 | — |
| 13 | Yi-1.5-9B | 8.8B | 7.5 | 3.5 | 7.0 | 5.05 | 3.98x |
| 14 | GPT-NeoX-20B | 20.6B | 8.2 | 3.9 | 8.0 | 6.36 | 5.48x |
| 15 | Yi-1.5-6B | 6.1B | 8.9 | 4.4 | 6.8 | 5.05 | 3.98x |
| 16 | Qwen2.5-1.5B | 1.5B | 9.0 | 4.3 | 6.4 | 4.33 | — |
| 17 | OLMo-2-7B | 7.3B | 9.6 | 2.9 | 5.4 | 5.23 | 4.43x |
| 18 | StableLM-2-1.6B | 1.6B | 10.1 | 2.4 | 6.0 | 5.67 | — |
| 19 | SozKZ-Llama-600M | 0.6B | 10.2 | 11.7 | 8.9 | 1.17 | 1.17x |
| 20 | Baichuan2-13B | 13.9B | 11.6 | 2.8 | 8.0 | 5.36 | 4.83x |
| 21 | SmolLM2-360M | 0.4B | 11.8 | 4.3 | 6.8 | 7.50 | — |
| 22 | Mistral-7B-v0.3 | 7.2B | 13.9 | 5.7 | 5.9 | 4.05 | 3.12x |
| 23 | SozKZ-TinyLlama-kk | 1.1B | 15.4 | 19.4 | 8.9 | 1.17 | — |
| 24 | Qwen2.5-0.5B | 0.5B | 17.8 | 5.7 | 7.7 | 4.33 | — |
| 25 | TinyLlama-1.1B | 1.1B | 26.2 | 11.4 | 11.4 | 4.00 | — |
| 26 | Bloom-7B | 7.1B | 35.3 | 9.9 | 10.9 | 4.91 | 4.42x |
| 27 | CroissantLLM-1.3B | 1.3B | 36.1 | 3.8 | 8.6 | 5.45 | 4.50x |
| 28 | XGLM-7.5B | 7.5B | 2633.7 | 17.1 | 10.4 | 1.73 | 1.43x |

#### Не протестированные модели (ошибки загрузки)

| Model | Причина |
|-------|---------|
| Falcon-7B | PPL=inf (ошибка loss вычисления в transformers 5.5) |
| InternLM2.5-7B | PPL=inf (trust_remote_code конфликт) |
| Baichuan2-7B | PPL=inf (trust_remote_code конфликт) |
| InternLM2-20B | Ошибка загрузки ('type' key error) |
| Falcon2-11B | Ошибка загрузки ('type' key error) |
| Aya-23-8B | Gated repo (нет доступа) |
| Jais-13B | Gated repo (нет доступа) |
| MPT-7B | Repo удалён с HuggingFace |
| DBRX-base | Repo удалён/приватный |
| Mixtral-8x7B | Не поместился на диск (93GB bf16) |
| Qwen1.5-MoE-A2.7B | Не поместился на диск |
| OLMoE-1B-7B | Не поместился на диск |
| DeepSeek-MoE-16B | Не поместился на диск |

#### Примеры генерации (Top-5 по PPL kk)

**mGPT-1.3B-kazakh** (PPL kk=2.0) — связный, грамотный казахский:
> **Қазақстан — бұл** әлемдегі ең керемет дамыған, бай мемлекеттердің бірі. Алматыны басып алу туралы шешім Астанада 2004 жылдың 20 желтоқсанында жарияланды
> **Наурыз мейрамы — қазақ халқының** ежелгі көктем мерекесi. Бұл шаруашылдығында да, дарын иесi ретiнде де аса байлықты таппаған жыршысы мен ойшылы Абылай

**LLaMA-3.1-8B** (PPL kk=3.2) — хороший казахский, но переключается на английский:
> **Қазақстан — бұл** дәуірді үшінші кезеңде туындаған мемлекет. Оның жерінің аумағы Батыс Сібірген де едәуір ірі болып табылды.
> **Наурыз мейрамы — қазақ халқының** ежелгі салт-дәстүрлі, көне атаулармен "Айдай", "Сауын" деп аталуы да ескертілсе керек.

**Gemma-2-9B** (PPL kk=4.8) — грамотный казахский, длинные связные ответы:
> **Абай Құнанбайұлы — ұлы** ағартушы, қоғам өмірінің көрегені және қатал сыншысы. Оның шығармаларындағы тарихи тұлғаларды жазғанда сол кезеңдегі оқиғалар мен тарихты толық түсінуге мүмкіндік береді.

**Bloom-7B** (PPL kk=35.3) — MediaWiki мусор вместо связного текста:
> **Қазақстан — бұл** қалыптау менің тексерімен, [[{{MediaWiki:Policy-url}}|саңғы жардамаларын]] осы нҰтикін.

#### Ключевые выводы

1. **mGPT-семейство — лидер по казахскому**: ai-forever тренировал mGPT на 60+ языках включая казахский. Казахская fine-tuned версия (mGPT-1.3B-kazakh) показывает PPL=2.0 всего при 1.4B параметрах. Но русский (21.6) и английский (18.2) деградировали — модель "забыла" другие языки.

2. **LLaMA-3 — лучший баланс**: LLaMA-3.1-8B (PPL kk=3.2) и LLaMA-3.2-3B (PPL kk=3.5) показывают сильный казахский при сохранении хорошего русского и английского. Идеальная база для мультиязычной адаптации.

3. **Qwen2.5 — сильный мультиязычный**: Qwen2.5-14B (PPL kk=4.3) и Qwen2.5-7B (PPL kk=4.9) — лучший русский (3.2–3.7) среди всех моделей при хорошем казахском. Apache 2.0 лицензия.

4. **Gemma-2 — лучший tokenizer**: Единственная семья с 18/18 казахских символов в vocab и fertility 3.0 (vs 4.0–5.0 у остальных). Но PPL kk хуже (4.8–7.2).

5. **Размер не решает всё**: mGPT-1.3B (2.7) лучше GPT-NeoX-20B (8.2) в 14 раз меньшим размером. Данные важнее архитектуры для low-resource языков.

6. **Tokenizer fertility не коррелирует с PPL**: XGLM имеет лучшую fertility (1.43x kk/en) но худший PPL (2633.7). Bloom имеет средний tokenizer но плохой PPL (35.3). Качество pretraining данных важнее tokenizer efficiency.

7. **"English-first" модели слабы**: Mistral-7B (PPL kk=13.9), OLMo-2 (9.6), StableLM-2 (10.1) — модели без многоязычного pretraining значительно хуже.

#### Рекомендации для exp034 (Kazakh adaptation)

| Задача | Рекомендуемая база | Почему |
|--------|-------------------|--------|
| Kazakh-only модель ≤3B | LLaMA-3.2-3B | PPL kk=3.5, 3.2B, Llama license |
| Kazakh-only модель ≤1B | LLaMA-3.2-1B | PPL kk=4.4, 1.2B, хороший старт |
| Мультиязычная kk+ru+en | Qwen2.5-7B | PPL kk=4.9, ru=3.2, en=6.6, Apache 2.0 |
| Максимальное качество kk | mGPT-1.3B-kazakh + дообучение | PPL kk=2.0, но нужно восстановить ru/en |
| Максимальное качество при бюджете | LLaMA-3.1-8B | PPL kk=3.2, 8B, один H100 |

**Стоимость бенчмарка**: ~$2.69/hr × 0.5hr = ~$1.35 (RunPod H100 SXM) + бесплатно на казну-сервере.

**Скрипты**: `autoresearch/exp033_benchmark_kazakh.py`, `exp033_benchmark_10b.py`, `exp033_benchmark_20b.py`, `exp033_retry_bench.py`
**Результаты**: `docs/datasets/exp033_*_results.json`, `docs/datasets/exp033_mgpt_kazakh.json`

### EXP-034: mGPT-1.3B Kazakh Instruct v1

**Цель**: SFT fine-tune лучшей казахской base модели (mGPT-1.3B-kazakh, PPL kk=2.0) для следования инструкциям на казахском языке.

**Данные**: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2) — 52,201 instruction-output пар (MIT).

**Метод**: Full fine-tune, bf16, prompt format `### Нұсқаулық: / ### Жауап:`, 3 эпохи.

| Parameter | Value |
|-----------|-------|
| Base model | ai-forever/mGPT-1.3B-kazakh |
| Params | 1.42B |
| Batch (effective) | 32 |
| LR | 2e-5, cosine |
| Epochs | 3 |
| Final eval loss | **0.919** |
| Runtime | 79 min (H100 SXM) |
| Cost | ~$3.50 |

**Результат**: [stukenov/sozkz-mgpt-1.3b-kk-instruct-v1](https://huggingface.co/stukenov/sozkz-mgpt-1.3b-kk-instruct-v1)

Модель генерирует связный казахский по инструкциям, но фактическая точность ограничена размером (1.4B). Инструкт-версии mGPT до этого не существовало.

### EXP-035: mGPT-1.3B Kazakh-Russian Translation v1

**Цель**: Создать двунаправленную модель перевода kk↔ru на базе mGPT-1.3B-kazakh.

**Данные**: [stukenov/ekitil-parallel-kkru-v2](https://huggingface.co/datasets/stukenov/ekitil-parallel-kkru-v2) (kk-ru config) — 500K параллельных пар из 5.1M доступных. Источники: WMT19 crawl, KazParC, OPUS corpora.

**Метод**: 2-этапный pipeline:

**Stage 1 — Continual Pretrain (full fine-tune)**:
- Формат: `[KK>RU] source [SEP] target</s>` и `[RU>KK] source [SEP] target</s>`
- 1M примеров (500K x 2 направления), 1 эпоха
- BS=32, grad_accum=2, lr=2e-5, cosine, gradient checkpointing

**Stage 2 — SFT LoRA (instruction format)**:
- Формат: `### Аудар [KK>RU]:\nsource\n### Аударма:\ntarget</s>`
- LoRA r=32, alpha=64, targets: c_attn/c_proj/c_fc (25M trainable из 1.42B)
- 1M примеров, 1 эпоха, BS=32, grad_accum=2, lr=2e-5

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Eval loss | 1.054 | **0.896** |
| Train loss | ~1.05 | ~0.96 |
| Steps | 15,594 | 15,594 |
| Runtime | ~3.4h | ~2.9h |

**Общая стоимость**: ~$17 (6.3h H100 SXM @ $2.69/hr)

**Результат**: [stukenov/sozkz-mgpt-1.3b-translate-kkru-v1](https://huggingface.co/stukenov/sozkz-mgpt-1.3b-translate-kkru-v1)

#### Примеры перевода

| Направление | Вход | Выход |
|-------------|------|-------|
| KK→RU | Қазақстан Республикасы — Орталық Азиядағы мемлекет. | Республика Казахстан - государство в Центральной Азии. |
| RU→KK | Казахстан — государство в Центральной Азии. | Қазақстанда Орталық Азияда мемлекет орналасады. |
| KK→RU | Абай Құнанбайұлы — ұлы қазақ ақыны, ағартушы, ойшыл. | Академик Абая Кунанбаева - выдающийся казахский поэт и просветитель. |
| RU→KK | Образование является важнейшим направлением политики. | Білім мемлекеттік саясаттың мақсаты болып табылады. |

#### Выводы

1. **Decoder-only модель умеет переводить**: mGPT-1.3B после 2-этапного обучения даёт связные переводы kk↔ru
2. **Stage 1 критичен**: без continual pretrain на параллельных данных модель не знает маппинг между языками
3. **LoRA SFT работает**: 25M trainable params (1.8% от 1.42B) достаточно для обучения формату
4. **Проблема повторов**: на длинных выходах модель уходит в repetition loop — нужен repetition_penalty≥1.2
5. **Фактические ошибки**: имена и факты иногда искажаются (Абай→"Академик Абая Кунанбаева")
6. **Следующие шаги**: eval на FLORES+ для сравнения с deepvk/kazRush (BLEU 16.2) и NLLB

**Скрипты**: `autoresearch/exp035_translate_pretrain.py`, `exp035_translate_sft_lora.py`, `exp035_run.sh`

### EXP-036: Qwen2.5 500M KK Base — 7B токенов на morphbpe-100k

**Config**: `configs/experiments/exp036_qwen_500m.yaml`

**Статус**: ✅ Завершён. Модель обучена, загружена на HF, верифицирована на отдельном инстансе (kaznu A10).

**HuggingFace**: [stukenov/sozkz-core-qwen-500m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-qwen-500m-kk-base-v1)

#### Цель

Обучить dense казахскую LM на точной архитектуре **Qwen2.5-0.5B** (все фичи: GQA 7:1, RoPE θ=1M, Q/K/V bias, RMSNorm, tied embeddings) с нуля, используя морфологически-ориентированный токенизатор `sozkz-morphbpe-100k-kk-v1`. Проверить гипотезу: больший словарь + более зрелая архитектура дают лучший результат на тех же данных, даже при меньшем числе параметров.

#### Архитектура

```
Параметры:       447.5M
Скрытый размер:  896
Слои:            24
Головы:          14 (GQA 7:1, 2 KV heads)
Head dim:        64
Intermediate:    4864 (SwiGLU)
Vocab:           100,000 (morphbpe)
RoPE theta:      1,000,000
RMSNorm eps:     1e-6
Tied embeddings: yes
Attn bias Q/K/V: yes (Qwen2 specific)
Max pos emb:     32,768
```

Ровно параметры Qwen2.5-0.5B, но vocab изменён с 151k (multilingual) на 100k (казахский morphbpe).

#### Данные

- **Источник**: [`kz-transformers/multidomain-kazakh-dataset`](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) — ~80M строк, в основном казахский с небольшой долей русского.
- **Токенизация**: 16 процессов на kaznu (2×A10), block_size=2048, через `scripts/tokenizer/tokenize_dataset.py`. Результат: [`stukenov/sozkz-corpus-tokenized-kk-morphbpe100k-v1`](https://huggingface.co/datasets/stukenov/sozkz-corpus-tokenized-kk-morphbpe100k-v1) — ~3.58B токенов.
- **Итого для обучения**: ~7.0B токенов (несколько проходов по корпусу). Chinchilla ratio ≈ 15.6:1.

**Incident**: первая попытка токенизации на датасете `sozkz-corpus-clean-kk-text-v4` дала только 0.53B токенов (13M строк × ~40 токенов/строку), потому что тот датасет состоит из коротких отрывков. Переключились на `multidomain-kazakh-dataset` где длинные связные тексты — получили 3.58B токенов. Параллельно пришлось освободить ~400GB на kaznu (старые hub/datasets caches).

#### Обучение

| Параметр | Значение |
|---|---|
| Hardware | 8× H100 SXM 80GB (RunPod, spot community) |
| Cost | ~$45 (2.07h @ $21.5/hr) |
| Время | 2.07 часа |
| Total steps | 26,702 |
| Batch size (per GPU) | 16 |
| Grad accumulation | 2 |
| Seq length | 1024 |
| Effective batch | 262,144 tokens/step |
| Peak LR | 4e-4 (cosine, 500 warmup) |
| Optimizer | AdamW (β=[0.9, 0.95], wd=0.1) |
| Precision | bfloat16 AMP |
| Z-loss | 1e-4 |
| Throughput | **938K tokens/s** |
| Peak VRAM | 42.1 GB / 80 GB |

**Оптимизация скорости**: начальный бенчмарк с `seq_len=2048, BS=32` дал OOM → снизили до `seq_len=2048, BS=8, accum=4` и получили всего 288K tok/s. Далее уменьшили `seq_len` с 2048 до 1024, подняли `BS` до 16: итоговая скорость **938K tok/s** (~3.2× от первой рабочей версии). Для 24-слойного модели линейные слои доминируют во FLOPs, поэтому ускорение от seq_len=2048→1024 оказалось не 2×, как ожидалось, а только ~1.2×.

#### Результаты

| Метрика | Значение |
|---|---|
| Train loss (final) | **2.040** |
| Val BPB | **0.474** |
| Tokens seen | 7.0B |

#### Сравнение с другими sozkz-core KK base моделями

| Model | Params | Vocab | Tokens | Wall-clock | Val BPB ↓ | Train loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| [llama-300m](https://huggingface.co/stukenov/sozkz-core-llama-300m-kk-base-v1) | 325M | 50k | 9.0B | 1.82h 8×H100 | 0.781 | 2.848 |
| **qwen-500m (exp036)** | **447M** | **100k** | **7.0B** | **2.07h 8×H100** | **0.474** | **2.040** |
| [llama-600m](https://huggingface.co/stukenov/sozkz-core-llama-600m-kk-base-v1) | 587M | 50k | 9.0B | 5.90h 4×H100 | 0.756 | 2.713 |
| [llama-1b](https://huggingface.co/stukenov/sozkz-core-llama-1b-kk-base-v1) | 1078M | 50k | 16.2B | 9.53h 8×H100 | 0.673 | 2.636 |

500M Qwen2 **обогнала** 1B Llama по val BPB (0.474 vs 0.673) при вдвое меньшем размере и 2.3× меньшем объёме данных. Причины:

1. **100k morphbpe vocab** — морфемная токенизация казахского даёт системное преимущество. Один токен = одна значимая морфема, что упрощает задачу моделирования для агглютинативного языка.
2. **Qwen2.5 архитектура** — более зрелая, чем Llama-1 (GQA 7:1, RoPE θ=1M, Q/K/V bias).
3. **24 слоя vs 22** — больше глубины при том же бюджете параметров.

**Caveat**: `val_bpb` = `nats_per_token / log(2) / bytes_per_token`. Константа `bytes_per_token` разная для разных токенизаторов (6.5 для morphbpe-100k, ~4.5 для llama-50k). Метрика by design tokenizer-invariant, но точность зависит от корректности константы. Если реальное среднее для morphbpe меньше 6.5, реальная bpb будет выше (но всё равно ниже llama-1b).

#### Примеры генерации

`temperature=0.8, top_k=50, repetition_penalty=1.15`, NVIDIA A10:

```
PROMPT:  Қазақстан Республикасы
OUTPUT:  Қазақстан Республикасы Үкіметінің қаулысы 2014 жылғы 15 шілде
N 213. Қазақстан Республикасының Үкіметі ҚАУЛЫ ЕТЕДІ...

PROMPT:  Абай Құнанбайұлы
OUTPUT:  Абай Құнанбайұлы . Абайдың әкесі Құнанбайдан үш жас үлкен...
Абай 1845 жылы 28 шілдеде қазіргі Шығыс Қазақстан облысының Абай
ауданында туған...

PROMPT:  Ғылым мен білімнің
OUTPUT:  Ғылым мен білімнің ғылымының дамуына қосқан үлесі үшін Құрмет
грамотасы, ҚР БҒМ-нің Құрмет грамоталары берілді...
```

Модель уверенно различает регистры: юридический, биографический, новостной, образовательный. Грамматика (падежи, окончания, пунктуация) безошибочная.

#### Инциденты и уроки

1. **Broken `publish_checkpoint.py` background uploads**: скрипт был оставлен от exp028 с размерами Llama 1B (2048/50257/22 layers) и молча падал при всех промежуточных HF upload во время обучения. Обнаружено только в самом конце по ошибкам размерностей в логе. Финальный upload шёл через отдельный правильный `upload_500m_to_hf.py` (из `run_500m_training.sh`) — прошёл корректно. **Урок**: межэкспериментальные leftover-скрипты нужно проверять отдельно или переименовывать на скрипт per-experiment.

2. **4 дубликата pods на RunPod**: при первой попытке launch скрипт создал 4 pods из-за повторных запусков (не проверял существующие). Итого ~$76.5/hr горело, пока не заметили. Остановили 3 дубликата вручную. **Урок**: `launch_500m_pod.py` должен делать idempotency check перед созданием.

3. **Инференс из промежуточного чекпоинта**: посреди обучения запустили `quick_infer_500m.py` на step_19000 (71%, loss 2.127). Грамматика уже идеальная, модель различает регистры. Это дало уверенность, что полная тренировка не провалится.

4. **Верификация на отдельном инстансе (exp028 lesson applied)**: после завершения training НЕ уничтожали pod до верификации. Загрузили модель на kaznu (A10) из HF Hub, запустили `verify_exp036_hf.py` — всё загрузилось и генерирует корректно. Только после этого pod был уничтожен.

#### Скрипты

- `autoresearch/prepare_500m.py` — download + memmap writer
- `autoresearch/train_500m_ddp.py` — Qwen2 model + DDP training loop
- `autoresearch/upload_500m_to_hf.py` — HF conversion + strict=True + inference verification
- `autoresearch/run_500m_training.sh` — полный pipeline с Telegram notifications
- `autoresearch/launch_500m_pod.py` — RunPod provisioning с fallback GPU configs
- `ansible/run_tokenize_exp036.yml` — токенизация на kaznu

#### Выводы

1. **Vocab size matters for agglutinative languages**: переход с 50k BPE на 100k morphbpe дал бо́льший буст, чем масштабирование модели с 500M до 1B на llama-50k.
2. **Qwen2.5 арх работает лучше Llama-1 на том же бюджете**: GQA + Q/K/V bias + θ=1M RoPE — хорошая комбинация для scratch training.
3. **7B токенов достаточно для 500M модели** на казахском (Chinchilla ratio 15.6:1).
4. **Data pipeline — узкое место**: токенизация заняла сопоставимое время с training (несколько часов на kaznu vs 2.07h на 8×H100).
5. **Full round-trip verification работает**: пайплайн с `strict=True` + inference verification + external-instance verification предотвратил повторение exp028 инцидента.
