# EkiTil: Bilingual Kazakh-Russian Data Pipeline — Progress Report

**Дата**: 2026-03-26
**Бренд**: EkiTil (Екі Тіл = Два Языка)
**Цель**: подготовить данные для обучения двуязычной kk-ru модели ~600M на архитектуре Qwen3

---

## 1. Что было сделано

### 1.1 Аннотация казахского датасета (Phase 1)

**Исходный датасет**: `kz-transformers/multidomain-kazakh-dataset` (24.9M документов)

**Что сделали**:
- Разбили каждый документ на предложения
- Для каждого предложения запустили fasttext langdetect (язык + confidence)
- Присвоили уникальный `doc_id` (MD5 от пересобранного текста)
- Добавили поля: `source`, `domain`, `detected_lang`, `lang_confidence`, `num_chars`, `is_kk`

**Результат**: `stukenov/ekitil-corpus-annotated-kk-v1`
- **121.9M предложений** из 24.9M документов
- kk = 60.1M (49.3%)
- ru = 57.0M (46.8%) — неожиданно много русского уже в "казахском" датасете
- en = 2.3M (1.9%)
- другие языки = 2.5M (2.0%)

**Время**: ~2.1 часа на kaznu (2×A10, 1TB RAM)

### 1.2 Параллельный корпус kk↔ru (Phase 2)

**Источники**:
- Helsinki-NLP/opus-100 (kk-ru subset)
- Dauren-Nur/kaz_rus_parallel_corpora_KAZNU

**Результат**: `stukenov/ekitil-corpus-parallel-kkru-v1`
- ~135K параллельных пар (kk ↔ ru)
- Sentence-level aligned

**Время**: ~3 минуты

### 1.3 Токенизатор BPE 64K (Phase 3)

**Подход**: обучение ByteLevel BPE с нуля на сбалансированном kk+ru корпусе

**Обучающий корпус**: 4.9M предложений
- 50% казахский (2.5M, reservoir sampling из 60M)
- 45% русский (2.25M, reservoir sampling из 57M)
- 5% параллельные пары (135K)

**Специальные токены**:
| ID | Токен | Назначение |
|----|-------|-----------|
| 0 | `<\|endoftext\|>` | Конец документа |
| 1 | `<\|padding\|>` | Padding |
| 2 | `<\|startoftext\|>` | Начало текста |
| 3 | `<\|kk\|>` | Kazakh language tag |
| 4 | `<\|ru\|>` | Russian language tag |
| 5 | `<\|translate\|>` | Translation task marker |

**Результат**: `stukenov/ekitil-vocab-bpe-64k-kkru-v1`
- Vocab size: **64,000**
- Fertility: **1.56 tokens/word** (mixed kk+ru)
- BPE training: ~2 минуты (после 2.5ч sampling)

**Время**: ~2.5 часа (sampling) + 2 мин (BPE) на kaznu

### 1.4 Токенизация датасета — sentence-level (Phase 4, v1)

**Первая версия** — токенизация по предложениям (не оптимально для обучения):

**Результат**: `stukenov/ekitil-corpus-tokenized-kkru-v1` (v1, sentence-level)
- kk tokens: 1.33B
- ru tokens: 1.46B
- parallel tokens: 7.2M
- **Итого: ~2.8B токенов**
- **1,265,538 блоков × 2048**

**Проблема**: модель видит только отдельные предложения, не связные документы. Переделываем на document-level (Phase 4 v2).

**Время**: ~10 часов (streaming с HF Hub) на kaznu

### 1.5 Токенизация — document-level (Phase 4, v2) — В ПРОЦЕССЕ

**Что меняется**:
- Предложения пересобираются обратно в документы по `doc_id`
- Фильтрация на уровне документов (kk_ratio >= 0.7 или ru_ratio >= 0.7)
- Параллельные пары как один текст: `<|kk|> kk_text <|translate|> <|ru|> ru_text <eos>`
- Multiprocessing tokenization для скорости
- Запуск на RunPod с быстрой сетью

---

## 2. Артефакты на HuggingFace

| Repo | Тип | Статус | Размер |
|------|-----|--------|--------|
| `stukenov/ekitil-corpus-annotated-kk-v1` | Dataset | ✅ | 121.9M rows |
| `stukenov/ekitil-corpus-parallel-kkru-v1` | Dataset | ✅ | ~270K rows |
| `stukenov/ekitil-vocab-bpe-64k-kkru-v1` | Tokenizer | ✅ | 64K vocab |
| `stukenov/ekitil-corpus-tokenized-kkru-v1` | Dataset | ⏳ v2 | ~2.8B tokens (v1) |

---

## 3. Ключевые находки

1. **Русского в "казахском" датасете почти столько же сколько казахского** (57M vs 60M предложений). Не нужно отдельно собирать русский корпус — он уже есть.

2. **Fertility 1.56 tok/word** — хороший результат для двуязычного BPE. Для сравнения: наш предыдущий kk-only токенизатор (50K) давал ~1.8 tok/word.

3. **2.8B токенов** — для 600M модели по Chinchilla (20:1) нужно ~12B. Ratio 4.7:1 — датасет undertrained. Варианты:
   - Уменьшить модель до ~140M
   - Добавить больше данных (CulturaX ru, FineWeb-Edu)
   - Тренировать несколько эпох (2-4 эпохи до ~12B effective tokens)
   - Принять suboptimal ratio (многие модели тренируются с ratio < 20:1)

---

## 4. Инфраструктура

| Ресурс | Использование |
|--------|--------------|
| kaznu (2×A10, 1TB RAM) | Аннотация, токенизатор, sentence-level tokenization |
| RunPod RTX 3090 | Попытка аннотации (OOM), дата prep |
| RunPod (новый) | Document-level tokenization (в процессе) |

**Стоимость RunPod**: ~$3-5 (2 пода, несколько часов)

---

## 5. Скрипты

| Скрипт | Назначение |
|--------|-----------|
| `scripts/exp027/annotate_kk_dataset.py` | Phase 1: аннотация + langdetect |
| `scripts/exp027/add_russian_and_parallel.py` | Phase 2: parallel корпус |
| `scripts/exp027/train_tokenizer.py` | Phase 3: BPE 64K tokenizer |
| `scripts/exp027/tokenize_dataset.py` | Phase 4 v1: sentence-level tokenization |
| `scripts/exp027/tokenize_documents.py` | Phase 4 v2: document-level tokenization |
| `scripts/exp027/launch_runpod.py` | RunPod pod management |
| `ansible/exp027_inventory.ini` | Ansible inventory for RunPod |
| `ansible/exp027_deploy.yml` | Ansible deploy playbook |
| `ansible/exp027_status.yml` | Ansible status playbook |

---

## 6. Следующие шаги

1. ⏳ Завершить document-level tokenization (Phase 4 v2)
2. Решить вопрос с объёмом данных (2.8B vs 12B Chinchilla)
3. Определить финальную архитектуру модели (Qwen3 ~600M или уменьшить)
4. Создать конфиг `configs/experiments/exp027_bilingual_qwen_600m.yaml`
5. Запустить обучение на vast.ai / RunPod (8×H100)
6. Evaluation: BPB, MC QA, Belebele, FLORES BLEU kk↔ru
7. SFT/Instruct fine-tune
