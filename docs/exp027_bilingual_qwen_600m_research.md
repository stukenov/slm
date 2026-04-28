# Research: exp027 — Bilingual Kazakh-Russian 600M on Qwen3 Architecture

## 1. Цель

Обучить двуязычную модель ~600M параметров на архитектуре Qwen3, способную работать с казахским и русским языками, включая перевод в обе стороны. В отличие от предыдущих экспериментов (Llama, только казахский), здесь:
- Архитектура Qwen3 (лучшая на малых масштабах по бенчмаркам 2025)
- Двуязычный корпус: казахский + русский + параллельные пары
- Частичная адаптация токенизатора (не с нуля)
- Чиншилла-оптимальный объём данных

---

## 2. Почему Qwen3, а не Llama

| Фактор | Llama 3.x | Qwen3 |
|--------|-----------|-------|
| Минимальная модель | 1B (8B основная) | **0.6B** — ближе к нашему таргету |
| GQA | Да | Да (num_kv_heads < num_heads) |
| Vocab size | 128K | **151K** — шире покрытие языков |
| Кириллица в токенизаторе | Слабая (латино-центричный) | **Хорошая** — целенаправленно мультиязычный |
| Активация | SwiGLU | SwiGLU |
| RoPE | Да | Да |
| Контекст | 8K-128K | 32K-128K |
| Thinking mode | Нет | **Да** (hybrid thinking в Qwen3) |

**Ключевое преимущество**: Qwen3 токенизатор уже хорошо покрывает кириллицу (русский, казахский). Vocab 151K включает тысячи кириллических токенов. Это означает меньше адаптации и лучшую fertility.

### 2.1 Архитектура Qwen3-0.6B (референс)

Из релиза Qwen3 (апрель 2025):

```
hidden_size: 1024
num_hidden_layers: 28
num_attention_heads: 16
num_key_value_heads: 8          # GQA ratio 2:1
intermediate_size: 3072         # 3× hidden
vocab_size: 151,936
max_position_embeddings: 40,960
tie_word_embeddings: true
head_dim: 64
rope_theta: 1,000,000
```

Параметры: ~0.6B (без учёт embedding, с tied embeddings).

### 2.2 Наша целевая архитектура (~600M)

Берём Qwen3-0.6B как прямой референс и масштабируем:

**Вариант A — Близко к Qwen3-0.6B (рекомендуемый):**
```yaml
model_config:
  model_type: qwen3           # или qwen2 если transformers не поддерживает qwen3
  hidden_size: 1024
  num_hidden_layers: 28
  num_attention_heads: 16
  num_key_value_heads: 8
  intermediate_size: 3072
  vocab_size: 64000            # наш адаптированный токенизатор (см. секцию 4)
  max_position_embeddings: 4096
  tie_word_embeddings: true
  rope_theta: 1000000
```

Примерный подсчёт параметров:
- Embedding: 64K × 1024 = 65M (tied, считается 1 раз)
- Attention per layer: 1024 × (1024 + 512 + 512 + 1024) = ~3.1M × 28 = 87M
- MLP per layer: 1024 × 3072 × 3 (gate+up+down) = ~9.4M × 28 = 264M
- LayerNorms + misc: ~1M
- **Итого: ~417M** (маловато)

**Вариант B — Увеличенный до 600M:**
```yaml
model_config:
  model_type: qwen3
  hidden_size: 1280
  num_hidden_layers: 28
  num_attention_heads: 20
  num_key_value_heads: 4        # GQA ratio 5:1 (экономия параметров)
  intermediate_size: 4480       # 3.5× hidden
  vocab_size: 64000
  max_position_embeddings: 4096
  tie_word_embeddings: true
  rope_theta: 1000000
```

Подсчёт:
- Embedding: 64K × 1280 = 82M
- Attention per layer: 1280×1280 + 1280×256 + 1280×256 + 1280×1280 = ~3.6M × 28 = 100M
- MLP per layer: (1280 × 4480 × 2 + 4480 × 1280) = ~17.2M × 28 = 481M
- **Итого: ~580-600M** ✓

**Вариант C — Меньше vocab, больше модель:**
```yaml
vocab_size: 32000              # компактный токенизатор
hidden_size: 1280
# ... остальное как B
# Итого: ~560M + экономия на embedding
```

**Рекомендация**: Вариант B. GQA 5:1 экономит параметры на attention, позволяя вложить больше в MLP (где основное "знание"). Модель инициализируется случайно (from scratch), от Qwen3 берём только архитектурные решения.

---

## 3. Данные: Chinchilla-оптимальный бюджет

### 3.1 Бюджет токенов

По Chinchilla scaling law: оптимально ~20 токенов на параметр.

| Модель | Параметры | Оптимум токенов | Наш план |
|--------|-----------|-----------------|----------|
| 600M | 600M | **12B** | 12B (6B kk + 5B ru + 1B parallel) |

### 3.2 Казахский корпус (~6B токенов) — ФИЛЬТРАЦИЯ

Текущий датасет: `kz-transformers/multidomain-kazakh-dataset` (23.6M samples ≈ 9B+ токенов).

**Проблема**: датасет содержит мусор (дубликаты, machine-translated garbage, HTML артефакты, нерелевантный текст).

**План фильтрации**:

1. **Дедупликация** — MinHash (n=5, threshold=0.8) через `datatrove` или `text-dedup`
   - Ожидаемое сокращение: 20-40%

2. **Language ID** — отсеять тексты не на казахском
   - Инструменты: `fasttext` lid.176.bin, `lingua-py`
   - Порог: confidence > 0.8 для казахского

3. **Quality filtering**:
   - Удалить документы < 50 символов
   - Удалить документы с > 30% не-казахских символов (латиница, спецсимволы)
   - Удалить документы с perplexity выше порога (KenLM на чистом казахском)
   - Удалить HTML/boilerplate (regex-паттерны)

4. **Porn/toxic filtering** — если есть классификатор для казахского

5. **Domain balance** — сэмплировать пропорционально:
   - Новости: 30%
   - Книги/литература: 25%
   - Wikipedia: 15%
   - Образование/наука: 15%
   - Законодательство: 10%
   - Разное: 5%

**Целевой результат**: ~6B чистых токенов из исходных ~9B+

### 3.3 Русский корпус (~5B токенов)

Доступные высококачественные датасеты:

| Датасет | Размер | Качество | Примечание |
|---------|--------|----------|------------|
| `IlyaGusev/saiga_scored` | ~2B tok | Высокое | Уже отфильтрованный |
| `ai-forever/MERA` | Бенчмарк | — | Для eval, не для train |
| `CulturaX` (ru subset) | Огромный | Среднее | Нужна фильтрация |
| `mc4` (ru) | ~300GB | Среднее | Web crawl, нужна фильтрация |
| `wikipedia` (ru) | ~3B tok | Высокое | Чистый |
| `FineWeb-Edu` (ru) | Есть subset | Высокое | Образовательный контент |
| Flibusta/Lib.Rus.Ec dumps | ~10B+ tok | Высокое | Художественная литература, copyright вопросы |

**Рекомендуемый микс для 5B токенов:**
- Wikipedia (ru): ~2B tok (чистый, энциклопедический)
- FineWeb-Edu (ru subset): ~1.5B tok (образовательный)
- CulturaX (ru, filtered): ~1.5B tok (web, отфильтрованный)

**Фильтрация русского**: те же шаги что и для казахского (dedup, quality, length).

### 3.4 Параллельный корпус kk↔ru (~1B токенов)

Цель: модель должна понимать и переводить в обе стороны.

| Источник | Пар | Качество | Доступность |
|----------|-----|----------|-------------|
| **OPUS/CCAligned** (kk-ru) | ~2-5M | Среднее | Открытый |
| **OPUS/WikiMatrix** (kk-ru) | ~100-300K | Хорошее | Открытый |
| **Tatoeba** (kk-ru) | ~5-10K | Высокое | Маленький |
| **OPUS/GNOME/KDE** (kk-ru) | ~50-100K | UI-специфичный | Открытый |
| **kazNERD/kazParC** | ~100K+ | Хорошее | Академический |
| **WMT** | Нет kk-ru напрямую | — | — |
| **Tilmash** (kk-en, через en pivot) | ~500K+ | Среднее | HuggingFace |

**Формат параллельных данных для обучения:**
```
<|kk|> Қазақ тіліндегі сөйлем. <|ru|> Предложение на казахском языке.
```
или ChatML-формат:
```
<|im_start|>system\nTranslate from Kazakh to Russian<|im_end|>
<|im_start|>user\nҚазақ тіліндегі сөйлем.<|im_end|>
<|im_start|>assistant\nПредложение на казахском языке.<|im_end|>
```

**Важно**: параллельные данные перемешиваются с моноязычными, а не даются отдельным этапом. Оптимальная доля параллельных данных: 5-10% от общего объёма.

---

## 4. Токенизатор: частичная адаптация

### 4.1 Подход: обучаем с нуля на kk+ru корпусе

Токенизатор обучается **полностью с нуля** — не адаптация Qwen, не pruning. Причины:
- Полный контроль над vocab: каждый токен работает на наши языки
- Нет мёртвого веса (CJK, арабский, тайский из Qwen — это ~60-70% их vocab)
- Embedding matrix компактнее → больше параметров в модель
- Опыт уже есть: sozkz-core-gpt2-50k работает хорошо для казахского

### 4.2 План обучения токенизатора

**Алгоритм**: ByteLevel BPE (как в GPT-2/Llama/Qwen)

**Обучающий корпус** (сбалансированный сэмпл):
- 50% казахский текст (из отфильтрованного корпуса)
- 45% русский текст (из собранного корпуса)
- 5% параллельные пары (чтобы общие токены для обоих языков выучились)

**Варианты vocab size:**

| Vocab | Embedding (×1280) | Fertility kk | Fertility ru | Рекомендация |
|-------|-------------------|-------------|-------------|--------------|
| 32K | 41M | ~2.0 | ~1.8 | Компактно, но может быть тесно для 2 языков |
| 48K | 61M | ~1.7 | ~1.5 | Хороший компромисс |
| 64K | 82M | ~1.5 | ~1.3 | Оптимально для двуязычной модели |
| 96K | 123M | ~1.3 | ~1.2 | Много параметров уходит в embedding |

**Рекомендация**: **64K vocab** — достаточно места для двух кириллических языков + базовую латиницу/цифры/пунктуацию. При tied embeddings это 82M параметров (из 600M), что приемлемо.

**Шаги:**

1. Собрать обучающий корпус: ~5M документов (2.5M kk + 2.25M ru + 0.25M parallel)
2. Обучить BPE через `tokenizers` library:
   ```python
   from tokenizers import Tokenizer, models, trainers, pre_tokenizers
   tokenizer = Tokenizer(models.BPE())
   tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
   trainer = trainers.BpeTrainer(
       vocab_size=64000,
       special_tokens=["<|endoftext|>", "<|padding|>", "<|kk|>", "<|ru|>", "<|translate|>"],
       min_frequency=100,
   )
   ```
3. Добавить спецтокены для языковых тегов и задач
4. Замерить fertility на held-out тексте
5. Если fertility > 2.5 на одном из языков — пересмотреть пропорции корпуса

### 4.3 Метрика: fertility (токенов на слово)

Целевые значения:
- Казахский: < 2.0 tokens/word (текущий sozkz-50k даёт ~1.8)
- Русский: < 1.8 tokens/word
- Смешанный kk+ru текст: < 2.0 tokens/word

Сравниваем с бейзлайнами:
- sozkz-core-gpt2-50k (текущий, только kk): ~1.8 kk, ~3.5+ ru
- Qwen3 tokenizer (151K): ~2.5 kk (оценка), ~1.3 ru
- Llama3 tokenizer (128K): ~3.0+ kk, ~2.0 ru

---

## 5. Стратегия обучения

### 5.1 Подход: From scratch на архитектуре Qwen3

Берём только архитектуру Qwen3 (config), инициализируем случайно, обучаем с нуля на 12B токенов.

**Почему from scratch, а не continue pretrain:**
- Свой токенизатор (64K) несовместим с весами Qwen3 (151K vocab) — embedding layer не переиспользуется
- 12B токенов — это Chinchilla-оптимум для 600M, достаточно для cold start
- Полный контроль: модель учит ровно то, что нам нужно, без борьбы с forgetting
- Предыдущие эксперименты (exp013-exp023) подтвердили: from scratch работает хорошо на этих масштабах

### 5.2 Data mixing schedule

```
Phase 1 (0-30% steps): Warmup
  - 60% русский (знакомый для модели)
  - 30% казахский
  - 10% параллельный

Phase 2 (30-80% steps): Core training
  - 40% казахский (усиление)
  - 40% русский
  - 20% параллельный

Phase 3 (80-100% steps): Annealing
  - 45% казахский
  - 45% русский
  - 10% параллельный (уменьшаем чтобы не overfit на шаблон)
```

### 5.3 Гиперпараметры

```yaml
learning_rate: 3e-4            # стандартный LR для from scratch (как exp023)
warmup_steps: 1000
lr_scheduler_type: cosine
weight_decay: 0.1
max_grad_norm: 1.0
bf16: true
per_device_train_batch_size: 16
block_size: 2048               # увеличиваем контекст (было 1024)
num_train_epochs: 1            # 1 эпоха по 12B токенов
```

### 5.4 Hardware estimate

| GPU | Время (12B tok) | Стоимость |
|-----|-----------------|-----------|
| 8× H100 SXM | ~6-8h | ~$80-110 |
| 4× A100 80GB | ~12-16h | ~$60-80 |
| 1× H100 | ~24-30h | ~$25-35 |

---

## 6. Evaluation plan

| Бенчмарк | Задача | Языки |
|-----------|--------|-------|
| BPB (bits per byte) | Perplexity | kk, ru |
| KazMCQA | Multiple-choice QA | kk |
| Belebele | Reading comprehension | kk, ru |
| SIB-200 | Topic classification | kk, ru |
| FLORES-200 | Translation quality (BLEU) | kk↔ru |
| **NEW**: XNLI | NLI | ru |

Ключевая новая метрика: **BLEU на FLORES kk↔ru** — измерить качество перевода.

---

## 7. Checklist / Порядок действий

1. [ ] **Данные kk**: отфильтровать multidomain-kazakh-dataset (dedup, langid, quality) → ~6B tok
2. [ ] **Данные ru**: собрать ~5B токенов (Wikipedia + FineWeb-Edu + CulturaX)
3. [ ] **Данные parallel**: собрать kk↔ru пары из OPUS/CCAligned/WikiMatrix → ~1B tok
4. [ ] **Токенизатор**: обучить BPE 64K с нуля на сбалансированном kk+ru корпусе
5. [ ] **Токенизатор**: замерить fertility, сравнить с sozkz-50k и Qwen3
6. [ ] **Данные**: токенизировать всё новым токенизатором, залить на HF
7. [ ] **Модель**: инициализировать Qwen3 архитектуру ~600M from scratch (random init)
8. [ ] **Конфиг**: создать `configs/experiments/exp027_bilingual_qwen_600m.yaml`
9. [ ] **Обучение**: запустить на 8×H100 (vast.ai), 12B токенов
10. [ ] **Eval**: прогнать все бенчмарки + FLORES BLEU kk↔ru
11. [ ] **Publish**: `stukenov/ekitil-base-600m-kkru-v1`

---

## 8. Риски и mitigation

| Риск | Вероятность | Mitigation |
|------|-------------|------------|
| Qwen3 не поддерживается в transformers | Средняя | Проверить `transformers` version, fallback на Qwen2.5 |
| Catastrophic forgetting русского | Средняя | Data mixing + малый LR + eval на русских бенчмарках |
| Параллельных данных мало | Высокая | Back-translation: генерировать пары через существующие MT модели |
| Fertility Qwen tokenizer плохая на kk | Низкая | Если > 3.0, делать extend; если > 4.0, train from scratch |
| 12B токенов недостаточно | Низкая | Chinchilla говорит достаточно; можем добавить repeat |

---

## 9. Naming (EkiTil)

**Бренд**: EkiTil (Екі Тіл = Два Языка) — билингвальные казахско-русские модели.
**HF Collection**: `stukenov/ekitil` — "EkiTil: Bilingual Kazakh-Russian Language Models"

- Токенизатор: `stukenov/ekitil-vocab-bpe-64k-kkru-v1`
- Датасет kk: `stukenov/ekitil-corpus-filtered-kk-v1`
- Датасет ru: `stukenov/ekitil-corpus-ru-v1`
- Датасет parallel: `stukenov/ekitil-corpus-parallel-kkru-v1`
- Датасет tokenized: `stukenov/ekitil-corpus-tokenized-kkru-v1`
- Base model (позже): `stukenov/ekitil-base-600m-kkru-v1`
- Chat model (позже): `stukenov/ekitil-chat-600m-kkru-v1`
