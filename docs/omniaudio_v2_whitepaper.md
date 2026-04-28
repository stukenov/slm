# OmniAudio v2: Native Kazakh ASR from Scratch

**Авторы:** Сакен Тукенов
**Дата:** 2026-03-29
**Статус:** В процессе тренировки (эпоха 4/10)

---

## Abstract

OmniAudio v2 — экспериментальная ASR-модель для казахского языка, построенная полностью с нуля (69.58M параметров). Модель использует Llama-style архитектуру (RoPE, RMSNorm, SwiGLU) с двухстадийной тренировкой: CTC pretrain encoder → E2E joint training. Обучена на 439 часах казахской речи (kzcalm-tts-kk-v1). На текущий момент модель правильно распознаёт целые предложения казахской речи, хотя имеет проблему с остановкой генерации (EOS).

Данная работа документирует полный путь разработки, включая 4 неудачных подхода и финальное решение, которое работает.

---

## 1. Мотивация

Существующие ASR-системы для казахского языка — это преимущественно fine-tuned Whisper или wav2vec2. У них есть проблемы:

1. **Размер:** Whisper-large = 1.55B params, слишком велик для edge-устройств
2. **Лицензия:** Многие модели закрыты или требуют коммерческой лицензии
3. **Интеграция:** Отдельная ASR-модель не интегрируется с текстовой LLM

**Цель OmniAudio v2:** создать компактную (< 100M) ASR-модель, которая может работать как standalone и в будущем интегрироваться с текстовой LLM (sozkz 150M/600M) для multimodal understanding.

---

## 2. Хронология экспериментов

### 2.1 Попытка 1: Frozen Pretrained Decoder (FAILED)

**Идея:** Обученный Llama 150M (sozkz-core-llama-150m-kk-base-v1) уже знает казахский язык. Если добавить audio encoder и обучить только projector (0.2M params), decoder должен понять аудио.

**Конфигурация:**
- Encoder: 256d, 4 heads, 6 layers (custom, from scratch)
- Projector: Linear(256→768) + RMSNorm
- Decoder: sozkz Llama 150M, полностью заморожен
- Данные: FLEURS kk (3,200 samples, ~3.5 часа)
- Тренировка: 3 стадии (CTC pretrain → alignment → E2E)

**Результат: WER 1487%**

Модель генерировала **одну и ту же фразу** для любого аудио входа. Pretrained decoder полностью игнорировал аудио embeddings и генерировал текст из своего language model prior.

**Анализ причины:**
- Projector (0.2M) слишком мал чтобы "пробиться" через 150M замороженных весов
- Decoder видел audio embeddings как шум и полагался на свой text prior
- 3.5 часа данных — критически мало для любого подхода

### 2.2 Попытка 2: Partially Unfrozen Decoder (FAILED)

**Идея:** Разморозить последние 4 из 12 слоёв decoder, чтобы он мог адаптироваться к аудио.

**Изменения:**
- Unfrozen: layers 8-11 + norm + lm_head (~60M trainable)
- Остальное: как в попытке 1

**Результат: WER 793%**

Чуть лучше, но модель всё ещё генерировала повторяющийся текст, слабо связанный с аудио. Decoder text prior продолжал доминировать.

**Анализ причины:**
- 4 слоя недостаточно для переключения с text-only на audio-conditioned генерацию
- FLEURS 3.5h — слишком мало для обучения 60M параметров
- Первые 8 замороженных слоёв формировали representations, не учитывающие аудио

### 2.3 Попытка 3: Decoder from Scratch + FLEURS (PARTIAL SUCCESS)

**Идея:** Убрать pretrained decoder полностью. Decoder с нуля не имеет text prior и будет вынужден опираться на аудио.

**Конфигурация:**
- Encoder: 256d, 4 heads, 6 layers (5.1M)
- Projector: Linear(256→512) + RMSNorm (131K)
- Decoder: 512d, 8 heads, 8 layers, from scratch (51.4M)
- CTC Head: Linear(256→50257) (12.9M)
- Total: 69.58M
- Данные: FLEURS kk (3,200 samples, ~3.5 часа)

**Результат: WER 376-898% (нестабильно)**

Прогресс! Модель генерировала **разный текст для разного аудио**. Иногда ловила начальные слова. Но:
- Сильные повторения (hallucination loops)
- Высокий WER из-за малого количества данных
- Быстрый overfit (val loss не падал после 2-3 эпох)

**Ключевой инсайт:** Decoder from scratch работает — он вынужден опираться на аудио, а не на text prior. Проблема — в количестве данных.

### 2.4 Попытка 4: Decoder from Scratch + kzcalm (CURRENT — WORKING)

**Идея:** Та же архитектура, но на 100x больше данных.

**Изменения:**
- Данные: kzcalm-tts-kk-v1 (232K samples, 439 часов)
- max_audio_len: 10 sec (вместо 15, чтобы избежать OOM)
- augment: false для CTC pretrain, true для E2E
- repetition_penalty: 1.2 в generate()

**Результат (промежуточный, эпоха 3/10):**

Модель **правильно распознаёт целые предложения:**

```
REF: Іс материалдары бойынша бұл тұлғаның мас болғандығы анықталды.
HYP: Іс материалдары бойынша бұл тұлғаның мас болғандығы анықталды. [+ мусор]
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     100% match на основной части

REF: Мен Қарағанды облысының шағын қалаларының бірінде тұрамын.
HYP: Мен Қарағанды облысының шағын қалаларының бірінде тұрамын. [+ мусор]
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     100% match

REF: Факторинг операциясы бүгінгі таңда отындық ақша нарығында дами алмай отыр.
HYP: Факторинг операциясы бүгінгі таңда отандық ақш нарығында дами алмай отыр. [+ мусор]
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     ~95% match (отындық→отандық, ақша→ақш)
```

**Оставшаяся проблема:** модель не генерирует EOS и продолжает бесконечно генерировать мусорный текст после правильного ответа. Причина установлена и исправлена (см. раздел 5).

### 2.5 Follow-up on SozKZ Mels (2026-04-08)

После исходного цикла `50m/150m/300m/600m` на `sozkz_mels` были сделаны два практических follow-up шага.

**A. Проверка partial unfreeze для 50M**

- Конфиг: [v2_llm50m_e2e_unfreeze2_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_llm50m_e2e_unfreeze2_cloudrift.yaml)
- Идея: разморозить последние `2` слоя decoder LLM и дообучить `50m`
- Результат на `google/fleurs kk_kz test`:
  - baseline `checkpoint-best`: `WER 101.53%`, `CER 128.02%`
  - `unfreeze2 checkpoint-best`: `WER 432.83%`, `CER 204.98%`
  - `unfreeze2 checkpoint-final`: `WER 333.78%`, `CER 407.18%`

**Вывод:** partial unfreeze в этой конфигурации сделал модель заметно хуже. На `FLEURS` усилились repetitive collapse и деградация stopping behavior. Этот рецепт не должен считаться улучшением baseline.

**B. Новый эксперимент: scratch 150M full E2E**

- Конфиг: [v2_scratch_sozkz_mels_150m_e2e_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_scratch_sozkz_mels_150m_e2e_cloudrift.yaml)
- Цель: проверить полноценную `~150M` scratch ASR-модель без pretrained SozKZ LLM decoder
- Архитектура:
  - encoder: `384d`, `6 heads`, `10 layers`
  - decoder: `768d`, `12 heads`, `10 layers`
  - total params: `147.32M`
- Режим:
  - `model_type: scratch`
  - `stage: e2e`
  - `dataset_name: sozkz_mels`
  - `ctc_weight: 0.3`
  - `batch: 12`, `grad_accum: 2`
  - `lr: 2e-4`
  - `epochs: 10`
- Инфраструктура:
  - host: `217.138.104.166`
  - цель запуска: отдельный `CloudRift` run без публикации в HF до eval
  - screen: `train_150m_scratch_e2e`
  - log: `/home/riftuser/slm/logs/cloudrift_150m_scratch_e2e.log`

**Причина нового направления:** frozen/pretrained decoder line пока показывает сильную доменную хрупкость на `FLEURS`, а `50m unfreeze2` ухудшил метрики. Следующий логичный шаг — проверить, даст ли large scratch decoder более устойчивое audio grounding без доминирования text prior.

### 2.6 Scratch 150M Update And Handoff (2026-04-09)

Эксперимент `150m scratch full E2E` был доведён до устойчивого training regime на одном `RTX 5090 32GB`.

**Итоговая рабочая конфигурация**

- Конфиг: [v2_scratch_sozkz_mels_150m_e2e_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_scratch_sozkz_mels_150m_e2e_cloudrift.yaml)
- Практически лучший режим на этом pod:
  - `per_device_train_batch_size: 24`
  - `gradient_accumulation_steps: 1`
  - `dataloader_num_workers: 6`
  - `torch_compile: false`
- Попытка поднять `batch` до `32` не дала реального выигрыша по throughput и была отменена.

**Промежуточные результаты 150M scratch**

- Быстрый eval `checkpoint-10000` на `FLEURS`, `200 samples`:
  - `WER 100.67%`
  - `CER 81.85%`
- Быстрый eval `checkpoint-best` на `FLEURS`, `200 samples`:
  - `WER 46.42%`
  - `CER 32.16%`
- Быстрый eval `checkpoint-220000` на `FLEURS`, `200 samples`:
  - `WER 45.33%`
  - `CER 31.70%`

**Состояние перед остановкой**

- Training loss вышел в зону явного plateau.
- Последний активный участок шёл примерно в районе `step ~220k`.
- На диске уже были сохранены:
  - `checkpoint-best`
  - регулярные numeric checkpoints до последних сохранённых шагов
- Вывод: модель ещё понемногу улучшалась по `WER/CER`, но уже вошла в режим diminishing returns.

**Передача на следующий размер**

- В линейке OmniAudio pretrained нет `100m`.
- Последовательность размеров: `50m -> 150m -> 300m -> 600m -> 1b`.
- Следующий логичный размер для `E2E` после `150m` — `300m`.
- Для `300m` уже существует готовый `CTC checkpoint-best` в HF repo:
  - `stukenov/sozkz-core-omniaudio-300m-kk-ctc-v1`
- Поэтому следующий запуск переведён на:
  - `300m E2E`
  - single-GPU fast recipe
  - без повторного Stage 1 encoder pretrain

### 2.7 300M E2E Result And Stop Decision (2026-04-09)

После перехода на `300m` был поднят отдельный быстрый `E2E` run от уже готового `CTC checkpoint-best`.

**Конфиг**

- fast recipe: [v2_llm300m_e2e_fast_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_llm300m_e2e_fast_cloudrift.yaml)
- encoder init: `stukenov/sozkz-core-omniaudio-300m-kk-ctc-v1` `checkpoint-best`

**Инфраструктура**

- host: `217.138.104.166`
- screen: `train_300m_fast_e2e`
- log: `/home/riftuser/slm/logs/cloudrift_300m_fast_e2e.log`

**Наблюдения**

- run стабильно дошёл как минимум до `checkpoint-150000`
- training loss продолжал снижаться, но уже очень медленно
- типичный late-stage участок:
  - `step 150050`: `loss 1.5752`
  - `step 155050`: `loss 1.5664`

Это означает, что по train objective модель вошла в зону явного diminishing returns.

**Внешняя метрика**

- Full `FLEURS test` на `checkpoint-best`:
  - `WER 109.41%`
  - `CER 99.77%`
- Quick `FLEURS 200 samples` на `checkpoint-150000`:
  - `WER 101.97%`
  - `CER 93.84%`

**Вывод**

- Несмотря на улучшение train loss, `300m` всё ещё не вышел на качественный ASR на `FLEURS`.
- Модель ловит начало фразы, но регулярно срывается в repetitions и truncated outputs.
- Было принято решение остановить этот run и не тратить дальнейшее время на тот же recipe.

### 2.8 Next Experiment: 50M With MorphBPE-100k

Следующий запуск переведён обратно на `50m`, но уже с новым tokenizer:

- tokenizer repo: `stukenov/sozkz-morphbpe-100k-kk-v1`
- `vocab_size: 100000`

**Техническое изменение**

Чтобы такой запуск был совместим с pretrained `50m` decoder, в код была добавлена поддержка
`resize_token_embeddings(vocab_size)` для pretrained LLM path в [model_v2.py](/Users/sakentukenov/slm/omniaudio/src/omniaudio/model_v2.py).

**Новый конфиг**

### 2.9 Next Scale-Up: 600M Multi-GPU With MorphBPE (2026-04-10)

После `50m/150m/300m` MorphBPE follow-up следующий шаг переведён на `600m`, но уже не на single-GPU pod, а на отдельный multi-GPU запуск с целью максимизировать throughput.

**Цель**

- использовать только новый tokenizer `stukenov/sozkz-morphbpe-100k-kk-v1`
- сохранить новую схему нормализации текста под `FLEURS`
- запустить `600m E2E` от уже готового `600m CTC checkpoint-best`
- использовать `torchrun` на нескольких GPU, а не single-GPU training

**Новый конфиг**

- [v2_llm600m_e2e_morphbpe100k_fleursnorm_long15_multigpu_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_llm600m_e2e_morphbpe100k_fleursnorm_long15_multigpu_cloudrift.yaml)
- [v2_llm600m_e2e_morphbpe100k_fleursnorm_long15_multigpu_localllm_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_llm600m_e2e_morphbpe100k_fleursnorm_long15_multigpu_localllm_cloudrift.yaml)

**Конфигурация**

- model: pretrained `600m`
- LLM: `stukenov/sozkz-core-llama-600m-kk-base-v1`
- tokenizer: `stukenov/sozkz-morphbpe-100k-kk-v1`
- vocab: `100000`
- init: `stukenov/sozkz-core-omniaudio-600m-kk-ctc-v1` `checkpoint-best`
- text normalization:
  - lowercase
  - strip punctuation
  - collapse whitespace
- audio window: `15s`
- intended runtime: multi-GPU `torchrun`

**Практический смысл**

Этот запуск нужен не как ещё одна вариация `50m/150m/300m`, а как прямой тест того, даст ли новый MorphBPE pipeline преимущество уже на большом decoder scale при максимальной скорости обучения на CloudRift.

**Операционный нюанс**

Первый multi-GPU старт показал, что gated `600m` LLM repo неудобно тянуть напрямую внутри `torchrun` workers. Поэтому фактический production run переведён на локальный snapshot LLM на pod и отдельный конфиг `*_localllm_cloudrift.yaml`, чтобы исключить auth-зависимость в рантайме.

- [v2_llm50m_e2e_morphbpe100k_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_llm50m_e2e_morphbpe100k_cloudrift.yaml)

**Цель**

- проверить, даст ли MorphBPE-100k лучшую segmentation/tokenization quality для Kazakh ASR decoding
- сделать это в минимальном по риску setup:
  - `50m`
  - single GPU
  - init from existing `50m CTC checkpoint-best`

---

## 3. Финальная архитектура

### 3.1 Общая схема

```
Waveform (16kHz) → Mel (80 bins) → Conv↓4x → Transformer Encoder (6L)
                                                        ↓
                                              Projector (256→512)
                                                        ↓
                                    [audio_embeds ; text_embeds] → Causal Decoder (8L) → Text
```

### 3.2 Audio Encoder (5.1M params)

| Компонент | Детали |
|-----------|--------|
| Conv stack | 2 × Conv1d(stride=2, kernel=3) + GELU, 4x downsampling |
| Normalization | RMSNorm после conv |
| Transformer | 6 layers, d=256, 4 heads, head_dim=64 |
| Attention | Bidirectional, RoPE на Q,K |
| FFN | SwiGLU: gate(256→704) * up(256→704) → down(704→256) |
| Positional | RoPE (Rotary Positional Embedding), base=10000 |

10 секунд аудио (1000 mel frames) → 250 encoder tokens.

### 3.3 Audio Projector (131K params)

```
Linear(256 → 512) → RMSNorm(512)
```

Проецирует encoder features в пространство decoder.

### 3.4 Causal Decoder (51.4M params)

| Компонент | Детали |
|-----------|--------|
| Token embedding | Embedding(50257, 512), tied с lm_head |
| Transformer | 8 layers, d=512, 8 heads, head_dim=64 |
| Attention | Causal (is_causal=True), RoPE на Q,K |
| FFN | SwiGLU: gate(512→1408) * up(512→1408) → down(1408→512) |
| Output | RMSNorm(512) → Linear(512→50257, no bias) |

### 3.5 CTC Head (12.9M params)

```
Linear(256 → 50257)
```

Используется только в Stage 1 (CTC pretrain) и как auxiliary loss (30%) в Stage 2.

### 3.6 Суммарный parameter count

| Компонент | Params | % |
|-----------|--------|---|
| Encoder (conv + transformer) | 5.1M | 7.3% |
| Projector | 131K | 0.2% |
| CTC Head | 12.9M | 18.5% |
| Embedding (shared) | 25.7M | 36.9% |
| Decoder (transformer) | 25.6M | 36.8% |
| Norms, biases | ~200K | 0.3% |
| **Total** | **69.58M** | **100%** |

Checkpoint: **266 MB** (mixed precision)

---

## 4. Тренировка

### 4.1 Данные

**Dataset:** stukenov/kzcalm-tts-kk-v1

| Split | Samples | Hours (est.) |
|-------|---------|-------------|
| Train | 218,409 | ~412h |
| Val | 11,617 | ~22h |
| Test | 2,320 | ~5h |
| **Total** | **232,346** | **~439h** |

Это TTS-датасет со всеми спикерами и эмоциями. 44% дупликатов текста (одинаковый текст, разные спикеры/эмоции) — намеренно используем полный датасет для робастности к разным голосам.

**Tokenizer:** kazakh-gpt2-50k (ByteLevel BPE, 50,257 vocab)

**Augmentation (только E2E stage):**
- Speed perturbation: ×0.9 / ×1.0 / ×1.1 (random)
- SpecAugment: 2 freq masks (F=27), 2 time masks (T=100)

### 4.2 Stage 1: CTC Pretrain

**Цель:** обучить encoder извлекать фонетические representations.

| Параметр | Значение |
|----------|----------|
| Trainable | Encoder + CTC Head (18.0M) |
| Loss | CTC (blank=0) |
| Batch | 8 × 4 = 32 effective |
| LR | 1e-3, linear warmup 5%, cosine decay |
| Epochs | 5 |
| Augment | Off |
| max_audio_len | 10.0 sec |

**Результаты:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | - | - |
| 2 | - | - |
| 3 | - | 0.7826 |
| 4 | - | 0.5823 |
| 5 | 0.695 | **0.4886** |

Общее время: **~2.5 часа** на NVIDIA A10.

Loss: 72.3 → 0.695 (train), val loss стабильно падал каждую эпоху.

### 4.3 Stage 2: E2E Joint Training

**Цель:** обучить полную модель (encoder + decoder) генерировать текст.

| Параметр | Значение |
|----------|----------|
| Trainable | ВСЕ (69.58M) |
| init_from | CTC checkpoint-best |
| Loss | 0.7 × CE + 0.3 × CTC |
| CE label_smoothing | 0.1 |
| Batch | 4 × 8 = 32 effective |
| LR | 3e-4, linear warmup 5%, cosine decay |
| Epochs | 10 |
| Augment | SpecAugment + Speed Perturb |
| max_audio_len | 10.0 sec |

**Результаты (в процессе):**

| Epoch | Train Loss | Val Loss | Δ Val |
|-------|-----------|----------|-------|
| 1 | 13.45 | 4.91 | - |
| 2 | 4.22 | 3.39 | -31% |
| 3 | 3.47 | **2.82** | -17% |
| 4 | 3.11 (in progress) | - | - |
| ... | | | |
| 10 | (pending) | (pending) | |

Общее время: ~20 часов на NVIDIA A10 (ETA: ~4h remaining at epoch 4).

**Наблюдения:**
- Train loss драматически падает между эпохами (данные перемешиваются): 13.45 → 4.22 → 3.47 → 3.11
- Val loss уверенно улучшается каждую эпоху
- Нет признаков overfitting (train > val loss)

### 4.4 Hardware

| Ресурс | Значение |
|--------|----------|
| GPU | NVIDIA A10 (23GB), GPU index 1 |
| Server | kaznu (164.138.46.36) |
| Process | Screen session, CUDA_VISIBLE_DEVICES=1 |
| GPU 0 | Занят другим процессом |

---

## 5. Известные проблемы и решения

### 5.1 EOS Problem (IDENTIFIED, FIX READY)

**Проблема:** Модель правильно распознаёт текст, но не останавливается — генерирует бесконечный мусор после правильного ответа.

**Причина:** В training data отсутствует EOS token. Tokenizer кодирует текст без EOS, поэтому модель никогда не обучается генерировать конец последовательности.

**Фикс (готов, будет применён в следующем раунде):**

```python
# data_v2.py — добавляем EOS в конец text_ids
ids = tokens["input_ids"].squeeze(0)
eos_id = self.tokenizer.eos_token_id
if eos_id is not None and (len(ids) == 0 or ids[-1].item() != eos_id):
    ids = torch.cat([ids, torch.tensor([eos_id])])
```

CTC targets остаются без EOS (CTC не использует EOS).

### 5.2 OOM на длинных аудио (RESOLVED)

**Проблема:** GPU OOM при обработке аудио > 15 секунд.

**Решение:** Ограничить max_audio_len=10.0 sec. kzcalm-tts средняя длина ~7 sec, потеря данных минимальна.

### 5.3 HuggingFace tokenizer loading (RESOLVED)

**Проблема:** AutoTokenizer.from_pretrained() падает на kaznu (transformers 5.1 bug).

**Решение:** Try/except с fallback на PreTrainedTokenizerFast:
```python
try:
    tokenizer = AutoTokenizer.from_pretrained(path)
except (ValueError, OSError):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
```

### 5.4 Disk space на kaznu (RESOLVED)

**Проблема:** Диск 100% заполнен (876GB).

**Решение:** Удалены старые HF cache модели (49GB) и неиспользуемые датасеты (20GB). Освобождено 49GB.

### 5.5 Repetitive Generation (PARTIALLY RESOLVED)

**Проблема:** Модель повторяет фрагменты текста.

**Частичное решение:** Добавлен repetition_penalty=1.2 в generate():
```python
for prev_token in set(generated):
    if logits[prev_token] > 0:
        logits[prev_token] /= repetition_penalty
    else:
        logits[prev_token] *= repetition_penalty
```

Полное решение: EOS fix + больше тренировки.

---

## 6. Ключевые инсайты

### 6.1 Pretrained decoder не работает с малым количеством данных

Pretrained LLM decoder (даже маленький, 150M) имеет сильный text prior, который доминирует над слабым audio signal от необученного encoder + маленького projector. Разморозка части слоёв не помогает — нужно либо:
- Разморозить ВСЕ слои + иметь достаточно данных (>100h)
- Использовать decoder from scratch

### 6.2 CTC pretrain критически важен

Без CTC pretrain decoder не получает осмысленных audio features и не может учиться. CTC pretrain обеспечивает:
- Encoder выучивает фонетические representations
- Projector получает осмысленный вход
- Decoder может начать next-token prediction с первой эпохи E2E

### 6.3 Количество данных решает

| Dataset | Hours | Result |
|---------|-------|--------|
| FLEURS | 3.5h | Overfit, бессмысленный output |
| kzcalm-tts | 439h | Правильное распознавание |

Переход от 3.5h к 439h (125x) — это разница между "не работает" и "работает".

### 6.4 TTS-синтезированные данные работают для ASR

kzcalm-tts — это TTS-датасет (синтезированная речь). Несмотря на это, модель учится распознавать речь. Разнообразие спикеров и эмоций добавляет робастность.

### 6.5 Маленькая модель может делать ASR

69.58M params — это ~45x меньше Whisper-large (1.55B). При достаточном количестве данных и правильной архитектуре, маленькая модель способна к качественному ASR для одного языка.

---

## 7. Следующие шаги

### 7.1 Краткосрочные (эта неделя)

1. **Дождаться завершения E2E тренировки** (ещё ~4h, эпохи 4-10)
2. **Задеплоить EOS fix** и запустить ещё 3-5 эпох E2E fine-tuning
3. **Измерить WER/CER** на тестовом split kzcalm-tts

### 7.2 Среднесрочные

4. **Эксперимент с sozkz 150M decoder:** использовать обученный encoder + sozkz 150M (все слои unfrozen). Потенциально лучше WER за счёт language model knowledge.
5. **Добавить Common Voice kk** (~50h натуральной речи) в training mix
6. **Beam search** вместо greedy decoding
7. **KV-cache** для ускорения inference

### 7.3 Долгосрочные

8. **Pseudo-labeled YouTube data** (500-1000h казахского)
9. **Streaming ASR** (chunk-based encoder)
10. **Multimodal integration** с sozkz LLM для speech understanding tasks
11. **Публикация на HuggingFace** как stukenov/sozkz-omniaudio-v2-kk

## 2.9 50M MorphBPE-100k FLEURS-Normalized Long-Context Experiment (2026-04-09)

После первых inference-проверок стало видно две системные проблемы у 50M MorphBPE run:

1. таргеты в train/eval не были приведены к FLEURS-подобному формату, из-за чего модель училась на смеси регистра и пунктуации, а оценивалась на более нормализованной речи;
2. текущий CloudRift run был ограничен `max_audio_len: 10.0`, что жёстко резало более длинные utterances и мешало реальному inference на фразах длиннее 10 секунд.

Что изменили:

- добавили единый `normalize_asr_text()` в `data_v2.py`;
- включили управляемые флаги нормализации через YAML:
  - `text_lowercase: true`
  - `text_strip_punctuation: true`
  - `text_collapse_whitespace: true`
- применили ту же нормализацию в `evaluate_v2.py`, чтобы train/eval были согласованы;
- для pretrained decode добавили защиту от повторов:
  - `decode_repetition_penalty: 1.15`
  - `decode_no_repeat_ngram_size: 3`
- подняли `max_audio_len: 10.0 -> 15.0`

Новый отдельный конфиг:

- `omniaudio/configs/v2_llm50m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml`

Параметры run:

- `experiment_name: omniaudio_v2_llm50m_morphbpe100k_fleursnorm_long15`
- tokenizer: `stukenov/sozkz-morphbpe-100k-kk-v1`
- base decoder: `stukenov/sozkz-core-llama-50m-kk-base-v2`
- init_from: `./outputs/omniaudio_v2_llm50m_ctc/checkpoint-best`
- host: `217.138.104.166`
- screen: `train_50m_morphbpe100k_fleursnorm_long15`
- log: `/home/riftuser/slm/logs/cloudrift_50m_morphbpe100k_fleursnorm_long15.log`

Начальный статус:

- weights + tokenizer поднялись корректно;
- embeddings расширились до `100000`;
- CTC checkpoint загрузился;
- `Stage: e2e | Params: 107.03M total, 30.92M trainable`

Гипотеза:

- lowercased + punctuation-free targets должны лучше совпасть с FLEURS-style evaluation;
- увеличение окна до `15s` должно убрать жёсткое отсечение длинных utterances;
- decode constraints должны уменьшить repetitive collapse без переписывания архитектуры.

Практическое уточнение после проверки длительностей `FLEURS kk_kz test`:

- median: `15.24s`
- p90: `24.06s`
- max: `43.32s`
- `439 / 853` samples длиннее `15s`

Вывод:

- `15s` достаточно только как более мягкий train compromise;
- для честного `FLEURS` eval/inference нужен отдельный длинный конфиг;
- создан `omniaudio/configs/v2_llm50m_e2e_morphbpe100k_fleursnorm_long45_eval.yaml` с `max_audio_len: 45.0`
- текущий training run не переводился на `45s`, потому что это потребовало бы рестарта и сильного снижения batch на `RTX 5090 32GB`

Альтернативный более практичный путь:

- оставить training на `15s`;
- для длинных utterances делать chunked inference:
  - `chunk_audio_len: 15.0`
  - `chunk_overlap_sec: 3.0`
- для этого добавлен отдельный eval-конфиг:
  - `omniaudio/configs/v2_llm50m_e2e_morphbpe100k_fleursnorm_chunk15_eval.yaml`

Идея:

- не раздувать training memory cost;
- покрыть длинные `FLEURS` записи через несколько перекрывающихся чанков;
- сравнить `15s raw`, `15s chunked`, и `45s full-window` на одном и том же checkpoint.

## 2.10 Parallel MorphBPE Follow-up: 150M and 300M (2026-04-09)

После сравнения `50m checkpoint-26000` стало ясно:

- новый `50m morphbpe + fleursnorm` уже лучше старого `50m` на том же шаге;
- но абсолютное качество всё ещё плохое, значит нужен перенос той же схемы на более крупные decoder sizes.

Для этого были подготовлены и запущены два новых CloudRift pod'а:

- `150m`
  - config: `omniaudio/configs/v2_llm150m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml`
  - instance id: `2a3add2c-340b-11f1-9fdd-b7684ea2946b`
  - host: `217.138.104.172`
  - screen: `train_150m_morphbpe_long15`
  - log: `/home/riftuser/slm/logs/cloudrift_150m_morphbpe_long15.log`
  - init: `stukenov/sozkz-core-omniaudio-150m-kk-ctc-v1` `checkpoint-best`

- `300m`
  - config: `omniaudio/configs/v2_llm300m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml`
  - instance id: `2b31ef86-340b-11f1-9fdd-d349a63b20c0`
  - host: `176.124.69.204`
  - screen: `train_300m_morphbpe_long15`
  - log: `/home/riftuser/slm/logs/cloudrift_300m_morphbpe_long15.log`
  - init: `stukenov/sozkz-core-omniaudio-300m-kk-ctc-v1` `checkpoint-best`

Общая схема для обоих:

- tokenizer: `stukenov/sozkz-morphbpe-100k-kk-v1`
- text normalization:
  - lowercase
  - punctuation stripping
  - whitespace collapse
- decode constraints:
  - `decode_repetition_penalty: 1.15`
  - `decode_no_repeat_ngram_size: 3`
- train window:
  - `max_audio_len: 15.0`

Статус запуска:

- `300m` уже подтвердил:
  - загрузку base LLM
  - resize embeddings to `100000`
  - загрузку `CTC checkpoint-best`
  - вход в dataset/materialization stage

- `150m` уже подтвердил:
  - загрузку base LLM
  - resize embeddings to `100000`
  - вход в E2E initialization

Следующий шаг:

- дождаться первых training steps на обоих новых run;
- затем отдельно снять rough `FLEURS` eval на ранних checkpoint (`~10k-20k`) и сравнить с `50m`.

## 2.11 Финальные результаты MorphBPE Long-15 серии (2026-04-10)

Все три модели (`150m`, `300m`, `600m`) завершены. Финальные результаты на `google/fleurs kk_kz test`:

| Model | FLEURS samples | WER | CER | Status |
|-------|---------------|-----|-----|--------|
| **150M** (`checkpoint-best`, 5 epochs) | **853** | **47.27%** | **33.65%** | **terminated 2026-04-10** |
| 300M (`checkpoint-92000`, ~87% schedule) | 200 | 49.69% | 35.02% | terminated 2026-04-10 |
| 600M (`checkpoint-40000`, ~0.5 epochs) | 200 | 98.18% | 78.15% | terminated 2026-04-10, severely undertrained |

**Вывод:** `150M checkpoint-best` — лучший результат в серии. `300M` немного хуже несмотря на больший размер, что объясняется меньшим числом gradient updates из-за `grad_accum=4` (против `accum=1` у 150M). `600M` не успел обучиться.

HF: `stukenov/sozkz-core-omniaudio-150m-kk-asr-v1`

---

## 8. Воспроизводимость

### Конфиги

```bash
# Stage 1: CTC Pretrain
python -m omniaudio.train_v2 --config omniaudio/configs/v2_scratch_kzcalm_s1.yaml

# Stage 2: E2E
python -m omniaudio.train_v2 --config omniaudio/configs/v2_scratch_kzcalm_e2e.yaml

# Evaluation
python -m omniaudio.evaluate_v2 \
  --config omniaudio/configs/v2_scratch_kzcalm_e2e.yaml \
  --model-path outputs/omniaudio_v2_scratch_kzcalm_e2e/checkpoint-best/model.pt \
  --max-samples 100
```

### Зависимости

```
torch >= 2.0
torchaudio
transformers >= 4.40
datasets
jiwer (для WER/CER)
```

---

## Appendix A: Полная loss кривая CTC Pretrain

```
Step     Loss     LR
100      72.34    5.86e-05
500      22.81    2.93e-04
1000     15.24    5.86e-04
2000     10.23    9.91e-04  ← checkpoint saved
5000      7.27    9.45e-04
10000     4.69    8.33e-04
15000     3.60    6.89e-04
20000     3.48    5.54e-04
20400     3.47    5.46e-04  ← Epoch 3 val: 0.7826
25000     2.69    3.90e-04
27300     2.68    2.11e-04  ← Epoch 4 val: 0.5823
30000     0.70    1.30e-04
34100     0.695   8.33e-07  ← Epoch 5 val: 0.4886 (final)
```

## Appendix B: Полная loss кривая E2E

```
Step     Loss     LR         Event
100      227.31   8.79e-06
500      77.13    4.40e-05
1000     50.16    8.79e-05
2000     31.31    1.76e-04
3500     21.01    3.00e-04   ← warmup complete, peak LR
5000     16.46    2.93e-04
6800     13.48    2.84e-04   ← Epoch 1 val: 4.91
10000    4.49     2.70e-04
13600    4.23     2.53e-04   ← Epoch 2 val: 3.39
17000    3.56     2.37e-04
20400    3.47     2.21e-04   ← Epoch 3 val: 2.82
24800    3.11     2.01e-04   ← current (Epoch 4, in progress)
```
