# Google TRC Grant — SozKZ Project

**Status:** ACTIVE
**Project:** sozkz-trc
**Project Number:** 65934168684
**Grant Duration:** 30 days from activation
**Support Email:** trc-support@google.com
**Discord:** #tpu-research-cloud on Google Developer Community Discord

---

## Что дали (квота)

| TPU | Чипов | Тип | Зона | Примерная стоимость/час (все чипы) |
|-----|-------|-----|------|-------------------------------------|
| v6e | 64 | spot | us-east1-d | ~$96/hr |
| v4 | 32 | spot | us-central2-b | ~$31/hr |
| v4 | 32 | on-demand | us-central2-b | ~$103/hr |
| v5e | 64 | spot | us-central1-a | ~$31/hr |
| v6e | 64 | spot | europe-west4-a | ~$96/hr |
| v5e | 64 | spot | europe-west4-b | ~$31/hr |

**Итого чипов:** 320 TPU чипов (128 v6e + 128 v5e + 64 v4)

**Оценочная стоимость при полном использовании 24/7 за 30 дней:**
- v6e (128 spot): ~$138,000
- v5e (128 spot): ~$44,000
- v4 (32 spot + 32 on-demand): ~$97,000
- **ИТОГО: ~$280,000+**

Это один из самых крупных грантов кампании.

---

## КРИТИЧЕСКИЕ ПРАВИЛА (прочитай перед использованием!)

### 1. ЗОНЫ — создавай TPU ТОЛЬКО в указанных зонах!
Если создашь TPU в другой зоне — **пойдут реальные списания с карты**.

| TPU тип | Зона | Тип квоты |
|---------|------|-----------|
| v6e | **us-east1-d** | spot |
| v4 | **us-central2-b** | spot И on-demand |
| v5e | **us-central1-a** | spot |
| v6e | **europe-west4-a** | spot |
| v5e | **europe-west4-b** | spot |

### 2. Spot vs On-Demand
- **Spot (preemptible)** — дешевле, но Google может забрать TPU в любой момент. Используй для задач с checkpoint'ами.
- **On-demand** — гарантированно доступны, но только для v4 в us-central2-b (32 чипа). Используй для критичных задач.
- **Рекомендация Google:** предпочитай on-demand, переходи на spot если on-demand недоступен.

### 3. Бесплатно только TPU!
Остальные GCP сервисы (Storage, Compute Engine, networking) — платные. $300 вступительный кредит Google Cloud может покрыть эти расходы.

### 4. Не забывай удалять TPU!
Неиспользуемые TPU и Queued Resources продолжают занимать квоту. Если видишь ошибку "quota exhausted" — проверь, нет ли забытых ресурсов.

### 5. Квота не гарантирована
Google может забрать TRC квоту и TPU capacity в любое время. Высокий спрос.

---

## Архитектура — используй TPU VM + Queued Resource API

Google рекомендует использовать **TPU VM architecture** и **Queued Resource API**.

### Быстрый старт

```bash
# 1. Установи gcloud CLI (если не установлен)
curl https://sdk.cloud.google.com | bash
gcloud init
gcloud config set project sozkz-trc

# 2. Создай TPU VM (пример: v4, spot, us-central2-b)
gcloud compute tpus tpu-vm create sozkz-v4-worker \
  --zone=us-central2-b \
  --accelerator-type=v4-32 \
  --version=tpu-vm-tf-2.17.0-pjrt \
  --spot

# 3. Подключись к TPU VM
gcloud compute tpus tpu-vm ssh sozkz-v4-worker \
  --zone=us-central2-b

# 4. УДАЛИ когда закончишь!
gcloud compute tpus tpu-vm delete sozkz-v4-worker \
  --zone=us-central2-b
```

### Queued Resource API (рекомендуется)

```bash
# Создать queued resource (встанет в очередь, запустится когда ресурсы доступны)
gcloud compute tpus queued-resources create sozkz-queue \
  --node-id=sozkz-v5e-worker \
  --zone=us-central1-a \
  --accelerator-type=v5litepod-64 \
  --runtime-version=v2-alpha-tpuv5-lite \
  --spot

# Проверить статус
gcloud compute tpus queued-resources describe sozkz-queue \
  --zone=us-central1-a

# Удалить
gcloud compute tpus queued-resources delete sozkz-queue \
  --zone=us-central1-a
```

---

## Конфигурации TPU для задач SozKZ

### Для перевода датасета (основная цель)
Рекомендация: **v5e spot** в us-central1-a или europe-west4-b
- 64 чипа v5e = достаточно для массового инференса/перевода
- Spot дешевле и подходит для batch-задач с возможностью перезапуска

### Для обучения моделей (если будем использовать)
Рекомендация: **v4 on-demand** в us-central2-b
- 32 чипа v4, on-demand = стабильно, не прервётся
- v4 — лучшая производительность для training

### Для экспериментов
Рекомендация: **v6e spot** в us-east1-d или europe-west4-a
- Новейшие TPU, высокая производительность
- 64 чипа — достаточно для любых экспериментов

---

## Совместимость фреймворков

| Фреймворк | Поддержка | Примечание |
|-----------|-----------|------------|
| JAX | ✅ Нативная | Лучший выбор для TPU |
| PyTorch | ✅ Через PyTorch/XLA | Требует адаптации кода |
| TensorFlow | ✅ Нативная | Хорошо работает |
| Keras | ✅ Нативная | Через JAX или TF backend |

### Для SozKZ (PyTorch pipeline)
Нужно будет использовать **PyTorch/XLA**. Полезные ресурсы:
- PyTorch/XLA Performance debugging Part I-III (см. блог Google)
- https://github.com/pytorch/xla

### Для перевода датасета
Рекомендуется **JAX + Flax** для максимальной производительности на TPU.

---

## Флаги для разных типов квоты

### On-demand TPU
```bash
gcloud compute tpus tpu-vm create NAME \
  --zone=ZONE \
  --accelerator-type=TYPE \
  --version=RUNTIME_VERSION
```

### Spot (preemptible) TPU
```bash
gcloud compute tpus tpu-vm create NAME \
  --zone=ZONE \
  --accelerator-type=TYPE \
  --version=RUNTIME_VERSION \
  --spot
```

⚠️ **Используй правильный флаг!** `--spot` для spot квоты, без флага для on-demand.

---

## Мониторинг и безопасность

### Проверить текущие TPU
```bash
gcloud compute tpus tpu-vm list --zone=us-central2-b
gcloud compute tpus tpu-vm list --zone=us-central1-a
gcloud compute tpus tpu-vm list --zone=us-east1-d
gcloud compute tpus tpu-vm list --zone=europe-west4-a
gcloud compute tpus tpu-vm list --zone=europe-west4-b
```

### Проверить queued resources
```bash
gcloud compute tpus queued-resources list --zone=ZONE
```

### Проверить биллинг
- https://console.cloud.google.com/billing/01607B-11ED2C-3988BC

### Установить бюджетный алерт (РЕКОМЕНДУЕТСЯ!)
1. Зайди в Billing → Budgets & alerts
2. Создай бюджет на $50 (для non-TPU сервисов)
3. Установи alert на 50%, 90%, 100%

---

## Обязательства по гранту

1. **Поделиться исследованием** — публикации, open-source код, блог-посты
2. **Дать обратную связь Google** — помочь улучшить TRC и Cloud TPU
3. **Соблюдать Google AI Principles** — https://ai.google/principles/
4. **Terms & Conditions + Privacy Policy** Google

Для SozKZ это естественно — всё и так open-source (arXiv, HuggingFace, GitHub).

---

## Полезные ссылки

- **QuickStart:** https://cloud.google.com/tpu/docs/quickstart
- **TPU v4 User Guide:** https://cloud.google.com/tpu/docs/v4-users-guide
- **Performance Guide:** https://cloud.google.com/tpu/docs/performance-guide
- **Profiling Guide:** https://cloud.google.com/tpu/docs/profiling-guide
- **Troubleshooting:** https://cloud.google.com/tpu/docs/troubleshooting
- **TRC FAQ:** https://sites.research.google/trc/faq/
- **GCP Console:** https://console.cloud.google.com/?project=sozkz-trc
- **TPU Dashboard:** https://console.cloud.google.com/compute/tpus?project=sozkz-trc

---

## Чеклист перед началом работы

- [ ] Установить gcloud CLI
- [ ] `gcloud config set project sozkz-trc`
- [ ] Настроить бюджетный алерт (Billing → Budgets & alerts)
- [ ] Выбрать задачу (перевод датасета / обучение / эксперименты)
- [ ] Выбрать подходящую зону и тип TPU
- [ ] Создать TPU VM или Queued Resource
- [ ] Запустить задачу
- [ ] **УДАЛИТЬ TPU после завершения!**
