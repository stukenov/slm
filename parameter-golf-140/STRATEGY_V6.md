# Parameter Golf — Strategy v6 (2026-04-03)

## КРИТИЧЕСКОЕ ОБНОВЛЕНИЕ

**Наш PR #745 (1.0222 BPB) ОТКЛОНЁН.** Причина: Hedge Mixer нарушает Condition 1 и 2 (Issue #1017) — n-gram кэши смотрят на target token, entropy expert не является нормализованным распределением.

**У нас НЕТ активного submission на лидерборде.** PR #264 (1.1455) помечен как потенциально невалидный.

---

## 1. Текущее состояние соревнования (3 апреля 2026)

### Merged лидерборд (официальный)

| # | BPB | Автор | PR | Ключ |
|---|-----|-------|----|------|
| **1** | **1.1147** | @abaybektursun | #1019 | AR Self-Gen GPTQ + XSA-all + BigramHash 3072 (NO TTT) |
| 2 | 1.1194 | @abaybektursun | #549 | LeakyReLU² + Legal TTT |
| 3 | 1.1228 | @signalrush | #414 | GPTQ-lite + EMA |
| 4 | 1.1248 | @jfprincz | #287 | Partial RoPE + LN Scale |

### Pending PRs (открытые, не merged)

| BPB | Автор | PR | Ключ | Легальность |
|-----|-------|----|------|------------|
| **0.9300** | @resouer | #1229 | Per-sample SLOT + logit bias | ⚠️ SLOT causality debate |
| **0.9354** | unknown | #1263 | SLOT + QK-Gain 4.0 + Full GPTQ | ⚠️ SLOT causality debate |
| **0.9462** | unknown | #1303 | SLOT + QK-Gain 4.0 + XSA-11 | ⚠️ SLOT causality debate |
| **1.0846** | unknown | #1306 | **Causal SLOT** + Pre-quant TTT | ✅ Designed to be legal |
| 1.0925 | unknown | #1291 | Vocab4096 + MLP4x + SLOT | ⚠️ |
| 1.1043 | unknown | #1298 | Polar Express + SLOT + MuonEq-R | ⚠️ |
| 1.1079 | unknown | #1302 | Split-LR + N-gram Agreement + Full GPTQ | ✅ legal |

### Ключевые числа

- 1310+ PRs открыто, 10 merged records
- Дедлайн: **30 апреля** (27 дней)
- Лидерборд движется быстро — новые SLOT PRs каждый день

---

## 2. SLOT — новая доминирующая техника

### Что такое SLOT (Scored-position Learned Output Tuning)

Из arXiv:2505.12392v2. При eval-time:

1. Модель **заморожена** (`torch.no_grad()`)
2. Извлекаются hidden states из последнего слоя
3. Оптимизируется **per-sample delta** `[bsz, 1, 512]` + **logit bias** `[bsz, 1, vocab]`
4. 16 шагов AdamW, cosine LR 0.008→0.0008
5. Только scored positions (last `stride` tokens per window) в loss

### SLOT results

| PR | BPB | Type | Legal? |
|----|-----|------|--------|
| #1229 | **0.9300** | Standard SLOT + per-sample delta + logit bias | ⚠️ PR #1240 proved causal violation |
| #1263 | 0.9354 | Standard SLOT + QK-Gain 4.0 | ⚠️ same issue |
| #1306 | 1.0846 | **Causal SLOT** (context-only positions) + Pre-quant TTT | ✅ designed legal |

### SLOT legality

- **PR #1240** доказал: стандартный SLOT нарушает causal dependence (100% violation rate)
- **Issue #1017** определяет 4 условия валидности:
  - **Condition 1**: p_t зависит только от x_1...x_{t-1}
  - **Condition 2**: полное нормализованное распределение
  - **Condition 3**: score-before-update
  - **Condition 4**: single left-to-right pass
- **Causal SLOT** (PR #1306): оптимизирует delta только по context-only positions (уже оценённые). BPB хуже (1.08 vs 0.93) но legally safe
- **Организаторы пока не вынесли решение** по SLOT

---

## 3. Что случилось с нашими подходами

### Hedge Mixer — REJECTED

Причины отказа (от @valerio-oai):
1. N-gram кэши "look ahead to the target token to mix probabilities" → нарушает Condition 1
2. "Do not renormalize correctly" → нарушает Condition 2
3. Entropy expert — скаляр, а не распределение → нарушает Condition 2

### Как исправить

Чтобы сделать Hedge Mixer легальным:
1. **N-gram таблицы**: обновлять ТОЛЬКО из уже оценённых (scored) токенов, ПЕРЕД обработкой текущего chunk
2. **Mixing**: каждый expert должен выдавать полное распределение P(·) над vocab, а не скаляр
3. **Entropy expert**: удалить или заменить на unigram bias (полное распределение)
4. **Нормализация**: mixture = Σ(w_i * P_i(·)), убедиться что Σ P(a) = 1

### Pre-quant TTT — NEW (работает с GPTQ)

PR #1306 показал: AdamW TTT на full-precision EMA весах **до** GPTQ квантизации даёт -0.022 BPB.
- Наш старый SGD TTT на квантизованных весах давал 25 провалов (PR #756)
- Pre-quant TTT обходит проблему: адаптируем до квантизации → квантизованная модель лучше

---

## 4. Анализ: как попасть на 1-е место

### Текущая иерархия техник по impact

| Техника | BPB gain | Легальность | Сложность |
|---------|---------|------------|-----------|
| Standard SLOT | **-0.19** (1.11→0.93) | ⚠️ under debate | Medium |
| Causal SLOT | **-0.03** (1.11→1.08) | ✅ legal | Medium |
| Pre-quant TTT | **-0.02** | ✅ legal | Easy |
| Full Hessian GPTQ (AR self-gen) | **-0.005** | ✅ merged | Medium |
| QK-Gain 4.0 | **-0.005** | ✅ | Trivial |
| N-gram backoff | **-0.10 to -0.30** | ⚠️ "leaning towards accepting" | Hard |
| XSA all layers | **-0.006** | ✅ merged | Done |
| BigramHash 3072×112 | **-0.002** | ✅ merged | Easy |
| Polar Express optimizer | **-0.002** | ✅ | Easy |
| MuonEq-R | **-0.002** | ✅ | Easy |

### Три сценария для 1-го места

**Сценарий A: SLOT принят (вероятность ~50%)**
- Нужен лучший SLOT implementation
- Текущий топ: 0.9300 (PR #1229)
- Наш target: **<0.92** (SLOT + лучшая база + n-gram)

**Сценарий B: Только Causal SLOT принят (вероятность ~30%)**
- Causal SLOT + Pre-quant TTT + N-gram
- Текущий топ: 1.0846 (PR #1306)
- Наш target: **<1.05** (Causal SLOT + TTT + n-gram agreement)

**Сценарий C: Всё SLOT-подобное rejected (вероятность ~20%)**
- Остаётся: TTT + neural base + n-gram
- Текущий merged SOTA: 1.1147
- Наш target: **<1.10** (лучшая база + legal TTT + n-gram agreement)

---

## 5. ПЛАН ДЕЙСТВИЙ

### Принципы

1. **Готовим ДВА submission**: aggressive (SLOT) и conservative (no SLOT)
2. **Base = PR #1019** (merged SOTA, fastest, strongest quant)
3. **Только RunPod** для compute
4. **Каждую технику тестируем отдельно** на 1xH100 перед combined run

### Phase 0: Setup (Day 1)

| # | Задача |
|---|--------|
| 0.1 | Fork PR #1019 base (merged SOTA) — скачать `train_gpt.py` |
| 0.2 | Поднять RunPod 1xH100 для ablations (~$2/hr) |
| 0.3 | Smoke test: воспроизвести 1.1147 на 1 seed |
| 0.4 | Прочитать код PR #1229 (лучший SLOT) и PR #1306 (Causal SLOT) |

### Phase 1: Build Strongest Neural Base (Days 2-4, ~$20)

Поверх PR #1019 добавляем:

| # | Техника | Источник | Expected gain |
|---|---------|----------|---------------|
| 1.1 | **QK-Gain 4.0** | PR #1125, #1176 | -0.005 BPB |
| 1.2 | **Polar Express NS** (improved Newton-Schulz) | PR #1298, NanoGPT | -0.002 BPB |
| 1.3 | **MuonEq-R** (equalized row norms) | PR #1285, #1290 | -0.002 BPB |
| 1.4 | **Vocab 4096** (вместо 1024) + re-tokenize | PR #1291 | -0.005 BPB (risky) |
| 1.5 | **MLP 4x** (если влезает в 16MB) | PR #1291 | -0.003 BPB |
| 1.6 | **Warmdown 4000** (вместо 3500) | PR #1019 | -0.001 BPB |
| 1.7 | **Coprime-stride data loader** | PR #1306 | -0.003 BPB |
| | **Base target** | | **~1.095-1.105 BPB** |

Ablation sequence: 1xH100, 1 seed, 600s each. ~10 runs = ~$20.

### Phase 2: SLOT Implementation (Days 4-7, ~$15)

Реализуем ОБА варианта:

**2A. Standard SLOT** (aggressive):
| # | Задача |
|---|--------|
| 2A.1 | Implement per-sample delta `[bsz, 1, 512]` + logit bias `[bsz, 1, vocab]` |
| 2A.2 | Scored-position mask (last stride tokens per window) |
| 2A.3 | 16 AdamW steps, cosine LR 0.008→0.0008 |
| 2A.4 | Tune: steps (8/16/32), LR, delta dims |
| | **Target**: ~0.93-0.95 BPB |

**2B. Causal SLOT** (conservative):
| # | Задача |
|---|--------|
| 2B.1 | SLOT loss only on context-only positions (already-scored) |
| 2B.2 | Ensure Condition 1-4 compliance |
| | **Target**: ~1.07-1.08 BPB |

### Phase 3: Pre-quant TTT (Days 7-9, ~$10)

| # | Задача | Gain |
|---|--------|------|
| 3.1 | AdamW TTT на EMA весах **до** GPTQ квантизации | -0.02 BPB |
| 3.2 | 6 epochs, cosine LR | -0.005 vs 1 epoch |
| 3.3 | Combine с SLOT (SLOT после GPTQ, TTT до) | additive? |
| | **Combined target (A)**: **~0.91-0.93 BPB** | |
| | **Combined target (B)**: **~1.05-1.06 BPB** | |

### Phase 4: Legal N-gram Agreement (Days 9-12, ~$15)

Если n-gram legal (organizers "leaning towards accepting"):

| # | Задача | Gain |
|---|--------|------|
| 4.1 | Fix Hedge Mixer: каждый expert = полное P(·) over vocab | compliance |
| 4.2 | N-gram tables updated only from scored tokens, before current chunk | compliance |
| 4.3 | Remove entropy expert (scalar → нельзя) | compliance |
| 4.4 | Add 4-gram, 5-gram experts | -0.01 BPB |
| 4.5 | Entropy-adaptive alpha blending | -0.01 BPB |
| 4.6 | Compliance audit: verify Conditions 1-4 | critical |
| | **Target with n-gram**: **~0.85-0.90 BPB (A)** or **~0.95-1.00 BPB (B)** | |

### Phase 5: 3-Seed Validation + Submit (Days 12-15, ~$40)

| # | Задача | Cost |
|---|--------|------|
| 5.1 | 8xH100 run: Submission A (SLOT aggressive) × 3 seeds | ~$20 |
| 5.2 | 8xH100 run: Submission B (Causal SLOT conservative) × 3 seeds | ~$20 |
| 5.3 | Pick best legal submission |  |
| 5.4 | Write README, submission.json | |
| 5.5 | Submit PR | |

### Phase 6: Buffer + Iteration (Days 15-27, ~$50)

- React to organizer decisions on SLOT legality
- React to competing PRs
- Improve based on review feedback
- Try higher-order n-gram if n-gram confirmed legal
- Try order-13 n-gram oracle if time permits

---

## 6. Projected Results

| Submission | Expected BPB | If legal? | Rank estimate |
|-----------|-------------|-----------|---------------|
| **A: Standard SLOT + TTT + n-gram** | **0.85-0.93** | ⚠️ depends on SLOT ruling | **Top 1-3** |
| **B: Causal SLOT + TTT + n-gram** | **0.95-1.05** | ✅ legal by design | **Top 1-5** |
| **C: No SLOT, TTT + n-gram** | **1.05-1.10** | ✅ safe | **Top 5-10** |
| **Fallback: PR #1019 + improvements** | **1.09-1.10** | ✅ guaranteed | **~Top 3 merged** |

---

## 7. Бюджет (RunPod only)

| Phase | GPU | Cost |
|-------|-----|------|
| Phase 0: Setup | 1xH100, 2h | ~$4 |
| Phase 1: Neural base ablations | 1xH100, 10h | ~$20 |
| Phase 2: SLOT implementation | 1xH100, 7h | ~$15 |
| Phase 3: Pre-quant TTT | 1xH100, 5h | ~$10 |
| Phase 4: N-gram | 1xH100, 7h | ~$15 |
| Phase 5: 3-seed validation | 8xH100, 2h | ~$40 |
| Phase 6: Buffer | mixed | ~$50 |
| **Total** | | **~$154** |

---

## 8. Риски

| Risk | Probability | Mitigation |
|------|------------|------------|
| Standard SLOT rejected | 50% | Submission B (Causal SLOT) ready |
| Causal SLOT also rejected | 20% | Submission C (no SLOT, TTT + n-gram) |
| N-gram rejected | 30% | N-gram is optional addon, not core |
| Someone beats us with novel technique | 30% | 27 days buffer for iteration |
| RunPod instability | 20% | Screen sessions, checkpoint saves |
| Pre-quant TTT doesn't stack with SLOT | 30% | Test independently first |
| Reviewer finds compliance issue | 40% | Meticulous Condition 1-4 audit |

---

## 9. Key PRs to Study

| PR | Why |
|----|-----|
| **#1019** | Merged SOTA base — fork this |
| **#1229** | Best SLOT implementation (0.93) |
| **#1306** | Causal SLOT + Pre-quant TTT |
| **#1263** | SLOT + QK-Gain 4.0 (0.9354) |
| **#1298** | Polar Express + MuonEq-R |
| **#1291** | Vocab4096 + MLP4x |
| **#1285** | MuonEq-R + Depth Recurrence |
| **#1302** | N-gram Agreement (legal) |
| **#1240** | SLOT causality proof (must understand) |
| **#1017** | Field Guide to Valid Submissions (rules) |

---

## 10. Immediate Actions

```
1. [ ] Скачать train_gpt.py из PR #1019 (merged SOTA)
2. [ ] Скачать train_gpt.py из PR #1229 (best SLOT)
3. [ ] Скачать train_gpt.py из PR #1306 (Causal SLOT)
4. [ ] Прочитать PR #1240 (SLOT causality proof) полностью
5. [ ] Прочитать Issue #1017 (validity rules) полностью
6. [ ] Запустить RunPod 1xH100, воспроизвести PR #1019 baseline
7. [ ] Реализовать QK-Gain 4.0 — самый простой первый шаг
```
