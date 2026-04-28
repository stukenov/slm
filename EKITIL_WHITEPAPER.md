# EkiTil: Bilingual Kazakh-Russian Language Models

**EkiTil** (Екі Тіл — «Два Языка») — семейство двуязычных казахско-русских языковых моделей, обученных с нуля на сбалансированном корпусе из казахских и русских текстов.

**Authors**: Saken Tukenov
**Date**: 2026-04
**Status**: EkiTil-123M and EkiTil-300M trained and published

---

## Abstract

EkiTil is a family of bilingual Kazakh-Russian causal language models trained from scratch on a curated bilingual corpus. Unlike previous SozKZ models (Kazakh-only), EkiTil targets native bilingual competence in both Kazakh and Russian with cross-lingual transfer through parallel data.

Two models have been trained and published:
- **EkiTil-123M** (124.7M params): 2.47B tokens, 1 epoch, final loss 3.07 / BPB 4.44
- **EkiTil-300M** (245.9M params): 4.94B tokens, 2 epochs, final loss 2.93 / BPB 4.22

Both use the Qwen3 architecture with a custom 64K BPE tokenizer optimized for Kazakh and Russian, achieving 1.56 tokens/word fertility on mixed text. The training corpus comprises 2.47B unique tokens (1.33B Kazakh + 1.46B Russian + 7.2M parallel). Multi-epoch training enables larger models to train at near-Chinchilla-optimal ratios on the same corpus. Total training cost: ~$40 on RunPod (H100 GPUs).

---

## 1. Motivation

Kazakh is a low-resource Turkic language with ~18M speakers. Despite growing digitalization in Kazakhstan, existing large language models handle Kazakh poorly due to:

1. **Tokenizer inefficiency**: Models like Llama-3 (~3.0 tok/word on Kazakh) and GPT-4 waste context on Kazakh text
2. **Minimal pre-training data**: Kazakh represents <0.01% of Common Crawl
3. **Bilingual reality**: Kazakhstan is functionally bilingual (Kazakh + Russian); a practical model must handle both languages and translation between them

Previous SozKZ experiments (exp001–exp028) focused on Kazakh-only models. EkiTil extends this to bilingual kk-ru, reflecting actual language use in Kazakhstan.

### 1.1 Why Bilingual, Not Multilingual

A focused bilingual model for the kk-ru pair offers several advantages over multilingual approaches:

- **Concentrated capacity**: All model parameters serve exactly two languages instead of being diluted across dozens
- **Cultural alignment**: Kazakh and Russian share significant cultural, institutional, and technical vocabulary
- **Practical utility**: The vast majority of Kazakh speakers are bilingual in Russian; translation between kk↔ru is the dominant NLP need
- **Data efficiency**: Russian data is abundant and high-quality, providing a strong signal for shared representations

---

## 2. Data Pipeline

### 2.1 Kazakh Corpus Annotation

**Source**: `kz-transformers/multidomain-kazakh-dataset` (24.9M documents)

Processing pipeline:
1. Document → sentence splitting
2. Per-sentence language detection (fasttext `lid.176.bin`)
3. Metadata annotation: `doc_id`, `source`, `domain`, `detected_lang`, `lang_confidence`, `num_chars`, `is_kk`

**Result**: [`stukenov/ekitil-corpus-annotated-kk-v1`](https://huggingface.co/datasets/stukenov/ekitil-corpus-annotated-kk-v1)

| Language | Sentences | Share |
|----------|-----------|-------|
| Kazakh | 60.1M | 49.3% |
| Russian | 57.0M | 46.8% |
| English | 2.3M | 1.9% |
| Other | 2.5M | 2.0% |
| **Total** | **121.9M** | 100% |

**Key finding**: The "Kazakh" dataset already contains nearly equal amounts of Russian (57M vs 60M sentences). No separate Russian corpus collection was needed.

### 2.2 Parallel Corpus

**Sources**:
- Helsinki-NLP/opus-100 (kk-ru subset)
- Dauren-Nur/kaz_rus_parallel_corpora_KAZNU

**Result**: [`stukenov/ekitil-corpus-parallel-kkru-v1`](https://huggingface.co/datasets/stukenov/ekitil-corpus-parallel-kkru-v1) — ~135K sentence-aligned parallel pairs

**Format**:
```
<|kk|> Қазақ тіліндегі сөйлем. <|translate|> <|ru|> Предложение на казахском языке. <|endoftext|>
```

### 2.3 Tokenizer

**Approach**: ByteLevel BPE trained from scratch on a balanced kk+ru corpus.

**Training corpus** (4.9M sentences):
- 50% Kazakh (2.5M, reservoir sampled from 60M)
- 45% Russian (2.25M, reservoir sampled from 57M)
- 5% Parallel pairs (135K)

**Result**: [`stukenov/ekitil-vocab-bpe-64k-kkru-v1`](https://huggingface.co/huggingface/stukenov/ekitil-vocab-bpe-64k-kkru-v1)

| Property | Value |
|----------|-------|
| Algorithm | ByteLevel BPE |
| Vocab size | 64,000 |
| Fertility (kk+ru mixed) | **1.56 tok/word** |
| Min frequency | 100 |

**Special tokens**:

| ID | Token | Purpose |
|----|-------|---------|
| 0 | `<\|endoftext\|>` | End of document |
| 1 | `<\|padding\|>` | Padding |
| 2 | `<\|startoftext\|>` | Start of text |
| 3 | `<\|kk\|>` | Kazakh language tag |
| 4 | `<\|ru\|>` | Russian language tag |
| 5 | `<\|translate\|>` | Translation task marker |

**Comparison with other tokenizers**:

| Tokenizer | Vocab | Fertility kk | Fertility ru | Fertility mixed |
|-----------|-------|-------------|-------------|-----------------|
| **EkiTil BPE-64K** | **64K** | **~1.5** | **~1.6** | **1.56** |
| SozKZ BPE-50K (kk-only) | 50K | ~1.8 | ~3.5+ | — |
| Qwen3 (151K) | 151K | ~2.5 | ~1.3 | — |
| Llama-3 (128K) | 128K | ~3.0+ | ~2.0 | — |

The EkiTil tokenizer achieves significantly better Kazakh fertility than general-purpose tokenizers while maintaining strong Russian coverage, thanks to the balanced bilingual training corpus.

### 2.4 Tokenized Dataset

**Result**: [`stukenov/ekitil-corpus-tokenized-kkru-v1`](https://huggingface.co/datasets/stukenov/ekitil-corpus-tokenized-kkru-v1)

| Metric | Value |
|--------|-------|
| Kazakh tokens | 1.33B |
| Russian tokens | 1.46B |
| Parallel tokens | 7.2M |
| **Total tokens** | **~2.47B** |
| Block size | 2048 |
| Number of blocks | 1,205,750 |

**Note**: Current version (v1) is sentence-level tokenized. Document-level tokenization (v2) preserving cross-sentence coherence is planned.

---

## 3. Model Architecture

### 3.1 Design Decisions

**Architecture**: Qwen3ForCausalLM (transformers native implementation)

**Why Qwen3 over Llama**:
- Qwen3 has a reference 0.6B model — closer to our target scale
- Better native CJK/Cyrillic coverage in architecture design
- GQA (Grouped Query Attention) for memory efficiency
- Hybrid thinking mode support in the architecture family

**Why from scratch, not continued pre-training**:
- Custom 64K tokenizer is incompatible with Qwen3's 151K vocab embedding weights
- 2.47B tokens at Chinchilla ratio (19.8:1) is sufficient for cold-start at 123M scale
- Full control over learned representations — no catastrophic forgetting issues
- Previous experiments (exp013–exp023) confirmed from-scratch works well at this scale

### 3.2 EkiTil-123M Configuration

```yaml
model_type: qwen3
vocab_size: 64000
hidden_size: 768
num_hidden_layers: 12
num_attention_heads: 12
num_key_value_heads: 4        # GQA ratio 3:1
head_dim: 64
intermediate_size: 2048
hidden_act: silu
max_position_embeddings: 2048
rms_norm_eps: 1e-6
rope_theta: 1000000
tie_word_embeddings: true
attention_bias: false
```

| Component | Parameters |
|-----------|-----------|
| Embedding (tied) | 49.2M |
| Attention (×12) | ~28.3M |
| MLP (×12) | ~47.2M |
| **Total** | **~124.7M** |

### 3.3 EkiTil-300M Configuration

```yaml
model_type: qwen3
vocab_size: 64000
hidden_size: 1024
num_hidden_layers: 16
num_attention_heads: 16
num_key_value_heads: 4        # GQA ratio 4:1
head_dim: 64
intermediate_size: 2816
hidden_act: silu
max_position_embeddings: 2048
rms_norm_eps: 1e-6
rope_theta: 1000000
tie_word_embeddings: true
attention_bias: false
```

| Component | Parameters |
|-----------|-----------|
| Embedding (tied) | 65.5M |
| Attention (×16) | ~41.9M |
| MLP (×16) | ~138.4M |
| **Total** | **~245.9M** |

### 3.4 Scaling Strategy

The EkiTil family uses multi-epoch training on the same 2.47B unique token corpus to train progressively larger models:

| Model | Actual Params | Architecture | Epochs | Total Tokens | Ratio | GPUs | Status |
|-------|--------------|-------------|--------|-------------|-------|------|--------|
| **EkiTil-123M** | 124.7M | 768d/12L/12h/4kv/2048i | 1 | 2.47B | 19.8:1 | 1×H100 | Published |
| **EkiTil-300M** | 245.9M | 1024d/16L/16h/4kv/2816i | 2 | 4.94B | 20.1:1 | 2×H100 | Published |
| EkiTil-600M | ~674M | 1280d/28L/20h/4kv/4480i | 5 | 12.35B | 18.3:1 | 4×H100 | Planned |

---

## 4. Training

### 4.1 Training Configuration

| Parameter | EkiTil-123M | EkiTil-300M |
|-----------|-------------|-------------|
| Optimizer | AdamW (β1=0.9, β2=0.95) | AdamW (β1=0.9, β2=0.95) |
| Learning rate | 6e-4 | 3e-4 |
| LR schedule | Cosine decay (min 10%) | Cosine decay (min 10%) |
| Warmup steps | 2,000 | 2,000 |
| Weight decay | 0.1 | 0.1 |
| Max grad norm | 1.0 | 1.0 |
| Precision | bf16 | bf16 |
| Batch size (per GPU) | 16 | 8 |
| Gradient accumulation | 8 | 8 |
| Effective batch (tok/step) | 262,016 | 262,016 |
| Sequence length | 2,048 | 2,048 |
| Epochs | 1 | 2 |
| Total steps | 9,424 | 18,849 |

### 4.2 Infrastructure

Training scripts: `scripts/exp027/train_ekitil.py` (unified) and `scripts/exp027/train_ekitil_123m.py` (legacy)
- Custom DDP multi-GPU training loop (torchrun)
- Memory-mapped data loading (numpy memmap, random-access blocks)
- Spot-instance friendly: checkpoint resume from local or HuggingFace
- Automatic checkpoint upload to HF every 2,000 steps
- Automatic final model upload to HuggingFace Hub on completion
- Autonomous monitoring via local cron (10-min interval)

**Actual training runs**:

| Model | Hardware | Time | Throughput | Peak VRAM | Cost |
|-------|----------|------|-----------|-----------|------|
| EkiTil-123M | 1× H100 80GB | 3.8h | 180K tok/s | 42.6 GB | ~$10 |
| EkiTil-300M | 2× H100 80GB | 6.63h | 207K tok/s | 29.6 GB/GPU | ~$30 |
| **Total** | | **10.4h** | | | **~$40** |

### 4.3 EkiTil-123M Results

| Metric | Value |
|--------|-------|
| Final loss | **3.0748** |
| Final BPB | **4.436** |
| Total steps | 9,424 |
| Total tokens | 2.47B |
| Training time | **3.8 hours** |
| Peak VRAM | 42.6 GB |
| Throughput | 180K tok/s |
| Hardware | 1× NVIDIA H100 80GB HBM3 |
| Cost | ~$10 (RunPod) |

**Loss curve:**
```
Step      Loss    BPB     LR
   500    7.07    10.20   1.5e-4  (warmup)
 1,000    5.48     7.91   3.0e-4
 2,000    3.99     5.75   6.0e-4  (peak lr)
 3,000    3.56     5.13   5.76e-4
 4,000    3.40     4.91   5.09e-4
 5,000    3.25     4.69   4.10e-4
 6,000    3.18     4.59   2.97e-4
 7,000    3.13     4.51   1.90e-4
 8,000    3.10     4.47   1.08e-4
 9,000    3.07     4.43   6.4e-5
 9,424    3.07     4.44   6.0e-5  (final)
```

**HuggingFace**: [`stukenov/ekitil-core-qwen3-123m-kkru-base-v1`](https://huggingface.co/stukenov/ekitil-core-qwen3-123m-kkru-base-v1)

### 4.4 EkiTil-300M Results

| Metric | Value |
|--------|-------|
| Actual params | **245.9M** |
| Final loss | **2.925** |
| Final BPB | **4.220** |
| Total steps | 18,849 |
| Total tokens | 4.94B (2 epochs) |
| Training time | **6.63 hours** |
| Peak VRAM | 29.6 GB (per GPU) |
| Throughput | 207K tok/s |
| Hardware | 2× NVIDIA H100 80GB HBM3 |
| Cost | ~$30 (RunPod) |

**Loss curve (checkpoints):**
```
Step       Loss    BPB     LR        Epoch
   942     6.27    9.05    1.41e-4   0.10  (warmup)
 1,884     4.62    6.67    2.83e-4   0.20
 2,826     3.79    5.47    2.98e-4   0.30
 3,768     3.54    5.11    2.93e-4   0.40
 4,710     3.37    4.86    2.83e-4   0.50
 5,652     3.31    4.77    2.70e-4   0.60
 6,594     3.22    4.65    2.53e-4   0.70
 7,536     3.15    4.54    2.34e-4   0.80
 8,478     3.09    4.46    2.13e-4   0.90
 9,420     3.09    4.46    1.90e-4   1.00  (epoch 2 start)
11,304     3.01    4.34    1.43e-4   1.20
13,188     2.95    4.25    9.85e-5   1.40
15,072     2.96    4.27    6.21e-5   1.60
16,956     2.93    4.23    3.83e-5   1.80
18,849     2.93    4.22    3.00e-5   2.00  (final)
```

**Scaling improvement over 123M**: Loss 2.93 vs 3.07 (−4.6%), BPB 4.22 vs 4.44 (−5.0%)

**HuggingFace**: [`stukenov/ekitil-core-qwen3-300m-kkru-base-v1`](https://huggingface.co/stukenov/ekitil-core-qwen3-300m-kkru-base-v1)

---

## 5. Generation Examples

Inference on RTX 3090 24GB, `temperature=0.8, top_p=0.9, repetition_penalty=1.1`:

### 5.1 Kazakh Generation

**Prompt:** `Қазақстан — Орталық Азиядағы`

| Model | Output |
|-------|--------|
| **123M** | Қазақстан — Орталық Азиядағы ең ірі сауда орталығы. Жыл сайын әлемнің 300-ге жуық елімен сауда айналымы артып келеді. |
| **300M** | Қазақстан — Орталық Азиядағы ең ірі, әрі беделді мемлекет. Біздің ел Орталық Азияда тұңғыш рет біздің мемлекетіміздің тәуелсіздігін таныған, әлемдік қауымдастықтағы орны мен ролін жоғары бағалайды. |

**Prompt:** `Бүгін ауа райы өте`

| Model | Output |
|-------|--------|
| **123M** | Бүгін ауа райы өте маңызды. Дегенмен, ауа райы да тұрақсыз болып тұрады. Өйткені ауа температурасы түнде +10-15 градус болады. |
| **300M** | Бүгін ауа райы өте құбылмалы болып, түнде күн жылынады. Алдын ала болжам бойынша, бұл өңірлерде су тасқыны болмайды. |

### 5.2 Russian Generation

**Prompt:** `Искусственный интеллект — это`

| Model | Output |
|-------|--------|
| **123M** | Искусственный интеллект — это человеческий организм, который он использует в повседневной жизни. У человека есть своя система интеллекта. |
| **300M** | Искусственный интеллект — это основа всех наших усилий. Надеемся на то, что наше поколение будет использовать все лучшее для дальнейшего процветания страны. |

### 5.3 Translation (kk→ru)

**Prompt:** `<|kk|> Менің атым Сакен. <|translate|> <|ru|>`

| Model | Output | Quality |
|-------|--------|---------|
| **123M** | Смотреть | Poor — too few parallel examples at 123M scale |
| **300M** | После рождения я вышел в другой человек | Poor — parallel data was only 0.3% of corpus |

**Observation:** Both models produce fluent monolingual text in Kazakh and Russian. Translation capability is limited because parallel data comprised only 0.3% of the training corpus (7.2M tokens out of 2.47B). A dedicated translation fine-tune or more parallel data would be needed.

### 5.4 Qualitative Comparison

- **300M generates more coherent, longer passages** than 123M — consistent with lower perplexity
- Both models show **Kazakhstan-centric knowledge** (Nazarbayev, Astana, local news) reflecting the training data
- **123M hallucinates facts** more frequently (e.g., "Москва — столица Казахстана")
- Both models handle **Cyrillic script** natively with no character-level errors

---

## 6. Formal Evaluation Plan

### 5.1 Language Modeling

| Metric | Dataset | Languages |
|--------|---------|-----------|
| BPB (bits per byte) | Held-out split | kk, ru |
| Perplexity | Held-out split | kk, ru |

### 5.2 Downstream Tasks

| Benchmark | Task | Languages |
|-----------|------|-----------|
| KazMCQA | Multiple-choice QA | kk |
| Belebele | Reading comprehension | kk, ru |
| SIB-200 | Topic classification | kk, ru |
| XNLI | Natural Language Inference | ru |

### 5.3 Translation

| Benchmark | Task | Direction |
|-----------|------|-----------|
| FLORES-200 | BLEU score | kk→ru, ru→kk |

Translation quality is the key novel metric for EkiTil, enabled by the parallel training data with explicit `<|translate|>` markers.

---

## 7. HuggingFace Artifacts

| Repository | Type | Status |
|------------|------|--------|
| `stukenov/ekitil-corpus-annotated-kk-v1` | Dataset | Published |
| `stukenov/ekitil-corpus-parallel-kkru-v1` | Dataset | Published |
| `stukenov/ekitil-vocab-bpe-64k-kkru-v1` | Tokenizer | Published |
| `stukenov/ekitil-corpus-tokenized-kkru-v1` | Dataset | Published (v1, sentence-level) |
| `stukenov/ekitil-core-qwen3-123m-kkru-base-v1` | Model | **Published** (loss 3.07, BPB 4.44) |
| `stukenov/ekitil-core-qwen3-300m-kkru-base-v1` | Model | **Published** (loss 2.93, BPB 4.22) |
| `stukenov/ekitil-core-qwen3-300m-kkru-checkpoints` | Checkpoints | Published (step 8K, 16K) |
| `stukenov/ekitil-core-qwen3-600m-kkru-base-v1` | Model | Planned |

---

## 8. Roadmap

### 7.1 Data Pipeline (Complete)

- [x] Corpus annotation with language detection (121.9M sentences)
- [x] Parallel corpus collection kk↔ru (135K pairs)
- [x] BPE 64K tokenizer training (fertility 1.56)
- [x] Dataset tokenization v1 sentence-level (2.47B tokens, 1.2M blocks)

### 7.2 EkiTil-123M (Complete)

- [x] Pre-training on 1×H100 (3.8h, loss 3.07, BPB 4.44)
- [x] Publish model to HuggingFace
- [x] Model card / README
- [ ] Evaluation on benchmarks (BPB, KazMCQA, Belebele, FLORES)

### 7.3 EkiTil-300M (Complete)

- [x] Pre-training on 2×H100 DDP (6.63h, 2 epochs, loss 2.93, BPB 4.22)
- [x] HF checkpoint uploads (step 8K, 16K)
- [x] Publish model to HuggingFace
- [x] Model card / README
- [ ] Evaluation on benchmarks

### 7.4 EkiTil-600M (Planned)

- [ ] Train on 4×H100 (5 epochs, ~674M params, ~18h estimated)
- [ ] Alternatively: expand corpus to ~12B tokens first for single-epoch training
- [ ] Publish model to HuggingFace
- [ ] Evaluation on benchmarks

### 7.5 EkiTil-Chat (Future)

- [ ] SFT on bilingual instruction data
- [ ] DPO/RLHF alignment
- [ ] ChatML format support
- [ ] Publish chat model

### 7.6 Document-level Tokenization v2 (Future)

- [ ] Reassemble sentences back into documents by `doc_id`
- [ ] Filter by document language ratio (kk_ratio ≥ 0.7 or ru_ratio ≥ 0.7)
- [ ] Re-tokenize and re-pack into 2048-token blocks
- [ ] Retrain models on v2 data for better cross-sentence coherence

---

## 9. Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/exp027/annotate_kk_dataset.py` | Phase 1: corpus annotation + langdetect |
| `scripts/exp027/add_russian_and_parallel.py` | Phase 2: parallel corpus assembly |
| `scripts/exp027/train_tokenizer.py` | Phase 3: BPE 64K tokenizer training |
| `scripts/exp027/tokenize_dataset.py` | Phase 4 v1: sentence-level tokenization |
| `scripts/exp027/tokenize_documents.py` | Phase 4 v2: document-level tokenization |
| `scripts/exp027/train_ekitil_123m.py` | Legacy 123M training script |
| `scripts/exp027/train_ekitil.py` | **Unified training script** (123M/300M/600M, DDP, HF checkpoint upload) |
| `scripts/exp027/prepare_bilingual_data.py` | Bilingual data preparation |
| `scripts/exp027/launch_runpod.py` | RunPod pod management |
| `scripts/exp027/launch_training.py` | Automated pod creation + training launch |

---

## 10. Lessons Learned

Key insights from SozKZ experiments (exp001–exp028) that informed EkiTil design:

1. **From-scratch works at small scale**: exp013 (50M) and exp014 (150M) both converged well on Kazakh-only data
2. **Custom tokenizer is essential**: Llama/GPT tokenizers waste 2-3× on Kazakh text
3. **bf16, not fp16**: A10/A100 GPUs perform better with bfloat16 (exp004 finding)
4. **Chinchilla ratio matters**: exp028 (1.08B, 9B tokens, ratio 8.3:1) showed signs of underfitting; EkiTil-123M targets 19.8:1
5. **Verify HF round-trip before expensive runs**: exp028 lost $205 because QK-Norm weights were silently dropped by HuggingFace's LlamaForCausalLM
6. **DDP data coordination**: Rank 0 must download/tokenize data before other ranks start, or NCCL timeouts occur
7. **Pre-cache data before DDP launch**: Downloading 1.2M blocks takes ~10 min, which exceeds NCCL default timeout. Solution: run `download_data()` as a single-process step before `torchrun`
8. **H100 OOM at batch=32 with 64K vocab**: Logits tensor (32×2048×64000) = 15.6 GB. Solution: batch=16 + grad_accum=8 for same effective batch
9. **Multi-epoch training works**: EkiTil-300M trained 2 epochs on same data, loss improved from 3.09 (end of epoch 1) to 2.93 (end of epoch 2) — 5.2% additional improvement from the second pass
10. **Autonomous training pipeline**: Local cron monitoring every 10 min + cascading pod creation enables hands-off multi-model training with crash recovery

---

## 11. Related Work

- **SozKZ**: Kazakh-only language models (Llama architecture, 50M–1.08B)
- **Qwen3**: Base architecture family (Alibaba, 2025)
- **Chinchilla scaling laws** (Hoffmann et al., 2022): Optimal token-to-parameter ratio ~20:1
- **OPUS**: Open parallel corpus collection for translation
- **kz-transformers/multidomain-kazakh-dataset**: Primary Kazakh text source

---

## Changelog

| Date | Event |
|------|-------|
| 2026-03-26 | Data pipeline phases 1-3 completed (annotation, parallel, tokenizer) |
| 2026-03-27 | Phase 4 v1 (sentence-level tokenization) completed, 2.47B tokens |
| 2026-03-27 | Training script finalized, model architecture set to 123M |
| 2026-04-05 | Whitepaper created |
| 2026-04-05 | EkiTil-123M trained (1×H100, 3.8h), uploaded to HF |
| 2026-04-06 | EkiTil-300M trained (2×H100, 6.6h), uploaded to HF |
| 2026-04-06 | EkiTil-600M training started (4×H100), stopped by user at step 500 |
| 2026-04-06 | Model cards published for 123M and 300M |
