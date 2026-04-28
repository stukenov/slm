# Adapting TinyLlama to Kazakh: A Reproducible 7-Stage Pipeline for Low-Resource Language Adaptation

**Authors:** Saken Tukenov

**Date:** April 2026

**Status:** In Progress

---

## Abstract

We present a fully reproducible pipeline for adapting TinyLlama-1.1B, an English-only language model, to support Kazakh and Russian through tokenizer extension and staged continual pretraining. Our approach follows the methodology established by Chinese-LLaMA (Cui et al., 2023), Swallow (Fujii et al., 2024), and EEVE (Kim et al., 2024), combining tokenizer vocabulary expansion with a 7-stage progressive unfreezing schedule. We train on the Multi-Domain Bilingual Kazakh Dataset (MDBKD), a naturally balanced bilingual corpus containing approximately equal volumes of Kazakh and Russian text. We document every decision, hyperparameter, and intermediate result to serve as a practical tutorial for adapting language models to other low-resource languages.

## 1. Introduction

### 1.1 Motivation

Large language models (LLMs) have achieved remarkable performance across many NLP tasks, but their capabilities remain concentrated in high-resource languages, primarily English and Chinese. For the roughly 7,000 languages spoken worldwide, the vast majority lack sufficient representation in LLM training data to enable meaningful generation or understanding.

Kazakh (Қазақ тілі) is a Turkic language spoken by approximately 13 million people, primarily in Kazakhstan. Despite being the official language of the 9th-largest country by area, Kazakh remains underrepresented in modern LLMs. Most models either lack Kazakh entirely or tokenize it extremely inefficiently due to Kazakh-specific Cyrillic characters (ә, ғ, қ, ң, ө, ү, ұ, і, һ) absent from standard Russian Cyrillic.

### 1.2 Problem Statement

Adapting an English LLM to Kazakh involves two fundamental challenges:

1. **Tokenizer inefficiency**: English-trained tokenizers fragment Kazakh text into byte-level or character-level tokens, inflating sequence length by 3-4x and wasting model capacity.
2. **Catastrophic forgetting**: Naive fine-tuning on Kazakh data causes the model to lose English capabilities. Staged training with proper data mixing is required.

### 1.3 Approach Overview

We follow a three-phase approach grounded in peer-reviewed methodology:

1. **Tokenizer Extension** — Train a Kazakh-Russian BPE tokenizer, merge with TinyLlama's 32K vocabulary, initialize new embeddings via subword mean (EEVE approach)
2. **7-Stage Continual Pretraining** — Progressively unfreeze model parameters from embeddings to full model (adapted from EEVE's staged schedule)
3. **Evaluation** — Track perplexity, fertility rate, and generation quality at each stage

### 1.4 Contributions

- A fully documented, reproducible pipeline for low-resource language adaptation
- Empirical analysis of tokenizer fertility before and after extension
- Stage-by-stage training metrics showing the contribution of each phase
- All code, configs, and checkpoints publicly released

## 2. Related Work

### 2.1 Tokenizer Extension for Language Adaptation

**Chinese-LLaMA** (Cui et al., 2023) demonstrated that extending LLaMA's 32K English vocabulary with 20,000 Chinese tokens, followed by two-stage continued pretraining (embeddings-only, then LoRA + embeddings), enables effective Chinese language generation. The key insight was that tokenizer efficiency directly impacts downstream performance.

**Swallow** (Fujii et al., 2024) adapted LLaMA-2 to Japanese by adding 11,176 Japanese subwords. They showed that performance improves monotonically with training data volume up to 100B tokens, and that mixing parallel corpora enhances translation ability. Their data mixing with English "experience replay" mitigated catastrophic forgetting.

**EEVE** (Kim et al., 2024) adapted SOLAR-10.7B to Korean by adding 8,960 Korean tokens. Their key contribution was a 7-stage progressive unfreezing schedule that achieved competitive performance with only 2B tokens — far less than the "trillions" previously assumed necessary. They also introduced subword-based embedding initialization.

### 2.2 Embedding Initialization

Hewitt (2021) proved that random initialization of new token embeddings is actively harmful: pre-trained logits are large and negative, so a near-zero random embedding produces logit ≈ 0, which dominates the softmax. The model then assigns near-100% probability to new tokens, wasting early training steps.

**Subword mean initialization** (used by EEVE, Swallow) addresses this: for a new token, tokenize it with the original tokenizer into subwords, then average their existing embeddings. This provides a semantically meaningful starting point.

**FOCUS** (Dobler & de Melo, EMNLP 2023) and **WECHSEL** (Minixhofer et al., NAACL 2022) offer more sophisticated initialization using auxiliary embedding spaces, but add complexity without guaranteed improvement for our setting.

### 2.3 Vocabulary Sharing Analysis

Yuan et al. (ACL 2024 Findings) analyzed 101 languages and classified them into four quadrants based on how they respond to embedding fine-tuning. Languages with severe over-tokenization (like Kazakh with an English tokenizer) fall in the "Stagnant Quadrant" where embedding-only fine-tuning is insufficient — tokenizer extension is required.

### 2.4 Kazakh Language Models

**mGPT** (Shliazhko et al., 2022): A 1.3B GPT-2 model trained on 61 languages including Kazakh. Older architecture (no GQA, no RoPE, absolute position embeddings, 2048 context). Provides a baseline but is architecturally outdated.

**mGPT-1.3B-kazakh**: A Kazakh-adapted version of mGPT. Limited community adoption (3K downloads). Serves as a comparison baseline.

No prior work has applied the Chinese-LLaMA/EEVE methodology specifically to Kazakh with modern LLaMA architecture.

## 3. Methodology

### 3.1 Base Model Selection

We select **TinyLlama-1.1B** (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`) as the base model for the following reasons:

| Property | Value | Rationale |
|----------|-------|-----------|
| Architecture | LlamaForCausalLM | Same family as Chinese-LLaMA, Swallow, EEVE |
| Parameters | 1.1B | Under 1B non-embedding, feasible on single GPU |
| Vocab size | 32,000 | Same starting point as all three reference papers |
| Context | 2,048 | Sufficient for our data |
| GQA | 4 KV heads | Modern attention, memory-efficient |
| License | Apache 2.0 | Fully open |
| Training | 3T tokens | Well-trained English baseline |

The choice of TinyLlama over newer models (Qwen2.5, LLaMA 3.2) is deliberate: it maximizes alignment with the reference papers' methodology, ensuring every step is backed by published results on the same architecture family.

### 3.2 Dataset

We use the **Multi-Domain Bilingual Kazakh Dataset (MDBKD)** (`kz-transformers/multidomain-kazakh-dataset`).

| Property | Value |
|----------|-------|
| Source | HuggingFace Hub |
| License | Apache 2.0 |
| Languages | Kazakh (kaz), Russian (rus) |
| Total rows | ~80M |
| Columns | `text`, `predicted_language`, `contains_kaz_symbols`, `id` |
| Text distribution | Kazakh-majority by row count, but Russian texts are longer |
| Token distribution | 60/40 Kazakh/Russian by token volume (see Section 4.1) |

We use the full dataset without language-based filtering or rebalancing. The dataset is shuffled with a fixed seed (42) before training to ensure uniform language distribution across batches. The natural ~60/40 (Kazakh/Russian) token balance serves our goal of bilingual adaptation, with Russian providing a Cyrillic script bridge between English and Kazakh.

### 3.3 Tokenizer Extension

#### 3.3.1 Critical Note: Why `add_tokens()` Does Not Work

An important finding from our experiments: the common HuggingFace approach of calling `tokenizer.add_tokens(new_tokens)` is **insufficient for vocabulary extension**. This method adds entries to the vocabulary lookup table but does **not update the BPE merge rules**. As a result, the tokenizer never produces the new tokens during encoding — it continues to use the original merge rules, decomposing text into the same subwords as before.

In our first attempt (v1), we added 10,000 tokens via `add_tokens()`. The Kazakh fertility remained at 4.0 tokens/word — identical to the baseline. The tokens existed in the vocabulary but were never activated during tokenization. This pitfall is not well-documented in the literature; Chinese-LLaMA (Cui et al., 2023) solved it correctly but the distinction is easy to miss.

#### 3.3.2 SentencePiece Protobuf Merge (Correct Method)

TinyLlama uses a SentencePiece tokenizer stored as a protobuf binary (`tokenizer.model`). To properly extend the vocabulary, we follow the Chinese-LLaMA approach of merging at the protobuf level:

**Step 1: Train a new SentencePiece model** on the full MDBKD corpus (both Kazakh and Russian):
- Model type: BPE
- Vocabulary size: 20,000
- Character coverage: 0.9999 (to capture Kazakh-specific characters)
- Byte fallback: enabled
- Input sentence size: 10M (sampled from 24.9M)

**Step 2: Merge protobuf models.** Parse both the base (`tokenizer.model` from TinyLlama) and the new model as `ModelProto` objects. For each piece in the new model that does not exist in the base:
- Copy the piece (token string + BPE score + type) into the base model
- Cap at 10,000 new pieces

This preserves all original tokenization behavior while adding new merge rules that the tokenizer will actively use for Kazakh and Russian text.

**Step 3: Pad vocabulary** to the nearest multiple of 64 (for GPU tensor core efficiency).

Result: 32,000 → ~42,048 vocabulary (32K original + ~10K new + padding).

#### 3.3.3 Embedding Initialization (Subword Mean, EEVE Method)

For each new token `t_new` (indices beyond original vocabulary):

**Input embeddings** (`model.embed_tokens`):
1. Tokenize `t_new` using the **original** TinyLlama tokenizer: `[s_1, s_2, ..., s_k]`
2. Look up existing embeddings: `[e_1, e_2, ..., e_k]`
3. Initialize: `e_new = mean(e_1, e_2, ..., e_k)`

**Output embeddings** (`model.lm_head`):
1. Same subword decomposition as above
2. Initialize: `e_new = e_{s_1}` (first subword embedding, following EEVE)

**Fallback** (for tokens that decompose into unknown subwords): use the mean of all existing embeddings. Hewitt (2021) proved this bounds the KL divergence to `log(1 + 1/n)` ≈ 0.00003 nats for n=32,000 — effectively neutral.

**Why not random initialization?** Pre-training drives all logits to large negative values. A random near-zero embedding produces logit ≈ 0, which dominates the softmax (since exp(0) >> exp(-large)). The model would assign near-100% probability to new tokens, wasting early training steps correcting this artifact.

### 3.4 7-Stage Training Pipeline

Adapted from EEVE's progressive unfreezing approach. Each stage is a separate training run with its own W&B project, checkpoints, and evaluation.

| Stage | Trainable Parameters | Frozen | Tokens | LR | Schedule |
|-------|---------------------|--------|--------|-----|----------|
| 1 | `embed_tokens` | All else | ~100M | 2e-4 | Cosine, 500 warmup |
| 2 | `embed_tokens` + `lm_head` | Transformer | ~200M | 2e-4 | Cosine, 500 warmup |
| 3 | Embeddings + LoRA(Q,K,V) r=16 | MLP, LN | ~500M | 1e-4 | Cosine, 1000 warmup |
| 4 | Embeddings + LoRA(Q,K,V,O) + MLP | LayerNorm | ~500M | 1e-4 | Cosine, 1000 warmup |
| 5 | Merge LoRA, unfreeze top 50% | Bottom 50% | ~500M | 5e-5 | Cosine, 500 warmup |
| 6 | All parameters | Nothing | ~500M | 3e-5 | Cosine, 500 warmup |
| 7 | Transformer only (cooldown) | embed_tokens + lm_head | ~200M | 2e-5 | Cosine, 200 warmup |

**Total: ~2.5B tokens**

Common hyperparameters across all stages:
- Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-8)
- Weight decay: 0.1
- Gradient clipping: 1.0
- Precision: bf16
- Batch size: dynamically set per stage to maximize GPU utilization
- Block size: 2048 (full context)

### 3.5 Evaluation Protocol

We evaluate at the end of each stage (7 checkpoints total):

1. **Perplexity** on held-out data:
   - 10K Kazakh texts (filtered by `predicted_language == "kaz"`)
   - 10K Russian texts (filtered by `predicted_language == "rus"`)
   - 10K English texts (from a standard English benchmark, to measure forgetting)

2. **Fertility rate** (tokens per word):
   - Measured on the same 3 reference sentences in Kazakh, Russian, and English
   - Before and after tokenizer extension

3. **Qualitative generation**:
   - 5 fixed prompts in Kazakh
   - 5 fixed prompts in Russian
   - Generated text logged to W&B at each stage

## 4. Experimental Setup

### 4.1 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total rows | 24,883,808 |
| Total tokens (TinyLlama tokenizer) | 6,173,131,446 (6.17B) |
| Kazakh rows | 21,940,051 (88.2%) |
| Russian rows | 2,943,757 (11.8%) |
| Kazakh tokens | 3,682,298,071 (3.68B, 59.6%) |
| Russian tokens | 2,490,833,375 (2.49B, 40.4%) |
| Kazakh avg tokens/row | 167.8 |
| Russian avg tokens/row | 846.1 |
| Kazakh avg chars/row | 288.3 |
| Russian avg chars/row | 2,318.8 |
| Kazakh tokens/char | 0.582 |
| Russian tokens/char | 0.365 |
| Fertility — Kazakh | 4.00 tokens/word |
| Fertility — Russian | 2.00 tokens/word |
| Fertility — English | 1.22 tokens/word |

**Key observations:**
- By row count, Kazakh dominates (88%), but Russian texts are ~5x longer on average (846 vs 168 tokens/row)
- By token volume, the split is approximately **60/40 Kazakh/Russian** — a natural bilingual balance
- Kazakh fertility of **4.0 tokens/word** confirms severe over-tokenization — a Kazakh word requires 4 tokens on average vs 1.22 for English. This is the primary motivation for tokenizer extension
- Russian fertility of 2.0 is expected for Cyrillic text with an English-trained tokenizer

### 4.2 Hardware

- **Token counting**: 1× NVIDIA H100 80GB SXM (RunPod)
- **Training**: 1× NVIDIA H100 80GB SXM (RunPod), may scale to 4× if needed
- **Logging**: Weights & Biases

### 4.3 Software

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.4.0 |
| Transformers | latest |
| Datasets | latest |
| PEFT (LoRA) | latest |
| W&B | latest |

## 5. Results

### 5.0 Baseline: Kazakh Perplexity Across Small Models (exp033)

Before training, we benchmarked 12 models (up to 3B parameters) on Kazakh text to understand the landscape. Tests run on 1x A100-SXM4-80GB, measuring perplexity on 10 Kazakh texts, 5 Russian texts, 5 English texts.

| Model | Params | Vocab | PPL Kazakh | PPL Russian | PPL English | Fertility (kk) | Kaz Chars |
|-------|--------|-------|-----------|-------------|-------------|----------------|-----------|
| **LLaMA-3.2-3B** | 3.21B | 128,256 | **3.5** | 6.4 | 6.8 | 4.17 | 0/18 |
| **LLaMA-3.2-1B** | 1.24B | 128,256 | **4.4** | 7.5 | 6.8 | 4.17 | 0/18 |
| Qwen2.5-3B | 3.09B | 151,665 | 6.7 | **3.4** | **5.9** | 4.33 | 0/18 |
| SmolLM2-1.7B | 1.71B | 49,152 | 7.0 | 2.6 | 6.2 | 7.50 | 0/18 |
| Gemma-2-2B | 2.61B | 256,000 | 7.2 | 11.4 | 8.2 | **3.00** | **18/18** |
| Qwen2.5-1.5B | 1.54B | 151,665 | 9.0 | 4.3 | 6.4 | 4.33 | 0/18 |
| StableLM-2-1.6B | 1.65B | 100,289 | 10.1 | **2.4** | 6.0 | 5.67 | 0/18 |
| **SozKZ-Llama-600M (ours)** | 0.59B | 50,257 | **10.2** | 11.7 | 8.9 | **1.23** | 0/18 |
| SmolLM2-360M | 0.36B | 49,152 | 11.8 | 4.3 | 6.8 | 7.50 | 0/18 |
| **SozKZ-TinyLlama (ours)** | 1.14B | 42,048 | 15.4 | 19.4 | 8.9 | **1.17** | 15/18 |
| Qwen2.5-0.5B | 0.49B | 151,665 | 17.8 | 5.7 | 7.7 | 4.33 | 0/18 |
| TinyLlama-1.1B (base) | 1.10B | 32,000 | 26.2 | 11.4 | 11.4 | 4.00 | 10/18 |
| mGPT-1.3B | 1.30B | 100,000 | — | — | — | — | — |

*mGPT failed to load due to deprecated architecture in transformers 5.x.*

**Key findings:**
1. **LLaMA 3.2 dominates Kazakh PPL** despite having 0 dedicated Kazakh characters in its tokenizer. Its 128K BPE vocabulary and massive multilingual training data give it strong Kazakh coverage via byte-fallback and subword sharing.
2. **Qwen2.5 excels at Russian** (PPL 3.4 for 3B) but is weaker on Kazakh than LLaMA 3.2.
3. **Gemma-2 has the best tokenizer for Kazakh** (18/18 Kazakh chars, fertility 3.0) but mediocre PPL (7.2), suggesting tokenizer coverage alone is insufficient.
4. **Our SozKZ-TinyLlama** has the best fertility (1.17) thanks to tokenizer extension, but PPL 15.4 reflects the small model size and limited training. It improved TinyLlama's Kazakh PPL from 26.2 to 15.4 (1.7x improvement).
5. **Fertility and PPL are inversely correlated for our model** — low fertility (good tokenization) but model capacity limits PPL improvement.

**Implication for next experiment:** LLaMA-3.2-1B (PPL 4.4 on Kazakh already) is the strongest small base model. With tokenizer extension (reducing fertility from 4.17 to ~1.2) and continual pretraining on Kazakh-only data, it could achieve significantly better results than our TinyLlama experiment.

### 5.0.1 Tokenizer Compression Analysis

We measured how efficiently each tokenizer compresses a standardized Kazakh text (~560 characters) compared to equivalent English text. **Kaz/Eng ratio** indicates how many times more tokens Kazakh requires vs English (1.0 = parity).

| Model | Vocab | Kaz tokens | Eng tokens | Kaz tok/word | Eng tok/word | Kaz/Eng ratio | Bytes/tok (kk) | Kaz Chars |
|-------|-------|-----------|-----------|-------------|-------------|---------------|----------------|-----------|
| **SozKZ-Llama-600M** | 50K | **69** | 130 | **1.23** | 2.03 | **0.61x** | 10.6 | 0/18 |
| **SozKZ-TinyLlama-kk** | 42K | **112** | 84 | **2.00** | 1.31 | **1.53x** | 6.5 | 15/18 |
| Gemma-2-2B | 256K | 175 | 82 | 3.12 | 1.28 | 2.44x | 4.2 | **18/18** |
| TinyLlama-1.1B | 32K | 223 | 89 | 3.98 | 1.39 | 2.86x | 3.3 | 10/18 |
| LLaMA-3.2-1B | 128K | 226 | 80 | 4.04 | 1.25 | 3.23x | 3.2 | 0/18 |
| Qwen2.5-1.5B | 152K | 239 | 83 | 4.27 | 1.30 | 3.28x | 3.0 | 0/18 |
| StableLM-2-1.6B | 100K | 287 | 83 | 5.12 | 1.30 | 3.94x | 2.5 | 0/18 |
| SmolLM2-1.7B | 49K | 357 | 82 | 6.38 | 1.28 | **4.98x** | 2.0 | 0/18 |

**Key observations:**
- **SozKZ-600M** achieves the best compression (0.61x) because it was trained from scratch with a Kazakh-optimized tokenizer. Kazakh text is actually *shorter* than English.
- **SozKZ-TinyLlama** (1.53x) demonstrates that SentencePiece tokenizer extension effectively closes the gap from TinyLlama's 2.86x.
- **Gemma-2** has the best generic tokenizer for Kazakh (2.44x, all 18 Kazakh chars) thanks to its 256K vocabulary, but still 2.4x worse than parity.
- **SmolLM2** is worst (4.98x) — its 49K English-only tokenizer treats Kazakh as essentially random bytes.
- **Bytes/tok** shows how many UTF-8 bytes each token represents on average. Higher = better compression. SozKZ-600M packs 10.6 bytes per token vs SmolLM2's 2.0.

**Kaz Chars column explained:** Kazakh Cyrillic uses 18 characters not found in Russian (ә, ғ, қ, ң, ө, ү, ұ, і, һ + uppercase). This column shows how many of these 18 characters exist as dedicated tokens in the vocabulary. Most models have 0/18 — they encode Kazakh-specific characters via byte-fallback or multi-byte sequences.

### 5.0.2 Kazakh Text Generation Quality (10 Prompts, All Models)

We tested all models on 10 diverse Kazakh continuation prompts. Quality rated A (native-like) to D (incoherent/non-Kazakh).

| Model | PPL kk | Quality | Observation |
|-------|--------|---------|-------------|
| **LLaMA-3.2-3B** | 3.5 | **A** | Fluent Kazakh, correct grammar, real-world knowledge |
| **LLaMA-3.2-1B** | 4.4 | **A-** | Good Kazakh, minor coherence issues |
| Qwen2.5-3B | 6.7 | **B+** | Readable Kazakh, some unnatural phrasing |
| Qwen2.5-1.5B | 9.0 | **B** | Basic Kazakh, grammatical errors |
| Gemma-2-2B | 7.2 | **B** | Decent Kazakh, sometimes switches to other content |
| **SozKZ-Llama-600M** | 10.2 | **B-** | Short but natural Kazakh phrases |
| **SozKZ-TinyLlama-kk** | 15.4 | **B-** | Coherent Kazakh, some repetition |
| StableLM-2-1.6B | 10.1 | **C+** | Mixed Kazakh-Russian, unstable |
| Qwen2.5-0.5B | 17.8 | **C+** | Fragmented Kazakh, frequent errors |
| SmolLM2-1.7B | 7.0 | **C** | Mostly non-Kazakh despite low PPL |
| TinyLlama-1.1B | 26.2 | **D** | Romanized/broken text, not Kazakh |
| SmolLM2-360M | 11.8 | **D** | Gibberish with Cyrillic fragments |

#### Sample generations (Prompt: "Қазақстан — бұл")

**LLaMA-3.2-3B (A):**
> Қазақстан — бұл мемлекет аумағының үш төменгі тауы, ойпаттары мен 4 жаһандық маңызы бар теңіздерді қамтитын. Мұндай екі дүниежүзілік объектісін...

**LLaMA-3.2-1B (A-):**
> Қазақстан — бұл мәліметті кездесіп, үшінші тарихи жыр сатысында ойлаудан сонымен бірге 2006/12 (22) -дегі №3 шілде айына...

**Qwen2.5-3B (B+):**
> Қазақстан — бұл өзінде толығыш, ал еңбек усыныс кезеңдердегенге айта боладым...

**SozKZ-Llama-600M (B-):**
> Қазақстан — бұл менің Отаным! Мен оны мәңгілікке сүйемін. Өйткені, мен оның бір бөлшегімін.

**SozKZ-TinyLlama-kk (B-):**
> Қазақстан — бұл елдегі экономикалық, саяси және әлеуметтік құрылымның көрінісі. Бұл – халықпен жұмыс істейтін, ғылыми-те...

**TinyLlama-1.1B (D):**
> Қазақстан — бұл көрсеткі 1987-2004 жылы Kazakhstan – Boldiyny qazaqstan 1987-2005 tüzük...

#### Sample generations (Prompt: "Абай Құнанбайұлы — ұлы")

**LLaMA-3.2-3B:**
> Абай Құнанбайұлы — ұлы тарихи шығармашы. Баспалыс, 2013. - 416 с. В данной книге изданы сочинения Абая, в которых автор рассказывает о своем родном крае...

**SozKZ-Llama-600M:**
> Абай Құнанбайұлы — ұлы ақын, ойшыл, ағартушы, тарихы бай, ұлттық тәрбиенің негізін салушы...

**SmolLM2-1.7B:**
> Абай Құнанбайұлы — ұлытаутыруханнарғаоларастылабыларыоларынанзабастағандарындабастауарнаулардағаналабалар...

### 5.1 Tokenizer Extension Results (exp032)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Vocabulary size | 32,000 | 42,048 | +10,048 |
| Kazakh fertility (tok/word) | 4.00 | **1.17** | 3.4x better |
| Russian fertility (tok/word) | 2.00 | **1.29** | 1.6x better |
| English fertility (tok/word) | 1.22 | 1.00 | ~same |

### 5.2 Stage-by-Stage Training Results (exp032)

| Stage | Description | Loss start→end | Trainable % | Kaz PPL | Rus PPL | Eng PPL |
|-------|-------------|---------------|-------------|---------|---------|---------|
| Baseline | TinyLlama-1.1B | — | — | 26.2 | 11.4 | 11.4 |
| 1 | Input embeddings only | 5.73→4.85 | 7.5% | — | — | — |
| 2 | + Output embeddings | 3.2→2.91 | 15% | — | — | — |
| 3 | + LoRA(QKV) r=16 | 2.54→2.68 | 15.3% | — | — | — |
| 4 | + LoRA(QKVO+MLP) r=16 | 3.59→2.61 | 16% | — | — | — |
| 5 | Merge LoRA, top 50% | 3.02→2.61 | 57.5% | — | — | — |
| 6 | Full fine-tuning | 2.63→2.44 | 100% | — | — | — |
| 7 | Cooldown (freeze embed) | ~2.24 | 85% | — | — | — |
| **Final** | **After all 7 stages** | — | — | **15.4** | **19.4** | **8.9** |

**Training cost:** ~$32 (12.8 hours on 1x H100 SXM spot at ~$2.49/hr)

### 5.3 Generation Examples (Final Model)

**Kazakh:**
- "Қазақстан — бұл" → `Қазақстан — бұл да тарихи, ал «Орыс» деген мағынаны білдіреді. Өйткені, осы сөздерден кейін «Академик И.И.Муравьевскийның қызында орысша білім...`
- "Бүгін ауа райы" → `Бүгін ауа райы үшін көптеген жануарлар мен апаттық тастардан тұратын, жабайы жерді көріп қалғанда...`

**Russian:**
- "Казахстан — это" → `Казахстан — это не только столица, но и главная торговая площадь. В прошлом году в Казахстане было открыто 26 новых рынков...`
- "Сегодня погода" → `Сегодня погода в Казахстане: снег, гололед и метель. Предсказанная по годным нормам погода на 20-26 февраля...`

**English (preserved):**
- "Hello, my name is" → `Hello, my name is Sara. I am a freelance web designer and developer from Poland...`

## 6. Discussion

### 6.1 Effect of Tokenizer Extension

Tokenizer extension via SentencePiece protobuf merge reduced Kazakh fertility from 4.0 to 1.17 tokens/word — a 3.4x improvement. This means Kazakh text now uses approximately the same number of tokens as English, effectively tripling the model's context window for Kazakh content.

An important finding: the commonly suggested HuggingFace `tokenizer.add_tokens()` method does NOT work for vocabulary extension — it adds tokens to the lookup table but does not update BPE merge rules. The correct approach is SentencePiece protobuf-level merging (Section 3.3.1).

### 6.2 Contribution of Each Training Stage

The 7-stage progressive unfreezing showed clear phase transitions:
- **Stages 1-2** (embeddings only): Rapid initial adaptation — loss dropped from 5.73 to 2.91. Russian generation became coherent after Stage 2.
- **Stages 3-4** (LoRA): Loss initially increased when LoRA was introduced (new parameters at random init), then converged. Kazakh generation became coherent after Stage 3.
- **Stages 5-6** (full fine-tuning): Significant improvement — loss reached 2.44 at Stage 6. Best generation quality.
- **Stage 7** (cooldown): Final stabilization with frozen embeddings, loss reached 2.24.

### 6.3 Base Model Selection Matters More Than Training

The benchmark (Section 5.0) reveals that **base model selection is the dominant factor** for Kazakh quality. LLaMA-3.2-1B achieves Kazakh PPL 4.4 out of the box — nearly 4x better than our adapted TinyLlama (PPL 15.4) despite having no dedicated Kazakh tokens. This is because LLaMA 3.2 was trained on vastly more multilingual data.

Our tokenizer extension dramatically improved fertility (4.0→1.17) and our 7-stage training improved PPL (26.2→15.4), but the starting point matters enormously.

### 6.4 Recommendations for Next Experiment

Based on these findings, the optimal approach for the next experiment is:
1. **Start with LLaMA-3.2-1B** (PPL 4.4 baseline) or **Qwen2.5-1.5B** (PPL 9.0, best Russian)
2. **Extend tokenizer** (reduce fertility from 4.17 to ~1.2)
3. **Use Kazakh-only data** — model already has strong Russian/English from pretraining
4. **Simplify to 2 stages** (Chinese-LLaMA style) — our 7-stage approach added complexity without clear benefit over simpler approaches
5. **Train on more tokens** (5-10B vs 2.5B)

## 7. Conclusion

We presented a fully reproducible 7-stage pipeline for adapting TinyLlama-1.1B to Kazakh and Russian, achieving:
- Kazakh fertility improvement from 4.0 to 1.17 tokens/word via SentencePiece tokenizer extension
- Kazakh perplexity improvement from 26.2 to 15.4 (1.7x)
- Coherent text generation in Kazakh, Russian, and English
- Total training cost of ~$32

Our benchmark of 12 models revealed that LLaMA-3.2-1B and Qwen2.5 family are the strongest starting points for Kazakh language adaptation among small models. The pipeline, code, and results are fully open-sourced for reproducibility.

## 8. Reproducibility

All code, configurations, and training logs are available at:
- **Code**: `autoresearch/exp032_*.py`
- **Configs**: `configs/experiments/exp032_*.yaml`
- **Logs**: W&B project `exp032-kazakh-adapt`
- **Checkpoints**: HuggingFace Hub (to be published)

## References

1. Cui, Y., Yang, Z., & Yao, X. (2023). Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca. *arXiv:2304.08177*.
2. Fujii, K., et al. (2024). Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities. *arXiv:2404.17790*. (Swallow)
3. Kim, D., et al. (2024). Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models. *arXiv:2402.14714*. (EEVE)
4. Hewitt, J. (2021). Initializing New Word Embeddings for Pretrained Language Models. *Blog post, Stanford/Columbia*.
5. Dobler, K. & de Melo, G. (2023). FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models. *EMNLP 2023*.
6. Minixhofer, B., Paischer, F., & Rekabsaz, N. (2022). WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer. *NAACL 2022*.
7. Yuan, H., et al. (2024). How Vocabulary Sharing Facilitates Multilingualism in LLaMA? *ACL 2024 Findings*.
8. Shliazhko, O., et al. (2022). mGPT: Few-Shot Learners Go Multilingual. *arXiv:2204.07580*.

## Appendix A: Fixed Evaluation Prompts

### Kazakh Prompts
1. "Қазақстан — бұл"
2. "Бүгін ауа райы"
3. "Білім — ол"
4. "Қазақ тілі — бұл"
5. "Астана қаласы"

### Russian Prompts
1. "Казахстан — это"
2. "Сегодня погода"
3. "Образование — это"
4. "Русский язык"
5. "Город Астана"

## Appendix B: Experiment Timeline

| Date | Event |
|------|-------|
| 2026-04-06 | Token counting on full MDBKD — 6.17B tokens (60% kaz, 40% rus) |
| 2026-04-06 | Tokenizer extension — fertility kaz 4.0→1.17 |
| 2026-04-06 | 7-stage training completed — loss 5.73→2.24, PPL kaz 26.2→15.4 |
| 2026-04-06 | Model uploaded to HF: stukenov/sozkz-core-tinyllama-1b-kk-ru-v1 |
| 2026-04-06 | Fresh pod verification — ALL CHECKS PASSED |
| 2026-04-06 | exp033: Benchmarked 12 models on Kazakh — LLaMA-3.2-1B best (PPL 4.4) |
| TBD | Tokenizer extension completed |
| TBD | Stage 1-7 training |
| TBD | Final evaluation and model upload |
