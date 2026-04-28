# exp_017: Morpheme-aware Byte-level BPE Tokenizer for Kazakh (100K)

**Date:** 2026-04-08
**Author:** Saken Tukenov
**Status:** In Progress

---

## Abstract

Standard byte-level BPE tokenizers (GPT-2, LLaMA) learn subword merges on raw byte streams without linguistic awareness. For agglutinative languages like Kazakh, this produces suboptimal tokenization: merge boundaries frequently bisect morphemes, creating semantically meaningless tokens and inflating token counts. We apply the morpheme-aware byte-level BPE approach introduced in HyperCLOVA X (NAVER, 2024) to Kazakh. The method pre-segments text using a morphological analyzer before BPE training, constraining merges to respect morpheme boundaries. We train a 100K-vocab tokenizer on a large Kazakh corpus and evaluate against standard byte-level BPE baselines.

## 1. Background

### 1.1 The Problem with Standard BPE for Agglutinative Languages

Kazakh is a Turkic agglutinative language where a single orthographic word can contain 5-7 morphemes:

```
үйлерімізде = үй + лер + іміз + де
(house)       (PL)  (POSS.1PL) (LOC)
"in our houses"
```

Standard byte-level BPE treats text as a flat byte stream. Merges are learned purely by frequency, ignoring morphological structure. This leads to:

- **Cross-morpheme tokens:** e.g., `лерім` spans the plural suffix and possessive — neither is a valid morpheme alone in this split
- **Inconsistent segmentation:** The same morpheme (e.g., `-лар/-лер` plural) gets different tokenizations depending on the stem it attaches to
- **Higher fertility:** More tokens needed per word compared to linguistically-informed segmentation
- **Worse downstream LM performance:** The model must implicitly re-learn morphological structure from token sequences

### 1.2 HyperCLOVA X Approach

The HyperCLOVA X Technical Report (NAVER, 2024) describes a morpheme-aware byte-level BPE tokenizer for Korean — another agglutinative language with similar challenges:

1. **Morphological pre-segmentation:** Input text is analyzed by a morphological analyzer (MeCab-ko for Korean). Words are split into constituent morphemes.
2. **Boundary marking:** Morpheme boundaries are preserved as separators during BPE training. BPE merges cannot cross these boundaries.
3. **Byte-level fallback:** The byte-level encoding ensures complete coverage — OOV characters are represented as byte sequences, guaranteeing no unknown tokens.
4. **Inference pipeline:** The same morphological pre-processing is applied at inference time before tokenization.

The result: subword units align with linguistic structure, improving both compression efficiency and downstream task performance.

### 1.3 Why This Matters for Kazakh

| Property | Korean | Kazakh |
|----------|--------|--------|
| Morphological type | Agglutinative | Agglutinative |
| Suffixes per word | 2-5 | 2-7 |
| Vowel harmony | No | Yes (front/back) |
| Writing system | Hangul (syllabic blocks) | Cyrillic (alphabetic) |
| Available morphological analyzer | MeCab-ko (mature) | Apertium-kaz (rule-based, good coverage) |

Kazakh's rich agglutinative morphology makes it an ideal candidate for this approach. Additionally, vowel harmony means suffix forms vary predictably (e.g., `-лар/-лер`, `-да/-де/-та/-те`), and a morpheme-aware tokenizer can learn these as unified morphemes rather than separate tokens.

## 2. Method

### 2.1 Morphological Analyzer: Apertium-kaz

We use [Apertium](https://apertium.org/) — an open-source rule-based machine translation platform with a mature Kazakh morphological transducer (`apertium-kaz`).

**Capabilities:**
- Full morphological analysis: stems, suffixes, POS tags
- High coverage for standard Kazakh text
- Deterministic, reproducible output
- Handles vowel harmony variants

**Example output:**
```
Input:  "Мектептегі оқушылар математиканы оқып жатыр"
Output: "Мектеп▁теғі оқу▁шы▁лар математика▁ны оқы▁п жат▁ыр"
```

Where `▁` marks morpheme boundaries within a word.

**Fallback strategy:** For words Apertium cannot analyze (neologisms, foreign borrowings, proper nouns), we pass them through unsegmented. Byte-level BPE handles these gracefully.

### 2.2 Training Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Raw Kazakh   │────>│ Apertium-kaz     │────>│ Morpheme-marked │────>│ Byte-level   │
│ Corpus       │     │ Segmentation     │     │ Text            │     │ BPE Training │
│ (~80M docs)  │     │ + Fallback       │     │                 │     │ (100K vocab)  │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
```

**Step 1: Corpus preparation**
- Source: `kz-transformers/multidomain-kazakh-dataset` (~80M rows) + `saken-tukenov/sozkz-corpus-clean-kk-text-v2`
- Filter: Kazakh-only text (`predicted_language == "kaz"` or all text from kk-only corpus)

**Step 2: Morphological segmentation**
- Process corpus through Apertium-kaz in streaming mode
- Insert morpheme boundary marker `\x1F` (ASCII Unit Separator) between morphemes within words
- Preserve word boundaries (spaces) and punctuation as-is
- Track coverage statistics: % of words successfully analyzed

**Step 3: BPE training**
- Use HuggingFace `tokenizers` library
- Byte-level BPE with custom pre-tokenizer that splits on both whitespace AND morpheme boundary marker
- 100,000 vocabulary size
- `min_frequency=2`
- Special tokens: `<|endoftext|>`, `<|padding|>`, `<|startoftext|>`

**Step 4: Tokenizer wrapping**
- Wrap as `PreTrainedTokenizerFast` for HuggingFace compatibility
- Custom `encode()` pipeline: raw text → Apertium segmentation → BPE encoding
- `decode()` strips morpheme markers transparently

### 2.3 Pre-tokenizer Design

The critical design choice is how to prevent BPE merges from crossing morpheme boundaries. We use a **custom pre-tokenizer** in the HuggingFace `tokenizers` library:

```python
pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(pattern=Regex(r"\x1F"), behavior="removed"),  # Split on morpheme boundary
    pre_tokenizers.ByteLevel(add_prefix_space=False),
])
```

This ensures:
1. Text is first split at morpheme boundaries (the `\x1F` marker)
2. Each morpheme is then encoded at the byte level
3. BPE merges can only happen within a single morpheme unit

### 2.4 Alternative: Apertium-free Approach

If Apertium proves too slow or has insufficient coverage, we have a fallback plan:

**Unsupervised morphological segmentation** using [Morfessor](https://github.com/aalto-speech/morfessor):
- Data-driven, no linguistic rules needed
- Train on Kazakh word frequency list
- Produces morpheme-like segments based on MDL (Minimum Description Length)
- Much faster than Apertium at inference time

We evaluate both approaches and compare.

## 3. Evaluation Plan

### 3.1 Metrics

| Metric | Description |
|--------|-------------|
| **Fertility** | Average tokens per word. Lower = better compression. |
| **Morpheme alignment** | % of token boundaries that coincide with true morpheme boundaries |
| **Consistency** | Same morpheme → same token(s) across different words |
| **Coverage** | % of vocabulary that corresponds to valid morphemes or morpheme sequences |
| **Compression ratio** | Bytes per token on held-out text |
| **Downstream perplexity** | Train a small LM with each tokenizer, compare perplexity |

### 3.2 Baselines

1. **kazakh-bpe-32k** — our existing 32K byte-level BPE (`saken-tukenov/kazakh-bpe-32k`)
2. **Standard BPE 100K** — byte-level BPE 100K without morpheme awareness (same corpus)
3. **LLaMA tokenizer** — `meta-llama/Llama-3.2-1B` tokenizer (128K vocab, multilingual)

### 3.3 Test Set

Hand-curated set of 100 Kazakh sentences covering:
- Simple words (1-2 morphemes)
- Complex agglutinated forms (4+ morphemes)
- Numbers, dates, mixed Kazakh-Russian text
- Technical/scientific vocabulary
- Colloquial speech

## 4. Expected Outcomes

1. **20-30% fertility improvement** over standard BPE on complex agglutinated words
2. **Higher morpheme alignment** — token boundaries should match morpheme boundaries >80% of the time
3. **Better suffix consistency** — plural `-лар/-лер`, case markers, possessive suffixes tokenized consistently
4. **Comparable or better compression ratio** overall (short words may see no improvement)

## 5. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Apertium too slow for 80M docs | Use Morfessor as fallback; parallelize with multiprocessing |
| Apertium low coverage on modern/slang text | Byte-level fallback handles OOV; measure and report coverage |
| Morpheme boundary marker leaks into final tokens | Validate in post-processing; marker is stripped in decode |
| Custom pre-tokenizer breaks HF compatibility | Test full save/load/push cycle before committing |
| Marginal improvement doesn't justify complexity | Compare fairly; document findings regardless of outcome |

## 6. Files

| File | Description |
|------|-------------|
| `morpheme_segmenter.py` | Apertium-kaz wrapper + Morfessor fallback |
| `train_tokenizer.py` | Main training script (morpheme-aware BPE 100K) |
| `evaluate_tokenizer.py` | Evaluation: fertility, alignment, consistency, comparison |
| `requirements.txt` | Dependencies |

## 7. References

1. HyperCLOVA X Technical Report. NAVER, 2024. — Morpheme-aware tokenization for Korean.
2. Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", ACL 2016. — Original BPE for NLP.
3. Wang et al., "Neural Machine Translation with Byte-Level Subwords", AAAI 2020. — BBPE.
4. Apertium: A free/open-source platform for rule-based machine translation. https://apertium.org/
5. Virpioja et al., "Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline", Aalto University, 2013.

## 8. Results

*To be filled after experiment completion.*
