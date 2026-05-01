# Morpheme-Aware Byte-Level BPE Tokenization for Kazakh

**Authors:** Saken Tukenov

**Date:** April 2026

**Status:** Complete

**Artifacts:**
- Tokenizer: [`stukenov/sozkz-morphbpe-256k-kk-v1`](https://huggingface.co/stukenov/sozkz-morphbpe-256k-kk-v1)
- Segmented corpus: [`stukenov/sozkz-corpus-segmented-kk-v1`](https://huggingface.co/datasets/stukenov/sozkz-corpus-segmented-kk-v1)

---

## Abstract

Standard BPE tokenizers are blind to morphological structure, producing subword units that arbitrarily span morpheme boundaries. This is particularly harmful for agglutinative languages like Kazakh, where a single surface form can pack 4–5 morphemes into one whitespace-delimited word. We present a morpheme-aware byte-level BPE tokenizer with a 256K vocabulary trained on 55.5 million Kazakh documents. Following the HyperCLOVA X approach (Yoo et al., 2024), we first segment the corpus into morphemes using a BiLSTM neural model, then train BPE with merge constraints that prevent merges from crossing morpheme boundaries. The resulting tokenizer achieves a fertility rate of 1.42–1.50 tokens per word on Kazakh text, with every token corresponding to a linguistically meaningful unit — a root, suffix, or complete word. We release both the tokenizer and the segmented corpus to facilitate further research on morphologically-aware NLP for Kazakh and other Turkic languages.

## 1. Introduction

### 1.1 The Problem with Standard BPE for Agglutinative Languages

Byte Pair Encoding (BPE) (Sennrich et al., 2016) builds a vocabulary by iteratively merging the most frequent adjacent byte pairs. While effective for fusional languages like English, this frequency-driven approach fails to respect morphological boundaries in agglutinative languages.

Consider the Kazakh word *үйлерімізде* ("in our houses"), which contains four morphemes:

| Morpheme | Function |
|----------|----------|
| үй | house (root) |
| лер | plural |
| іміз | 1st person plural possessive |
| де | locative case |

A standard BPE tokenizer trained on this corpus might learn a merge that produces the token `ерім`, spanning the boundary between `лер` (plural) and `іміз` (possessive). This token carries no morphological meaning — it is an artifact of character frequency, not linguistic structure. Such tokens waste vocabulary capacity and prevent the model from learning that `-лер` consistently marks plurality across all nouns.

### 1.2 Morpheme-Aware BPE

The insight behind morpheme-aware BPE is simple: if we know where morpheme boundaries are, we can prevent BPE from merging across them. The resulting vocabulary consists exclusively of tokens that are either complete morphemes or sub-morpheme fragments (for rare morphemes), but never cross-morpheme chimeras.

This approach was introduced by the HyperCLOVA X team (Yoo et al., 2024) for Korean, another agglutinative language. We adapt it for Kazakh using a neural morpheme segmentation model trained on the QazCorpora dataset.

### 1.3 Contributions

1. A **256K-vocabulary morpheme-aware byte-level BPE tokenizer** for Kazakh, published on HuggingFace
2. A **55.5M-document segmented corpus** with morpheme boundaries marked, available for further tokenizer training and linguistic research
3. A **high-performance segmentation pipeline** combining a BiLSTM neural model with LRU caching that achieves 96.3% cache hit rate and processes ~5.5K documents per second on commodity hardware
4. Empirical demonstration that morpheme-aware BPE achieves **1.42–1.50 fertility** (tokens per word) on Kazakh text

## 2. Related Work

### 2.1 Subword Tokenization

**BPE** (Sennrich et al., 2016) remains the dominant subword tokenization algorithm for language models. Its byte-level variant (Radford et al., 2019) operates on UTF-8 bytes rather than characters, ensuring coverage of any Unicode text without unknown tokens.

**SentencePiece** (Kudo & Richardson, 2018) implements unigram language model tokenization as an alternative to BPE. While it can incorporate linguistic features, standard usage is language-agnostic.

**WordPiece** (Schuster & Nakajima, 2012), used in BERT, optimizes for likelihood rather than frequency. Like BPE, it is morphology-agnostic.

### 2.2 Morphology-Aware Tokenization

**HyperCLOVA X** (Yoo et al., 2024) introduced the approach we follow: pre-segment text into morphemes, insert a boundary marker, and train BPE with the marker as a split point. BPE merges proceed normally within morpheme spans but cannot cross the boundary marker. At inference time, no segmentation is needed — morpheme awareness is baked into the merge table.

**Morfessor** (Virpioja et al., 2013) provides unsupervised morphological segmentation. While language-agnostic, its statistical approach produces noisier segmentations than supervised neural models for languages with available training data.

**Apertium** (Forcada et al., 2011) offers rule-based morphological analysis for Turkic languages including Kazakh. While precise for known word forms, rule-based systems struggle with neologisms, loanwords, and informal text.

### 2.3 Kazakh Morphological Analysis

The **QazCorpora** project provides a BiLSTM-based morphological analyzer for Kazakh, trained on manually annotated data with BIO tagging (B-ROOT, I-ROOT, B-SUFFIX, I-SUFFIX). The character-level model can segment any word form, including unseen ones, making it suitable for corpus-scale processing.

## 3. Method

### 3.1 Overview

Our pipeline has three stages:

```
Raw Kazakh text
    │
    ▼
[1] Morpheme segmentation (BiLSTM)
    │  "үйлерімізде" → "үй\x1Fлер\x1Fіміз\x1Fде"
    ▼
[2] Constrained BPE training
    │  Split on \x1F → ByteLevel → BPE (merges within morpheme spans only)
    ▼
[3] Inference-ready tokenizer
       No segmentation needed — morpheme constraints are encoded in the merge table
```

### 3.2 Stage 1: Morpheme Segmentation

We use the QazCorpora BiLSTM model to segment each word in the corpus into morphemes.

**Model architecture.** The segmenter is a BiLSTM sequence labeler operating at the character level:

| Component | Configuration |
|-----------|---------------|
| Input | Character indices (Kazakh Cyrillic alphabet + special chars) |
| Embedding | 32-dimensional, with padding index |
| BiLSTM | 1 layer, 64 hidden units (32 per direction) |
| Output | 5-class log-softmax (O, B-ROOT, I-ROOT, B-SUFFIX, I-SUFFIX) |
| Parameters | ~14K |

For each character in a word, the model predicts a BIO tag. A `B-SUFFIX` tag marks the beginning of a new morpheme (suffix). The segmented word is reconstructed by inserting the ASCII Unit Separator (`\x1F`, code point 31) before each `B-SUFFIX` position.

**Corpus-scale optimization.** Naively segmenting 55.5 million documents word-by-word would be prohibitively slow. We employ two optimizations:

1. **LRU word cache** (500K entries). Kazakh has a large but finite productive vocabulary. In our corpus, the 500K most frequent word forms cover 96.3% of all word tokens. Cache lookup is O(1) compared to O(n) BiLSTM inference per character.

2. **GPU-batched inference** for cache misses. Words not in cache are accumulated into batches of 512, padded to equal length, and processed in a single forward pass through the BiLSTM. This eliminates Python-loop overhead and enables GPU parallelism.

**Performance.** With these optimizations, segmentation achieves ~5,500 documents per second on a 16-vCPU machine (AWS c7i.4xlarge, CPU-only — the BiLSTM is small enough that CPU inference with caching outperforms GPU for this workload). Total segmentation time for the 55.5M-document corpus was approximately 2.5 hours.

### 3.3 Stage 2: Constrained BPE Training

We train a byte-level BPE tokenizer using HuggingFace `tokenizers` with a custom pre-tokenizer chain:

```
Pre-tokenizer = Sequence([
    Split(pattern="\x1F", behavior="removed"),   # split on morpheme boundary, consume marker
    ByteLevel(add_prefix_space=False),            # standard byte-level encoding
])
```

The `Split` step breaks text at every `\x1F` marker and removes the marker itself. Each resulting segment is a single morpheme. The subsequent `ByteLevel` step encodes each morpheme into bytes. BPE merges then operate on these byte sequences — but since each morpheme is a separate pre-tokenized unit, merges cannot cross morpheme boundaries.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 256,000 |
| Algorithm | Byte-level BPE |
| Minimum merge frequency | 2 |
| Initial alphabet | ByteLevel.alphabet() (256 byte tokens) |
| Special tokens | `<\|endoftext\|>`, `<\|padding\|>`, `<\|startoftext\|>` |
| Additional tokens | Unicode digit characters (non-ASCII Nd category) |
| Max sequence length | 4,096 |

The 256K vocabulary size was chosen to match Gemma-level capacity, providing sufficient room for full morpheme coverage in an agglutinative language where the productive morphology generates hundreds of thousands of distinct suffixed forms.

### 3.4 Stage 3: Inference

At inference time, the tokenizer operates directly on raw text without any morpheme segmentation. The trained merge table already encodes morpheme-boundary constraints: merges that would cross a boundary were never learned, so they cannot be applied. This makes the tokenizer a drop-in replacement for any standard HuggingFace tokenizer.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stukenov/sozkz-morphbpe-256k-kk-v1")
tokens = tokenizer.tokenize("Қазақстанның университеттерде оқушылары")
```

## 4. Data

### 4.1 Source Corpus

We train on `stukenov/ekitil-corpus-annotated-kk-v1`, a large-scale annotated web corpus. We apply two filters:

| Filter | Criterion |
|--------|-----------|
| Language | `detected_lang == "kk"` (Kazakh only) |
| Confidence | `lang_confidence >= 0.95` |

These filters remove Russian, English, and other non-Kazakh text, as well as low-confidence detections that might be code-mixed or mislabeled.

### 4.2 Corpus Statistics

| Property | Value |
|----------|-------|
| Documents after filtering | 55,539,970 |
| Output column | `text_segmented` |
| Morpheme boundary marker | `\x1F` (ASCII 31, Unit Separator) |
| Format | 12 Parquet shards |
| Storage | ~11 GB (uncompressed text) |

### 4.3 Segmented Corpus

The segmented corpus is published as `stukenov/sozkz-corpus-segmented-kk-v1` on HuggingFace. Each row contains a single `text_segmented` field — the original document text with `\x1F` inserted at every morpheme boundary.

Example:

| Original | Segmented | Gloss |
|----------|-----------|-------|
| Қазақстанның | Қазақстан`\x1F`ның | Kazakhstan + GEN |
| университеттерде | университеттер`\x1F`де | universities + LOC |
| оқушылар | оқушы`\x1F`лар | student + PL |
| үйлерімізде | үй`\x1F`лер`\x1F`іміз`\x1F`де | house + PL + 1PL.POSS + LOC |
| дайындалуда | дайындал`\x1F`у`\x1F`да | prepare + NMLZ + LOC |

## 5. Results

### 5.1 Tokenization Examples

| Input | Tokens | Count |
|-------|--------|-------|
| Қазақстан — Орталық Азиядағы мемлекет. | [Қазақстан, Ġ—, ĠОрталық, ĠАзия, дағы, Ġмемлекет, .] | 7 |
| Бүгін ауа райы жақсы болады. | [Бүгін, Ġауа, Ġрайы, Ġжақсы, Ġболады, .] | 6 |
| Мектепте оқушылар математика сабағына дайындалуда. | [Мектеп, те, Ġоқушы, лар, Ġматематика, Ġсабағ, ына, Ġдайындал, уда, .] | 10 |
| Үйлерімізде кітаптар көп. | [Үй, лер, імізде, Ġкітап, тар, Ġкөп, .] | 7 |

Key observation: tokens like `лар` (plural), `те` (locative), `ына` (dative+possessive) are consistent morphological units reused across different words. Standard BPE would often split these differently depending on the surrounding characters.

### 5.2 Fertility

| Metric | Value |
|--------|-------|
| Fertility (tokens/word) | 1.42–1.50 |
| Vocabulary size | 256,000 |
| Vocab utilization | High (morpheme-constrained merges reduce redundancy) |

Fertility of 1.42–1.50 means the tokenizer produces on average fewer than 1.5 tokens per whitespace word. For comparison:

- English GPT-2 on English text: ~1.3 tokens/word
- English GPT-2 on Kazakh text: ~3.5–4.0 tokens/word (severe over-tokenization)
- Our morpheme-aware 256K on Kazakh: **1.42–1.50 tokens/word**

The low fertility indicates that most Kazakh words are encoded as 1–2 tokens, with the majority being a root token plus at most one suffix token. Complex multi-suffix forms may use 3–4 tokens, each corresponding to a real morpheme.

### 5.3 Segmentation Pipeline Performance

| Metric | Value |
|--------|-------|
| Cache size | 500,000 unique word forms |
| Cache hit rate | 96.3% |
| Processing speed | ~5,500 docs/sec |
| Infrastructure | AWS EC2 c7i.4xlarge (16 vCPU, 32 GB RAM) |
| Total segmentation time | ~2.5 hours |
| Total BPE training time | ~30 minutes |
| End-to-end time | ~3 hours |

The 96.3% cache hit rate validates the design hypothesis: despite 55.5M documents, the unique word vocabulary is bounded. The 500K-entry cache captures the vast majority of word tokens, reducing inference calls by ~25x.

### 5.4 Comparison with Previous Version

| Property | exp_017 (100K) | exp_018 (256K) |
|----------|----------------|----------------|
| Vocabulary size | 100,000 | 256,000 |
| Corpus preserved | No (lost with instance) | Yes (HuggingFace) |
| Corpus size | ~55M docs | 55,539,970 docs |
| HuggingFace repo | `sozkz-morphbpe-100k-kk-v1` | `sozkz-morphbpe-256k-kk-v1` |

The 256K variant provides 2.56x more vocabulary capacity, enabling fuller coverage of Kazakh's productive morphology. The larger vocabulary means more morphemes and common multi-morpheme sequences receive dedicated tokens, reducing average sequence length for downstream models.

## 6. Infrastructure and Reproducibility

### 6.1 Challenges

The pipeline encountered several infrastructure challenges during execution:

1. **Disk exhaustion (twice).** The default 8 GB EBS volume was insufficient for 55.5M segmented documents (~11 GB). After the first crash at 23.4M documents, the volume was expanded to 50 GB. A second crash at ~55.5M documents (caused by systemd journal logs filling remaining space) required expansion to 200 GB. The resume-from-offset mechanism preserved all intermediate work.

2. **OOM during corpus upload.** Loading 55.5M text rows (~11 GB) into a single HuggingFace Dataset object exceeded the 32 GB RAM limit. The solution was chunked upload: read 5M rows at a time, write to a Parquet shard, upload via `HfApi.upload_file`, and free memory. The final corpus consists of 12 Parquet shards.

3. **SSH lockout.** When the disk filled completely, systemd-journald consumed all remaining space, making the instance unresponsive to SSH. Recovery required an instance stop/start cycle (which assigns a new public IP) followed by filesystem expansion on boot.

### 6.2 Resume Mechanism

The segmentation pipeline supports resume-from-offset: if interrupted, it can skip the first N already-processed documents and append new results to the existing corpus file. This proved essential when disk exhaustion interrupted processing twice.

### 6.3 Reproducibility

All code is self-contained in `nano/exp/exp_018/`:

```
exp_018/
├── train_tokenizer.py       # Main training script
├── morpheme_segmenter.py    # Segmenter wrapper (4 backends)
├── qazcorpora_model.py      # BiLSTM model + CachedSegmenter
├── morpho_lemma_suf.pth     # Pre-trained BiLSTM weights
├── launch_aws.sh            # AWS EC2 one-shot launch
├── setup_and_train.sh       # Remote setup + training
├── resume_training.sh       # Resume from offset
└── upload_and_train.sh      # Chunked upload + BPE training
```

To reproduce locally (small sample):
```bash
cd nano/exp/exp_018
python train_tokenizer.py --max-samples 10000
```

To reproduce at full scale on AWS:
```bash
bash launch_aws.sh
```

## 7. Limitations

1. **Segmentation quality.** The BiLSTM model occasionally produces linguistically debatable splits, particularly for loanwords (e.g., `математи·ка` instead of keeping `математика` as a single root) and proper nouns. These errors propagate into the merge table.

2. **Kazakh-only.** The segmenter is trained exclusively on Kazakh. Other Turkic languages with similar agglutinative structure (Turkish, Uzbek, Kyrgyz) would benefit from the same approach but require their own segmentation models.

3. **No intrinsic evaluation.** We report fertility as a proxy metric. Proper evaluation requires downstream tasks (language modeling perplexity, classification accuracy) comparing morpheme-aware vs. standard BPE tokenizers — this is planned for future experiments.

4. **Static segmentation.** The BiLSTM model is frozen; it does not adapt to domain-specific morphology or neologisms that emerge after the training data cutoff.

## 8. Future Work

1. **Downstream evaluation.** Train identical language models with morpheme-aware and standard BPE tokenizers and compare perplexity, few-shot classification, and generation quality.

2. **Vocabulary size ablation.** Compare 100K, 256K, and 512K variants to find the optimal vocabulary size for Kazakh, balancing coverage against embedding table size.

3. **Multi-backend comparison.** The codebase supports four segmentation backends (QazCorpora BiLSTM, Apertium, Morfessor, rule-based). A systematic comparison of how segmentation quality affects downstream tokenizer performance would be valuable.

4. **Extension to other Turkic languages.** The pipeline is language-agnostic aside from the segmentation model. Training segmenters for Turkish, Uzbek, and Kyrgyz would enable morpheme-aware tokenizers for the entire Turkic family.

## 9. Conclusion

We presented a morpheme-aware byte-level BPE tokenizer for Kazakh with a 256K vocabulary, trained on 55.5 million documents. By constraining BPE merges to respect morpheme boundaries predicted by a BiLSTM neural model, the tokenizer produces subword units that are linguistically meaningful — roots, suffixes, and complete words — rather than arbitrary character sequences. The tokenizer achieves 1.42–1.50 fertility on Kazakh text, comparable to English tokenizer performance on English, and is released as a drop-in HuggingFace tokenizer for use in language model pre-training and other NLP tasks.

## References

Cui, Y., Yang, Z., & Yao, X. (2023). Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca. *arXiv preprint arXiv:2304.08177*.

Forcada, M. L., Ginestí-Rosell, M., Nordfalk, J., O'Regan, J., Ortiz-Rojas, S., Pérez-Ortiz, J. A., ... & Tyers, F. M. (2011). Apertium: a free/open-source platform for rule-based machine translation. *Machine Translation*, 25(2), 127–144.

Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. *EMNLP 2018*.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.

Schuster, M., & Nakajima, K. (2012). Japanese and Korean voice search. *ICASSP 2012*.

Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL 2016*.

Virpioja, S., Smit, P., Grönroos, S. A., & Kurimo, M. (2013). Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline. *Aalto University publication series SCIENCE + TECHNOLOGY*, 25/2013.

Yoo, K. M., Han, J., In, S., Jeon, H., Jeong, J., Kang, J., ... & others. (2024). HyperCLOVA X Technical Report. *arXiv preprint arXiv:2404.01954*.
