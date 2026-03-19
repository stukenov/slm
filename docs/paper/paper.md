---
title: "SozKZ: Small Language Models for Kazakh Trained from Scratch"
authors: Saken Tukenov
date: 2026-02-14
tags: [kazakh, small-language-model, llama, low-resource-nlp, from-scratch, tokenizer]
---

# SozKZ: Small Language Models for Kazakh Trained from Scratch

**Saken Tukenov**

Independent Researcher — saken@tukenov.kz

February 2026

## Abstract

We present **SozKZ**, a family of small language models (50M and 150M parameters) trained from scratch for Kazakh — a low-resource Turkic language with approximately 13 million speakers. Unlike existing approaches that rely on multilingual models or domain adaptation of English-centric models, SozKZ builds dedicated infrastructure from the ground up: custom ByteLevel BPE tokenizers optimized for Kazakh Cyrillic text, multi-stage data cleaning pipelines, and Chinchilla-optimal training schedules. We show that standard multilingual tokenizers exhibit fertility rates of 4.9–6.6 tokens per word on Kazakh text, while our dedicated tokenizers achieve 1.3–1.4, a 3.6–4.9× improvement in encoding efficiency. Our 150M model achieves a perplexity of ~20 on a held-out Kazakh evaluation set and generates coherent, topically relevant Kazakh text across multiple domains. We release all models, tokenizers, datasets, and training code as open-source artifacts on Hugging Face Hub.

## 1. Introduction

Kazakh is the official language of Kazakhstan and is spoken by approximately 13 million people worldwide. Written in Cyrillic script (with an ongoing transition to Latin), Kazakh is an agglutinative Turkic language with rich morphology, vowel harmony, and flexible word order. Despite growing digital presence, Kazakh remains underrepresented in NLP resources compared to high-resource languages.

Existing multilingual models such as BLOOM [1], mGPT [2], and XLM-RoBERTa [3] include Kazakh data but allocate minimal vocabulary entries to Kazakh tokens. This leads to poor tokenization efficiency — common Kazakh words are split into many subword pieces or even individual bytes, increasing sequence length and degrading model quality. We quantify this problem: the GPT-NeoX tokenizer [4] produces 6.58 tokens per Kazakh word on average, compared to 1.34 for our dedicated Kazakh tokenizer — a 4.9× difference.

We hypothesize that for a low-resource language like Kazakh, dedicated small models with language-specific tokenizers can be more practical and efficient than relying on large multilingual models. Our contributions are:

1. **Custom Kazakh tokenizers** (32K and 50K vocabulary) with 3.6–4.9× better encoding efficiency than multilingual alternatives
2. **Curated training data**: a 9-stage cleaning pipeline applied to ~28M raw texts, yielding high-quality Kazakh corpora totaling ~22M unique documents
3. **Two language models** trained from scratch: 50M and 150M parameter LlamaForCausalLM [5] architectures with Chinchilla-optimal [6] training
4. **Quantitative tokenizer analysis**: fertility measurements comparing dedicated vs. multilingual tokenizers on Kazakh text
5. **Complete reproducibility**: all code, configs, data, tokenizers, and models are open-source

## 2. Related Work

**Multilingual Models for Kazakh.** BLOOM [1] and mGPT [2] are large multilingual models that include Kazakh in their training data but with limited vocabulary coverage. XLM-RoBERTa [3] provides multilingual representations but is not designed for text generation. The kz-transformers project provides the largest open Kazakh dataset (multidomain-kazakh-dataset, 23.6M samples), which we use as a primary data source.

**Tokenization for Low-Resource Languages.** Sennrich et al. [7] introduced BPE for neural machine translation, and subsequent work has shown that tokenizer quality significantly impacts model performance for morphologically rich languages. Rust et al. [8] demonstrated that language-specific tokenizers are crucial for downstream task performance in low-resource settings. Ahia et al. [9] showed that multilingual tokenizers often produce highly fragmented representations for underrepresented languages, directly impacting both training efficiency and model quality.

**Small Language Models.** Recent work has shown that sub-200M parameter models can achieve useful capabilities when trained with sufficient data and proper scaling laws. Pythia [4] provides a suite of models from 14M to 12B parameters for studying language model behavior. TinyLlama [10] demonstrated that 1.1B-parameter models can be competitive when trained on large token counts. Zhang et al. [11] introduced TinyStories showing that very small models can exhibit emergent abilities with high-quality data.

**Chinchilla Scaling.** Hoffmann et al. [6] established that compute-optimal training requires approximately 20 tokens per parameter. We follow this guideline: ~1B tokens for 50M params and ~2.88B tokens for 150M params.

**Data Cleaning for Language Models.** Penedo et al. [12] introduced comprehensive pipelines for web-crawled data cleaning, including deduplication strategies. Lee et al. [13] showed that deduplication significantly improves language model quality and reduces memorization. We adapt these approaches with Kazakh-specific filters including script profiling and fastText [14] language identification.

## 3. Data

### 3.1 Source Corpora

We aggregate Kazakh text from six major public datasets on Hugging Face:

| Source | Raw Samples | Unique (after dedup) |
|--------|------------|---------------------|
| CulturaX [15] (kk) | 2,731,934 | 2,705,991 |
| HPLT 2.0 [16] (kaz_Cyrl) | 2,637,330 | 2,246,264 |
| C4 [17] (kk) | 2,371,528 | 2,230,795 |
| MADLAD-400 [18] (kk) | 1,807,996 | 1,807,827 |
| mOSCAR (kaz_Cyrl) | 245,869 | 245,869 |
| Wikipedia (kk) | 238,356 | 238,343 |
| **Total new** | **10,033,013** | **9,475,089** |

Combined with the kz-transformers multidomain dataset (12.4M unique texts), this yields approximately 21.9M unique Kazakh text documents.

### 3.2 Cleaning Pipeline

Our 9-stage pipeline processes raw text into training-ready data:

1. **Unicode NFC normalization** — canonical decomposition and composition
2. **Kazakh character filtering** — retain only texts containing valid Kazakh Cyrillic characters (including Қ, Ғ, Ү, Ұ, Ә, Ө, Ң, І, Һ)
3. **Script profiling** — require Cyrillic character ratio ≥ 60%
4. **FastText language identification** [14] — filter for Kazakh language label
5. **Junk/boilerplate removal** — URL density (max 5 per 1000 chars), HTML tags (max 5), control characters
6. **Repetition filtering** — detect and remove repetitive content
7. **Length filtering** — minimum 50 characters
8. **Exact deduplication** — MD5 hash-based, cross-source and against reference corpus
9. **Near deduplication** — MinHash LSH for fuzzy duplicates [13]

The pipeline reduces 28.4M raw texts to 13.7M clean texts (48.2% pass rate), confirming that over half of ostensibly "Kazakh" web-crawled data is noise, non-Kazakh content, or duplicates.

### 3.3 Tokenization

We train two custom ByteLevel BPE tokenizers [7] using the HuggingFace Tokenizers library:

| Tokenizer | Vocab Size | Training Data |
|-----------|-----------|---------------|
| BPE-32K | 32,000 | 23.6M Kazakh texts |
| GPT2-50K | 50,257 | 78K clean v2 texts |

Both tokenizers use ByteLevel pre-tokenization with special tokens `<|endoftext|>` (BOS/EOS) and `<|padding|>` (PAD).

**Tokenizer Fertility Analysis.** To quantify the advantage of dedicated tokenizers, we measure fertility (tokens per word) across four tokenizers on 10 diverse Kazakh sentences covering news, science, literature, and conversational domains:

| Tokenizer | Vocab Size | Avg Tokens/Sentence | Chars/Token | Fertility (Tokens/Word) |
|-----------|-----------|--------------------:|------------:|------------------------:|
| GPT-NeoX (Pythia) [4] | 50,304 | 52.6 | 1.22 | 6.58 |
| LLaMA-2 [19] | 32,000 | 39.2 | 1.64 | 4.90 |
| SozKZ BPE-32K (ours) | 32,000 | 11.5 | 5.59 | 1.44 |
| SozKZ GPT2-50K (ours) | 50,257 | 10.7 | 6.01 | 1.34 |

The GPT-NeoX tokenizer effectively decomposes Kazakh text into individual bytes (1.22 chars/token), while our tokenizers capture whole words and common morphemes. This 3.6–4.9× reduction in sequence length directly translates to faster training, lower memory usage, and richer contextual representation within the same context window.

## 4. Models

We train two LlamaForCausalLM [5] models from random initialization:

### 4.1 Architecture

| Parameter | 50M Model | 150M Model |
|-----------|-----------|------------|
| Parameters | 50.29M | 149.8M |
| Hidden size | 512 | 896 |
| Layers | 8 | 12 |
| Attention heads | 8 | 16 |
| Intermediate (SwiGLU) [20] | 1,344 | 2,560 |
| Context length | 1,024 | 1,024 |
| Vocab size | 50,257 | 32,000 |
| Tied embeddings | Yes | Yes |
| Positional encoding | RoPE [21] | RoPE [21] |
| Normalization | RMSNorm [22] | RMSNorm [22] |

### 4.2 Training

Both models follow Chinchilla-optimal [6] training with AdamW optimizer [23]:

| Parameter | 50M Model | 150M Model |
|-----------|-----------|------------|
| Training tokens | ~1.04B | ~2.88B |
| Epochs | 1 | 1 |
| Learning rate | 6e-4 | 3e-4 |
| LR schedule | Cosine | Cosine |
| Warmup steps | 500 | 500 |
| Weight decay | 0.1 | 0.1 |
| Precision | bfloat16 | bfloat16 |
| Hardware | 1× A100 | 8× H200 |
| Training time | ~6-8 hours | ~36 minutes |
| Effective batch size | 32 | 512 |

### 4.3 Training Curves

**150M Model:**

| Step | Train Loss | Epoch |
|------|-----------|-------|
| 500 | ~5.5 | 0.046 |
| 2,500 | ~3.5 | 0.228 |
| 5,000 | ~3.2 | 0.456 |
| 7,500 | ~3.0 | 0.683 |
| 10,000 | ~2.9 | 0.911 |
| 10,972 (final) | ~2.9 | 1.000 |

**Final eval loss: ~3.0 | Perplexity: ~20**

**50M Model (v2):**
Trained on the cleaned v2 corpus with the GPT2-50K tokenizer. The model converges within a single epoch over ~1.04B tokens.

## 5. Evaluation

### 5.1 Perplexity

| Model | Eval Loss | Perplexity |
|-------|----------|-----------|
| 50M (step 24K, partial) | 3.39 | ~30 |
| 150M (final, 1 epoch) | ~3.0 | ~20 |

Note: These perplexity values are computed with our custom tokenizers and are not directly comparable to perplexity reported for multilingual models using different tokenizers, as tokenizer vocabulary size and encoding efficiency affect perplexity calculations.

### 5.2 Generation Quality

We evaluate generation quality using 10 diverse Kazakh prompts spanning news, weather, literature, science, and conversational domains.

**150M Model — Example Generations:**

*Prompt:* Қазақстан — бұл
*Output:* Қазақстан — бұл адамның өмір сүруіне қажетті жағдай жасайтын құрал. Біздің қоғамымызда осы мәселені шешуде мемлекеттің рөлі жоғары...

*Prompt:* Бүгін ауа райы
*Output:* Бүгін ауа райы болжамы «Қазгидромет» РМК 25 қыркүйекке арналған ауа райы болжамын жариялады...

*Prompt:* Алыстағы ауылда бір кәрі шал тұратын,
*Output:* Алыстағы ауылда бір кәрі шал тұратын, көпқабатты үйдің баласындай өсіп-өнген...

*Prompt:* Жасанды интеллект технологиялары
*Output:* Жасанды интеллект технологиялары атты халықаралық ғылыми-тәжірибелік конференциясы өтті...

Both models demonstrate:
- Correct Kazakh morphology and agglutinative structure
- Domain-appropriate vocabulary and register
- Coherent sentence-level generation
- Occasional topic drift and repetition in longer sequences

### 5.3 Qualitative Analysis of Training Dynamics

Through incremental evaluation during training (50M model, exp004), we observe clear quality progression:

- **Step 2,000** (perplexity ~70): Basic Kazakh words emerge. Heavy repetition and frequent Russian code-switching. Kazakh character ratio in output: 0.169.
- **Step 8,000** (perplexity ~39): Improved coherence, less language mixing. Domain-specific content appears (COVID statistics reflecting training data).
- **Step 22,000** (perplexity ~30): Complex academic vocabulary, better narrative structure, longer coherent spans. Reduced repetition.

This progression confirms that even at 50M parameters, the model progressively acquires Kazakh morphology, vocabulary, and domain knowledge throughout training.

## 6. Discussion

### 6.1 Domain-Adaptive Pre-Training vs. From-Scratch

Our initial experiments (exp001–003) attempted domain-adaptive pre-training (DAPT) on English-centric Pythia [4] models. The original GPT-NeoX tokenizer fragments Kazakh text into near-byte-level representations (fertility 6.58), meaning the model must learn Kazakh character composition in addition to language modeling. Training from scratch with a dedicated tokenizer proved far more effective, as the model can focus entirely on language semantics rather than orthographic reconstruction.

### 6.2 Tokenizer Design for Agglutinative Languages

Our fertility analysis (Section 3.3) demonstrates that tokenizer choice has dramatic impact for agglutinative languages. A 1,024-token context window effectively covers ~760 Kazakh words with our tokenizer versus only ~156 words with GPT-NeoX — a 4.9× difference in effective context length. This has direct implications for coherence in generation and the model's ability to capture long-range dependencies.

### 6.3 Data Quality and Scale

The 48.2% pass rate of our cleaning pipeline reveals that web-crawled "Kazakh" data contains substantial noise. Key issues include: Russian-Kazakh mixed content, boilerplate and templates, near-duplicate content across sources. The cross-source deduplication step alone removed 524K documents that appeared in multiple crawls. We recommend aggressive quality filtering for any low-resource language data pipeline.

### 6.4 Scaling Laws for Low-Resource Languages

Following Chinchilla-optimal ratios [6], the 150M model (perplexity ~20) substantially outperforms the 50M model (perplexity ~30), confirming that scaling laws hold for low-resource languages when sufficient data is available. Our 22M unique documents provide enough tokens for models up to approximately 500M parameters at Chinchilla-optimal ratios.

## 7. Limitations

- **Factual accuracy**: Models generate plausible but often factually incorrect text. They should not be used as knowledge sources.
- **Domain bias**: Heavy bias toward news and government press release style, reflecting training data distribution.
- **Language mixing**: Occasional Russian words or phrases appear mid-sentence, reflecting the bilingual nature of the training data.
- **Repetition**: Tendency toward repetitive patterns in longer generations, a known issue with small autoregressive models.
- **Context length**: Limited to 1,024 tokens.
- **No instruction tuning**: Base models only; not aligned for dialogue or instruction following.
- **Evaluation scope**: We report perplexity and qualitative generation analysis but do not evaluate on downstream tasks (NER, classification, translation) due to the lack of standardized Kazakh benchmarks suitable for generative models of this scale.

## 8. Conclusion

We demonstrate that small, dedicated language models trained from scratch can be a practical and effective approach for low-resource languages. The core insight is that tokenizer quality is the single most impactful factor: a 4.9× improvement in encoding efficiency translates directly to better training efficiency and generation quality.

The SozKZ project provides the Kazakh NLP community with:

- Two pre-trained language models (50M, 150M parameters)
- Two custom Kazakh tokenizers (32K, 50K vocabulary)
- Large-scale cleaned Kazakh corpora (~22M unique texts)
- Complete, reproducible training pipelines

All artifacts are released under open licenses on Hugging Face Hub under the `saken-tukenov/` namespace.

Future work includes: (1) instruction tuning for dialogue and task-following capabilities, (2) extending support to the new Latin-script Kazakh orthography, (3) scaling to larger model sizes with additional compute, and (4) evaluation on emerging Kazakh NLP benchmarks.

## Released Artifacts

### Models
| Name | Parameters | Hub Link |
|------|-----------|----------|
| sozkz-core-llama-50m-kk-base-v2 | 50.29M | [saken-tukenov/sozkz-core-llama-50m-kk-base-v2](https://huggingface.co/saken-tukenov/sozkz-core-llama-50m-kk-base-v2) |
| sozkz-core-llama-150m-kk-base-v1 | 149.8M | [saken-tukenov/sozkz-core-llama-150m-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-llama-150m-kk-base-v1) |

### Tokenizers
| Name | Vocab | Hub Link |
|------|-------|----------|
| sozkz-vocab-bpe-32k-kk-base-v1 | 32K | [saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-vocab-bpe-32k-kk-base-v1) |
| sozkz-core-gpt2-50k-kk-base-v1 | 50K | [saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1](https://huggingface.co/saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1) |

### Datasets
| Name | Size | Hub Link |
|------|------|----------|
| sozkz-corpus-dedup-kk-web-v1 | 9.5M texts | [saken-tukenov/sozkz-corpus-dedup-kk-web-v1](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-dedup-kk-web-v1) |
| sozkz-corpus-clean-kk-text-v2 | 78K docs | [saken-tukenov/sozkz-corpus-clean-kk-text-v2](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-text-v2) |
| sozkz-corpus-clean-kk-pretrain-v2 | 1.04B tokens | [saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2](https://huggingface.co/datasets/saken-tukenov/sozkz-corpus-clean-kk-pretrain-v2) |

### Code
- Training pipeline: [github.com/sakentukenov/slm](https://github.com/sakentukenov/slm)
- Standalone recipe: `nanochat-kazakh/` directory in the repository

## Citation

```bibtex
@misc{tukenov2026sozkz,
  title={SozKZ: Small Language Models for Kazakh Trained from Scratch},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/collections/saken-tukenov/sozkz}
}
```

## Acknowledgments

- **kz-transformers** project for the multidomain Kazakh dataset
- **Meta AI** for the LLaMA architecture
- **Hoffmann et al.** for Chinchilla scaling laws
- **vast.ai** for GPU infrastructure
- **Hugging Face** for the model hosting platform

## References

[1] Workshop, B. et al. (2023). BLOOM: A 176B-Parameter Open-Access Multilingual Language Model. arXiv:2211.05100

[2] Shliazhko, O. et al. (2022). mGPT: Few-Shot Learners Go Multilingual. arXiv:2204.07580

[3] Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL 2020. arXiv:1911.02116

[4] Biderman, S. et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML 2023. arXiv:2304.01373

[5] Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971

[6] Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models. NeurIPS 2022. arXiv:2203.15556

[7] Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL 2016. arXiv:1508.07909

[8] Rust, P. et al. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. ACL 2021. arXiv:2012.15613

[9] Ahia, O. et al. (2023). Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models. EMNLP 2023. arXiv:2305.13707

[10] Zhang, P. et al. (2024). TinyLlama: An Open-Source Small Language Model. arXiv:2401.02385

[11] Eldan, R. & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759

[12] Penedo, G. et al. (2023). The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only. NeurIPS 2023. arXiv:2306.01116

[13] Lee, K. et al. (2022). Deduplicating Training Data Makes Language Models Better. ACL 2022. arXiv:2107.06499

[14] Joulin, A. et al. (2017). Bag of Tricks for Efficient Text Classification. EACL 2017. arXiv:1607.01759

[15] Nguyen, T. et al. (2024). CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages. ACL 2024. arXiv:2309.09400

[16] de Gibert, O. et al. (2024). A New Massive Multilingual Dataset for High-Performance Language Technologies. LREC-COLING 2024. arXiv:2403.14009

[17] Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR 2020. arXiv:1910.10683

[18] Kudugunta, S. et al. (2024). MADLAD-400: A Multilingual and Document-Level Large Audited Dataset. AAAI 2024. arXiv:2309.04662

[19] Touvron, H. et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288

[20] Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202

[21] Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864

[22] Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS 2019. arXiv:1910.07467

[23] Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019. arXiv:1711.05101
