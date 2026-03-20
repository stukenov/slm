# Feature Landscape: arXiv Paper on Efficient Small LM for Kazakh

**Domain:** Academic ML paper (arXiv preprint) -- low-resource language model training
**Researched:** 2026-03-20
**Confidence:** MEDIUM (based on training data knowledge of paper conventions through May 2025; web verification unavailable due to rate limits)

## Table Stakes

Features readers/reviewers expect. Missing any of these = paper dismissed as incomplete or amateurish.

### Paper Sections

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Abstract (structured) | First thing anyone reads; determines if they continue | Low | 150-250 words. Must state: problem, approach, key result, implication |
| Introduction with motivation | Must explain WHY Kazakh needs its own model | Low | Frame: large multilingual models are wasteful/weak for low-resource languages |
| Related Work section | Reviewers check you know the field | Med | Cover: (1) low-resource LM papers, (2) Kazakh NLP prior work, (3) efficient/small LM work |
| Data section with corpus statistics | Reproducibility requirement | Low | Token counts, sources, dedup method, cleaning pipeline. Already documented in WHITEPAPER.md |
| Tokenizer description | Custom tokenizer = must justify and analyze | Med | Vocab size rationale, fertility analysis (tokens/word vs multilingual tokenizers), coverage |
| Model architecture details | Reproducibility | Low | Layer counts, hidden dims, attention heads for each model size. Already have this |
| Training details (hyperparameters) | Reproducibility | Low | LR, batch size, schedule, optimizer, hardware, wall-clock time. Already have this |
| Evaluation on multiple tasks | Core contribution -- must prove the claim | **High** | See Benchmarks section below |
| Comparison with baselines | Without this the paper has no point | **High** | Must compare against multilingual models (Gemma, Llama-3, Qwen, Mistral) on same tasks |
| Results tables with error bars/significance | Standard academic rigor | Med | At minimum: mean scores. Ideally: confidence intervals or multiple runs |
| Conclusion | Expected section | Low | Summarize findings, acknowledge limitations, future work |
| Limitations section | Now standard in ML papers (post-2023) | Low | Honest about what the model cannot do |

### Evaluation Tasks (Minimum Set)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Perplexity on held-out Kazakh text | Most basic LM metric | Low | Already have eval code in `src/slm/evaluate.py`. Must use same tokenizer or report bits-per-byte for fair cross-model comparison |
| Multiple-choice knowledge benchmark | Tests factual knowledge/reasoning | Med | Already have `kk-socio-cultural-bench-mc` (7111 questions). Current 150M score ~10% (below 25% random -- likely a prompting/format issue to debug) |
| Text classification (sentiment) | Standard NLU task | Med | Kazakh sentiment datasets exist: `kz-transformers/kazakh-sentiment` or similar. Already trained exp025 (SFT sentiment) |
| Named Entity Recognition | Standard NLU task | Med | KazNERD dataset (Kazakh NER, ~112K sentences, 25 entity types). Published by Yeshpanov et al. |

### Figures and Visualizations

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Scaling curve (loss vs params) | Shows scaling behavior across 50M-600M | Low | Plot train/eval loss for each model size |
| Comparison bar chart (your models vs baselines) | Visual proof of core claim | Low | Grouped bar chart: task x model |
| Training loss curves | Shows convergence behavior | Low | Already logged in experiments |
| Tokenizer fertility comparison | Justifies custom tokenizer | Med | tokens/word for your BPE-50K vs Llama tokenizer vs Gemma tokenizer on Kazakh text |

## Differentiators

Features that make the paper stand out. Not expected but valued -- and some are achievable with existing work.

### Strong Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Efficiency analysis (params vs performance) | Core narrative: small specialized > large general | Med | Plot: accuracy vs model size (log scale). Show where your 600M intersects with Gemma-2B, Llama-3-8B etc. This IS the paper's thesis |
| Bits-per-byte (BPB) instead of perplexity | Fair cross-tokenizer comparison -- perplexity is incomparable across different vocabularies | Med | BPB = cross_entropy_loss * tokens_per_byte. Essential for honest comparison with models using different tokenizers |
| Compute-matched comparison | Shows FLOP efficiency, not just param count | Med | Estimate training FLOPs for your models vs what competitors required. Strengthens the "efficient" claim |
| Scaling law analysis for Kazakh | Novel contribution -- scaling laws for low-resource agglutinative language | Med | Fit power law to your 50M/150M/300M/600M results. Compare exponent to Chinchilla/Kaplan for English |
| Tokenizer fertility deep-dive | Agglutinative languages fragment badly in multilingual tokenizers | Med | Show: Kazakh text requires 2-3x more tokens in Llama tokenizer vs your 50K BPE. This directly explains why small specialized models win |
| Open-source contribution framing | Community value -- models, data, tokenizer all released | Low | List all HuggingFace artifacts. Emphasize reproducibility |
| Ablation studies | Shows you understand what matters | Med | At minimum: (1) tokenizer vocab size impact, (2) data size impact (1B vs 5B vs 9B tokens) |

### Nice-to-Have Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Morphological analysis | Kazakh is agglutinative -- show model learns morphology | High | Probe model representations for case/number/tense. Interesting but heavy lift |
| Cross-lingual transfer analysis | Do Kazakh models transfer to other Turkic languages? | High | Evaluate on Turkish/Uzbek/Kyrgyz data. Compelling but scope creep |
| Human evaluation of generation | Gold standard for text quality | Med | 50-100 samples rated by native speakers on fluency, coherence, factuality |
| Downstream fine-tuning results | Shows model is useful as base for applications | Med | Fine-tune on 1-2 tasks (sentiment, NER) and show it matches larger fine-tuned models |
| Carbon footprint / cost analysis | Trendy, supports efficiency narrative | Low | Estimate kWh and CO2 for training. Compare with training a 7B model |
| Inference speed benchmarks | Practical value for deployment | Low | Tokens/sec on consumer hardware (single GPU, CPU). Small model = fast inference |

## Kazakh NLP Benchmarks and Datasets

**Confidence: MEDIUM** -- based on training data knowledge; specific dataset availability should be verified on HuggingFace.

### Known Benchmarks

| Benchmark/Dataset | Task | Size | Source | Confidence |
|-------------------|------|------|--------|------------|
| **kk-socio-cultural-bench-mc** | Multiple-choice QA (Kazakh culture/history) | 7,111 questions | kz-transformers (HuggingFace) | HIGH -- already used in project |
| **KazNERD** | Named Entity Recognition | ~112K sentences, 25 entity types | Yeshpanov et al., 2022 | MEDIUM -- well-cited in literature |
| **Kazakh sentiment** | Sentiment classification | Varies | Multiple sources on HF | MEDIUM -- project already has SFT for this |
| **FLORES-200** | Machine Translation (kk subset) | 1,012 sentences | Meta/NLLB project | HIGH -- widely used |
| **MASSIVE** | Intent classification / slot filling (kk subset) | ~16K utterances | Amazon, multilingual NLU | MEDIUM |
| **XL-Sum** | Summarization (kk subset) | ~6K articles | BBC Kazakh service | MEDIUM |
| **XCOPA** | Causal reasoning | 500 test | Ponti et al. -- check if kk included | LOW -- may not have Kazakh |
| **Belebele** | Reading comprehension MC | 900 passages | Meta, 122 languages including kk | MEDIUM -- good for reading comprehension |
| **SIB-200** | Topic classification | 204 per language | Adelani et al. | MEDIUM |

### Recommended Evaluation Suite (Prioritized)

**Tier 1 -- Must include (table stakes for the paper):**
1. **Perplexity / BPB** on held-out Kazakh text -- baseline LM quality metric
2. **kk-socio-cultural-bench-mc** -- already have infrastructure, Kazakh-specific
3. **Sentiment classification** -- already have fine-tuned model (exp025)
4. **KazNERD** -- standard NLU task, well-established dataset

**Tier 2 -- Should include (strengthens paper significantly):**
5. **Belebele** -- reading comprehension, Meta benchmark, covers Kazakh
6. **FLORES-200** -- translation quality proxy (even if not a translation model, BLEU on prompted translation shows language understanding)
7. **SIB-200** -- topic classification, easy to evaluate

**Tier 3 -- Nice to have:**
8. **XL-Sum** -- summarization (harder to evaluate, needs ROUGE)
9. **MASSIVE** -- intent classification
10. **Human evaluation** -- generation quality rated by native speakers

## Anti-Features

Features to explicitly NOT include in the paper.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Claiming SOTA on Kazakh | You almost certainly will not beat GPT-4/GPT-OSS-120B on absolute scores | Frame as "comparable quality at 100x fewer parameters" -- efficiency is the claim, not SOTA |
| Fine-tuning competitor models | Out of scope, and unfair comparison (you control fine-tuning quality) | Compare against publicly available checkpoints (base and instruct) |
| Exhaustive benchmark coverage | Diminishing returns past 4-5 tasks; paper becomes a benchmark report | Pick 4-6 diverse tasks that cover generation + understanding + classification |
| Multilingual evaluation | This is a Kazakh paper, not a multilingual one | Mention in future work that Turkic transfer is interesting |
| Prompt engineering for competitors | Cherry-picking prompts to make competitors look bad | Use standard prompt templates, report them, be transparent |
| Training new models for the paper | Scope creep -- already have months of experiments | Use existing checkpoints. If a model needs retraining, that is a red flag |
| GEC as primary benchmark | exp025 showed GEC fine-tune was weak | Mention GEC briefly in results, do not center the paper on it |
| Comparing with closed-source models (GPT-4, Claude) | Not reproducible, API costs, versions change | Stick to open-weight models: Gemma, Llama-3, Qwen, Mistral. Mention GPT-OSS-120B if open |

## Feature Dependencies

```
Tokenizer analysis --> Bits-per-byte calculation (need fertility stats to compute BPB)
Bits-per-byte --> Fair perplexity comparison (BPB enables cross-tokenizer comparison)
Multiple-choice eval --> Fix current scoring issue (10% << 25% random suggests bug)
Sentiment dataset --> Sentiment evaluation (need dataset, not just fine-tuned model)
KazNERD dataset --> NER evaluation (need to download and build eval pipeline)
All evaluations --> Results tables --> Comparison figures
Scaling curve --> Need all 4 model sizes evaluated on same tasks
Efficiency analysis --> Need competitor model sizes + scores on same tasks
```

## MVP Recommendation

The paper's minimum viable version needs exactly these features, in priority order:

### Must Complete First
1. **Fix MC benchmark scoring** -- 10% accuracy on 4-choice MC is below random (25%). This is almost certainly a prompt formatting or token-matching bug, not model quality. Debug before drawing conclusions.
2. **Bits-per-byte evaluation** -- compute BPB for all own models AND competitors on same held-out Kazakh text. This is the single most important metric for the paper's efficiency claim.
3. **Tokenizer fertility analysis** -- tokens/word comparison between your BPE-50K and Llama/Gemma tokenizers on Kazakh. This explains WHY small specialized models work.

### Then Build
4. **Sentiment classification evaluation** -- use existing fine-tuned model + evaluate competitors zero-shot
5. **KazNERD evaluation** -- download dataset, build simple eval, run on all models
6. **kk-socio-cultural-bench-mc** -- re-evaluate with fixed prompting on all models
7. **Belebele (kk subset)** -- reading comprehension, good diversity

### Then Write
8. **Results tables and figures** -- comparison tables, scaling curve, efficiency plot
9. **Paper sections** -- intro, related work, methodology, results, conclusion

### Defer to "Nice to Have"
- Human evaluation (time-consuming to organize)
- Morphological probing (interesting but tangential)
- Cross-lingual transfer (scope creep)
- FLORES translation eval (model not trained for translation)

## Comparable Papers to Study

These papers establish conventions for the genre. Study their structure and evaluation choices:

| Paper | Language(s) | Why Relevant | Key Evaluation Approach |
|-------|-------------|-------------|------------------------|
| AfriBERTa (Ogueji et al., 2021) | 11 African languages | Small model for low-resource, efficiency argument | NER, text classification, sentiment |
| Jais (Sengupta et al., 2023) | Arabic | From-scratch bilingual LM, comprehensive eval | Arabic benchmarks + general benchmarks, human eval |
| SEA-LION (AI Singapore, 2024) | Southeast Asian languages | Regional LM, multiple sizes | Multi-task benchmarks per language |
| YaLM (Yandex, 2022) | Russian/English | Large-scale from-scratch, open release | Perplexity, downstream tasks |
| GlotLID / MaLA-500 (various) | 500+ languages | Massively multilingual, covers Kazakh | Language ID, classification |
| TurkishBERTweet, BERTurk | Turkish | Turkic language (related to Kazakh), specialized model | NER, sentiment, POS tagging |
| Chinchilla (Hoffmann et al., 2022) | English | Scaling laws methodology | Loss vs compute curves, data-optimal training |

**Pattern from these papers:** The most impactful ones (1) pick 3-5 evaluation tasks, (2) compare against 5-10 baselines, (3) include an efficiency/cost analysis, and (4) release everything openly.

## Sources

- Project files: WHITEPAPER.md (experiment logs), eval scripts, results JSON files
- Training data knowledge (cutoff May 2025): paper conventions from ACL/EMNLP/NeurIPS proceedings
- KazNERD: Yeshpanov et al., "KazNERD: Kazakh Named Entity Recognition Dataset" (2022)
- Belebele: Bandarkar et al., "The Belebele Benchmark" (Meta, 2023)
- SIB-200: Adelani et al. (2023)
- FLORES-200: NLLB Team, Meta (2022)
- kk-socio-cultural-bench-mc: kz-transformers on HuggingFace (HIGH confidence -- used in project)

**Confidence notes:**
- HIGH: Paper section requirements, evaluation methodology conventions -- these are stable conventions
- MEDIUM: Specific Kazakh benchmark availability (KazNERD, Belebele kk subset) -- verify on HuggingFace before building eval pipeline
- LOW: XCOPA Kazakh inclusion -- needs verification
