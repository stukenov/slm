# Domain Pitfalls: arXiv Paper on Efficient Small LM for Kazakh

**Domain:** Low-resource language model training + efficiency-claim ML paper
**Researched:** 2026-03-20
**Confidence:** MEDIUM (domain expertise from literature patterns; web verification unavailable due to rate limits)

---

## Critical Pitfalls

Mistakes that will get the paper rejected, publicly criticized, or render claims invalid.

### Pitfall 1: Unfair Comparison — Apples to Oranges Model Evaluation

**What goes wrong:** Comparing a monolingual 600M model trained on Kazakh to a multilingual 7B+ model and claiming "efficiency" when the comparison is structurally unfair. Reviewers and Twitter will immediately flag: "Of course a model trained only on Kazakh does better on Kazakh per-parameter — it doesn't need to represent 100+ languages."

**Why it happens:** The efficiency narrative is tempting. Authors conflate "fewer parameters" with "more efficient" without controlling for training data volume, compute (FLOPs), or the multilingual tax.

**Consequences:** Core thesis collapses. Paper gets dismissed as "obvious result dressed up as contribution."

**Prevention:**
- Acknowledge the multilingual tax explicitly in Related Work or Discussion. Frame it as: "We demonstrate that for languages with sufficient monolingual data, a dedicated small model is a practical alternative to relying on multilingual models."
- Compare on equal footing where possible: report performance-per-FLOP, performance-per-training-token, and performance-per-inference-cost, not just performance-per-parameter.
- Include at least one monolingual Kazakh baseline if available (even a simple n-gram LM or LSTM for perplexity tasks) to show the Transformer architecture matters, not just the monolingual advantage.
- Compare against multilingual models of similar size (e.g., multilingual BERT-base at ~110M, or a small Gemma variant) to isolate size from language specialization.

**Detection:** If your "hero table" only compares your 600M model against 7B+ models, the comparison is unfair. The table must include models in the same parameter range.

**Phase mapping:** Evaluation pipeline design (before running benchmarks).

---

### Pitfall 2: Evaluation on Too-Few or Non-Standard Benchmarks

**What goes wrong:** Using 1-2 custom tasks (e.g., only perplexity + one downstream task) and claiming broad "Kazakh language understanding." Or using benchmarks that don't exist in standardized form for Kazakh and creating ad-hoc test sets with no validation.

**Why it happens:** Low-resource languages genuinely lack benchmarks. Authors either (a) test on very few tasks and overclaim, or (b) create custom benchmarks with no inter-annotator agreement, no standard splits, no prior baselines.

**Consequences:** Claims are unverifiable. No one can reproduce or compare. Paper contributes a result that exists in isolation.

**Prevention:**
- Use existing Kazakh benchmarks where they exist: KazNLP tasks, any Kazakh subsets in XTREME/XTREME-R, Kazakh portion of FLORES (translation), or Belebele (reading comprehension).
- For custom evaluations, document: dataset size, source, annotation process, inter-annotator agreement if human-annotated, exact train/val/test splits with hashes or identifiers.
- Cover at least 4 task types: generation quality (perplexity), classification (sentiment/topic), NLU (reading comprehension/NLI), and one generative task (summarization or translation).
- Report results on ALL tasks, including ones where the model loses. Cherry-picking wins destroys credibility.

**Detection:** If you can't name at least 3 distinct benchmarks with established baselines, the evaluation is too thin.

**Phase mapping:** Benchmark selection phase (before evaluation pipeline).

---

### Pitfall 3: Benchmark Contamination / Data Leakage

**What goes wrong:** The 23.6M sample training dataset may contain text that overlaps with benchmark test sets. For low-resource languages, the total available web text is small enough that train-test contamination is likely, not just possible.

**Why it happens:** Large web-scraped corpora for low-resource languages recycle the same sources. A Kazakh news article used in a sentiment benchmark might also appear in the training data. Nobody checks because the contamination detection tooling (e.g., n-gram overlap analysis) is less mature for non-English.

**Consequences:** Inflated benchmark scores. If discovered post-publication, the paper's results are invalidated.

**Prevention:**
- Run n-gram overlap detection (13-gram or higher) between training data and every benchmark test set. Report contamination rates in the paper (even if 0% — reporting the check itself adds credibility).
- For benchmarks derived from web sources, check URL/source overlap, not just text overlap.
- If contamination is found, report "clean" and "potentially contaminated" scores separately, or remove contaminated examples.
- Document the decontamination process in the paper's methodology section.

**Detection:** If you haven't run a single contamination check, assume contamination exists until proven otherwise.

**Phase mapping:** Evaluation pipeline phase (before reporting results). Must happen after benchmark selection but before final numbers.

---

### Pitfall 4: Missing or Misleading Efficiency Metrics

**What goes wrong:** Claiming "efficiency" based only on parameter count. A 600M model is not automatically "more efficient" than a 7B model — it depends on throughput, latency, memory, energy, and training cost.

**Why it happens:** Parameter count is the easiest number to cite. Actually measuring inference latency, throughput (tokens/sec), memory footprint, and training FLOPs requires extra work.

**Consequences:** Reviewers ask "efficient how?" and the paper has no answer. The efficiency claim — the paper's core thesis — becomes hand-waving.

**Prevention:**
- Report concrete efficiency metrics in a dedicated table:
  - Inference: tokens/sec on a standard GPU (e.g., A10, A100), batch size, latency (time-to-first-token, time-per-token)
  - Memory: peak GPU memory during inference at batch_size=1 and batch_size=32
  - Training: total GPU-hours, total FLOPs (estimate), cost in USD
  - Compare these metrics head-to-head with competitor models on the same hardware
- Frame efficiency as a tradeoff: "For X% of the performance, the model uses Y% of the compute/memory."
- Do NOT claim "matches" performance if the gap is >5% on key benchmarks. Use "approaches" or "achieves X% of the performance at Y% of the cost."

**Detection:** If your paper says "efficient" but the only number supporting it is parameter count, the claim is hollow.

**Phase mapping:** Evaluation pipeline (run inference benchmarks alongside task benchmarks).

---

### Pitfall 5: Translationese in Evaluation Data

**What goes wrong:** Using machine-translated or human-translated English benchmarks as Kazakh evaluation data. Translated text has different statistical properties than native text ("translationese") — simpler syntax, calques, unnatural word order. Models trained on native Kazakh text may perform differently on translated vs. native benchmarks, and scores won't reflect real-world utility.

**Why it happens:** Most NLP benchmarks originate in English. The fastest way to get a Kazakh benchmark is to translate one. Some "Kazakh" benchmarks in existing datasets are actually translations.

**Consequences:** Model may score well on translated benchmarks but poorly on native Kazakh text (or vice versa). Scores don't reflect actual Kazakh language capability.

**Prevention:**
- For each benchmark, document whether it is native Kazakh or translated. Flag translated benchmarks explicitly.
- Prefer native Kazakh benchmarks where they exist (e.g., Kazakh news classification, Kazakh Wikipedia-based tasks).
- If using translated benchmarks (e.g., FLORES for translation quality, Belebele for RC), acknowledge the translationese issue in the paper and discuss its impact.
- If creating custom benchmarks, use native Kazakh text sources (news, literature, social media), not translations.

**Detection:** Check the provenance of every benchmark dataset. If the "Kazakh" data was created by translating English data, it's translationese.

**Phase mapping:** Benchmark selection phase.

---

## Moderate Pitfalls

### Pitfall 6: Not Reporting Variance / Single-Run Results

**What goes wrong:** Reporting benchmark results from a single evaluation run without confidence intervals, standard deviations, or multiple seeds. Especially problematic for few-shot evaluation where prompt template choice and example selection can swing results by 5-10%.

**Prevention:**
- For few-shot evaluations: run with at least 3-5 different prompt templates and/or example orderings. Report mean and standard deviation.
- For generation tasks: report results on the full test set, not cherry-picked examples.
- For perplexity: this is deterministic, but report on the full test set, not a subset.
- If compute limits prevent multiple runs, state this explicitly: "Single-run results; variance analysis was not feasible due to compute constraints."

**Phase mapping:** Evaluation pipeline (built into the evaluation harness).

---

### Pitfall 7: Overclaiming the "From Scratch" Contribution

**What goes wrong:** Framing "we trained from scratch" as a major contribution when the architecture is standard Llama/GPT and the training procedure is standard next-token prediction. "From scratch" is only noteworthy if the paper also contributes novel architecture decisions, training procedures, or data curation techniques specific to Kazakh.

**Prevention:**
- Be precise about what is novel: the tokenizer design for Kazakh? The data curation pipeline? The scaling analysis across 50M-600M? The efficiency comparison methodology?
- Frame the contribution as the full pipeline + empirical evidence, not the act of training itself.
- Emphasize reproducibility: release model weights, tokenizer, training configs, and evaluation code. The pipeline IS the contribution.

**Phase mapping:** Paper writing (Introduction and Contributions sections).

---

### Pitfall 8: Ignoring Morphological Complexity in Tokenizer Evaluation

**What goes wrong:** Not evaluating whether the custom tokenizer (kazakh-bpe-32k) actually handles Kazakh morphology well. Kazakh is agglutinative — a single word can have many suffixes. A BPE tokenizer trained on insufficient data may over-segment Kazakh words, leading to longer sequences and worse effective context length compared to a larger multilingual tokenizer that has seen more Kazakh text.

**Prevention:**
- Report fertility (tokens per word) for your tokenizer vs. multilingual tokenizers (Llama tokenizer, Gemma tokenizer) on a held-out Kazakh text sample.
- Show example tokenizations of morphologically complex Kazakh words.
- If fertility is worse than multilingual tokenizers, acknowledge this and discuss the tradeoff (vocabulary efficiency vs. morphological coverage).
- Consider reporting "effective context length" — how many Kazakh words fit in 2048/4096 tokens with your tokenizer vs. competitors.

**Phase mapping:** Paper writing (Methodology section), but data should be collected during evaluation.

---

### Pitfall 9: No Qualitative Examples or Error Analysis

**What goes wrong:** Paper is all tables and numbers with no qualitative analysis. Reviewers and readers want to see what the model actually generates, what kinds of errors it makes, and where it fails compared to larger models.

**Prevention:**
- Include a qualitative analysis section or appendix with:
  - 3-5 generation examples (good AND bad) with comparison to a large model
  - Error categorization: what types of mistakes does the small model make? (factual errors, grammatical errors, repetition, code-switching to Russian, etc.)
  - Failure modes specific to model size: where does scale clearly matter?
- This is especially important for Kazakh because most reviewers won't read Kazakh — provide English translations alongside examples.

**Phase mapping:** Paper writing phase, but examples should be collected during evaluation.

---

### Pitfall 10: Not Addressing the "Why Not Fine-Tune?" Question

**What goes wrong:** Paper argues for training from scratch but doesn't compare against the obvious alternative: fine-tuning a multilingual model (Llama-3, Gemma) on Kazakh data. Reviewers will immediately ask: "Why not just fine-tune Gemma-2B on your Kazakh data? That would be a stronger baseline."

**Prevention:**
- Either include a fine-tuned baseline (e.g., Gemma-2B fine-tuned on your Kazakh corpus) or explain why this comparison is out of scope with a strong reason.
- If out of scope, acknowledge it as a limitation and future work item.
- Valid reasons: license restrictions on derivative models, the contribution is demonstrating the full pipeline, compute constraints.
- This is the single most likely "killer question" from reviewers. Have an answer.

**Phase mapping:** Evaluation pipeline (ideally, run at least one fine-tuning baseline). If not feasible, address in paper writing (Discussion/Limitations).

---

## Minor Pitfalls

### Pitfall 11: Poor Related Work Coverage

**What goes wrong:** Not citing the relevant low-resource LM literature, especially for Turkic languages. There are papers on Turkish, Uzbek, Azerbaijani models that share the agglutinative morphology challenge. Missing these suggests the authors didn't survey the field.

**Prevention:**
- Search specifically for: Turkic language models, Kazakh NLP, agglutinative language LMs, low-resource from-scratch training.
- Cite and position against: AraGPT (Arabic), SarvamAI (Indic languages), SEA-LION (Southeast Asian), AfricanVoices/AfroLM, and any Kazakh-specific work.

**Phase mapping:** Paper writing (Related Work section).

---

### Pitfall 12: Inconsistent Evaluation Setup Across Models

**What goes wrong:** Evaluating your model and competitors with different settings — different prompt templates, different number of few-shot examples, different max generation length, different sampling parameters. This silently biases results.

**Prevention:**
- Use identical evaluation settings for ALL models: same prompts, same few-shot examples, same decoding parameters (temperature, top-p, max tokens).
- Document all evaluation hyperparameters in the paper (appendix if needed).
- Use a standard evaluation harness (lm-evaluation-harness, or build one with fixed settings).
- For API-based models (GPT-OSS-120B), document API parameters used.

**Phase mapping:** Evaluation pipeline design.

---

### Pitfall 13: Not Reporting Training Data Statistics

**What goes wrong:** Saying "trained on 23.6M samples" without reporting: total tokens, domain distribution, deduplication method, data quality filtering, language identification accuracy. For Kazakh, web-scraped data often contains Russian text, code-switched text, or OCR artifacts.

**Prevention:**
- Report: total tokens (not just samples), domain breakdown (news/wiki/social/books), language purity (% actually Kazakh vs. Russian/mixed), deduplication rate, any quality filtering applied.
- If no language ID filtering was done, acknowledge this as a limitation.

**Phase mapping:** Paper writing (Data section), but analysis should be done during evaluation pipeline setup.

---

### Pitfall 14: Ignoring Inference Cost for API Models

**What goes wrong:** Comparing your model's inference cost against local inference of open models, but comparing against API cost for GPT-OSS-120B. This mixes two different cost models and makes the comparison misleading.

**Prevention:**
- Keep cost comparisons within the same deployment model: either all local inference costs or all API costs.
- If GPT-OSS-120B is only available via API, report its API cost separately and note the comparison is not apples-to-apples.
- Frame self-hosted cost honestly: include the cost of the GPU, not just the electricity.

**Phase mapping:** Evaluation pipeline (inference benchmarking section).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Benchmark Selection | Too few benchmarks, all translated from English | Prioritize native Kazakh tasks; aim for 4+ task types |
| Benchmark Selection | No existing baselines to compare against | Use multilingual benchmarks where baselines exist (FLORES, Belebele) |
| Evaluation Pipeline | Contamination in training data | Run n-gram overlap check before reporting any numbers |
| Evaluation Pipeline | Inconsistent settings across models | Build harness with fixed configs per benchmark |
| Evaluation Pipeline | Single-run variance | Build in multi-template evaluation for few-shot tasks |
| Results Analysis | Cherry-picking wins, hiding losses | Report ALL benchmarks, make a complete table, discuss losses |
| Paper Writing | Overclaiming efficiency without metrics | Dedicate a table to latency/throughput/memory, not just params |
| Paper Writing | Missing the "why not fine-tune?" answer | Either run one fine-tuning baseline or have a strong justification |
| Paper Writing | No qualitative examples | Collect examples during evaluation, include bilingual (KZ+EN) samples |
| Paper Submission | No code/model release plan | Release weights, configs, eval code, tokenizer on HuggingFace + GitHub |

## Sources

- Domain knowledge from patterns in published low-resource LM papers (AraGPT, AfroLM, SEA-LION, IndicBERT lineage)
- Known evaluation methodology critiques from the LM evaluation community (Eval Harness discussions, "Benchmark Contamination" literature by Sainz et al., Jacovi et al.)
- Agglutinative language tokenization challenges documented in Turkic NLP literature
- Web search verification was unavailable due to rate limiting; findings are based on training data patterns. Confidence: MEDIUM. Recommend verifying specific benchmark availability (KazNLP, Belebele Kazakh subset, FLORES Kazakh) via direct checks before committing to evaluation plan.
