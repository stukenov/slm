# Design Spec: Morphology-Aware Minimal-Edit GEC for Kazakh

**Date:** 2026-04-19
**Status:** Approved
**Scope:** Research subproject → arXiv preprint
**Directory:** `gec-paper/`

---

## 1. Project Structure

```
gec-paper/
├── configs/
│   ├── round1_nllb_baseline.yaml
│   ├── round2_nllb_organic.yaml
│   ├── round2_tagger.yaml
│   └── round3_ablation_*.yaml
├── src/
│   ├── taxonomy/
│   │   ├── schema.py           # 3-level taxonomy dataclasses
│   │   ├── classifier.py       # Auto-classifier by taxonomy
│   │   └── morph_analyzer.py   # apertium-kaz + Qwen-distilled wrapper
│   ├── data/
│   │   ├── synthetic.py        # Taxonomy-aware synthetic corruption
│   │   ├── organic_wiki.py     # Wikipedia kk edit history extractor
│   │   ├── organic_social.py   # Social media → GPT-4o correction pairs
│   │   ├── mixer.py            # Data mixture builder
│   │   └── multi_ref.py        # LLM multi-reference generator
│   ├── models/
│   │   ├── nllb_gec.py         # NLLB-200 fine-tuning for GEC
│   │   ├── edit_tagger.py      # Data-derived edit tag model
│   │   ├── morph_segmenter.py  # Qwen-distilled morpheme segmenter
│   │   └── reranker.py         # Quality estimation / reranking
│   └── eval/
│       ├── metrics.py          # EM, CER, Word F0.5, multi-ref F0.5
│       ├── benchmark.py        # Full benchmark runner
│       └── analysis.py         # Per-category breakdown, significance tests
├── scripts/
│   ├── collect_wiki_edits.py
│   ├── collect_social_data.py
│   ├── generate_synthetic.py
│   ├── generate_multi_ref.py
│   ├── train_nllb.py
│   ├── train_tagger.py
│   ├── train_morph_segmenter.py
│   ├── evaluate.py
│   └── run_pipeline.py         # Full dual-model inference pipeline
├── data/                       # Local data cache (gitignored)
├── paper/
│   ├── main.tex
│   ├── figures/
│   └── tables/
└── README.md
```

**HuggingFace naming (SozKZ convention):**
- Dataset: `stukenov/sozkz-gec-benchmark-kk-multiref-v1`
- NLLB model: `stukenov/sozkz-fix-nllb-600m-kk-gec-v1`
- Tagger: `stukenov/sozkz-fix-tagger-kk-gec-v1`
- Morpheme segmenter: `stukenov/sozkz-morph-qwen-500m-kk-v1`

---

## 2. Error Taxonomy — 3 Levels

```
Level 1 (3)          Level 2 (~15)              Level 3 (affix-level, for morphosyntax)
─────────────────────────────────────────────────────────────────────────────────────

ORTHOGRAPHY          spelling                   —
                     vowel_harmony              front_back_mismatch
                                                rounding_harmony
                                                boundary_harmony (stem+suffix boundary)
                     spacing                    missing_space, extra_space
                     punctuation                —

MORPHOSYNTAX         case                       nominative, genitive, dative,
                                                accusative, locative, ablative,
                                                instrumental
                     possessive                 person_mismatch (1/2/3),
                                                number_mismatch (sg/pl)
                     personal_ending            person, number, tense_agreement
                     plural                     extra_plural, missing_plural,
                                                allomorph (лар/лер/дар/дер/тар/тер)
                     negation                   double_negation, wrong_form
                     tense                      past/present/future confusion
                     postposition               wrong_postposition, case_government
                     agreement                  subject_verb, modifier_head
                     derivation                 wrong_derivational_suffix

SYNTAX_DISCOURSE     word_order                 verb_position, modifier_position
                     clause_structure           fragmented, run_on
                     missing_element            dropped_argument, missing_copula
                     redundant_element          repeated_word, pleonasm
                     discourse                  connector_misuse
```

- Level 3 only for MORPHOSYNTAX (80%+ of errors in agglutinative languages)
- Each example annotated as `(L1, L2, L3?)` tuple
- Taxonomy defined in `schema.py` as enum hierarchy
- Auto-classifier: rule-based for L1/L2, model-based for L3

---

## 3. Data Pipeline

### 3.1 Synthetic corruption (~50K pairs)
- Source: clean Kazakh texts from MDBKD (80M rows)
- Per `(L1, L2, L3)` error type — separate GPT-4o corruption prompt
- Balanced: ~3K per L2 category
- Verification: deterministic filters + round-trip check
- Cost: ~$30-40

### 3.2 Organic — Wikipedia edits (~25K pairs)
- Source: kk.wikipedia.org edit history dump
- Pipeline: download dump → extract diffs → filter (edit distance <20%, no vandalism, no structural edits) → auto-classify by taxonomy → deduplicate
- Free, realistic errors

### 3.3 Organic — Social media via GPT-4o (~12K pairs)
- Sources: Kazakh Telegram channels, Instagram/YouTube comments, Google Maps/2GIS reviews
- Pipeline: collect noisy texts → GPT-4o generates correction + taxonomy label → filter identity/large edits → sample 200 for manual quality check
- Cost: ~$15-20

### 3.4 Data mixture
```
Final dataset (~85-95K total):
├── synthetic:      50K  (~55%)
├── organic_wiki:   25K  (~27%)
├── organic_social: 12K  (~13%)
└── identity:        5K  (~5%)
```
Split: 90% train / 5% val / 5% test (test split fixed)

### 3.5 Multi-reference benchmark
- 700 sentences (500 from test + 200 new)
- GPT-4o generates 5 correction variants per sentence:
  1. Strict minimal-edit
  2. Conservative (fix only clear errors)
  3. Moderate (errors + minor fluency)
  4. Fluent rewrite
  5. Alternative valid correction
- Filter duplicates/invalid → typically 2-4 unique references per sentence
- Published as `stukenov/sozkz-gec-benchmark-kk-multiref-v1`

---

## 4. Dual-Model Architecture

### 4.1 NLLB-600M seq2seq (primary corrector)
- Base: `facebook/nllb-200-distilled-600M` (encoder-decoder, knows kaz_Cyrl)
- Fine-tuning: full fine-tune (not LoRA — 600M small enough)
- Input: `<kaz_Cyrl> {input_text}` (monolingual translation setup)
- With morphemes: `<kaz_Cyrl> бала|лар|ға мектеп|ке бар|ды`
- Inference: beam search (beam=5), length penalty, no_repeat_ngram
- Role: handles all error types, primary generation

### 4.2 Edit tagger (fast local corrector)
- Base: XLM-RoBERTa-base (280M)
- Architecture: token classification head → predicts edit tag per token
- Edit tags (data-derived):
  1. Extract all `(source_token, target_token)` pairs via alignment from training data
  2. Group transformations: `$KEEP`, `$DELETE`, `$REPLACE_{suffix}`, `$APPEND_{suffix}`, `$MERGE_NEXT`, `$SPLIT`
  3. Top-K most frequent (start K=2000, tune in ablation) → edit tag vocabulary
  4. Rare transformations → `$KEEP` (skip, leave for seq2seq)
- Inference: single forward pass, non-autoregressive, ~10-50x faster than seq2seq
- Role: high-precision local fixes (orthography, vowel harmony, suffix errors)

### 4.3 Inference pipeline
```
Input text
    │
    ▼
[Morpheme Segmenter] → segmented text
    │
    ▼
[Edit Tagger] → tagged corrections (high-confidence >0.9 applied)
    │
    ▼
[NLLB seq2seq] → full correction (complex/remaining errors)
    │
    ▼
[Reranker / QE] → select best output
    │
    ▼
Output text
```

Reranker selects between: tagger-only, NLLB-only, NLLB(tagger_output) cascaded.

### 4.4 Qwen 500M baseline
- Existing `sozkz-fix-qwen-500m-kk-gec-v2` retrained on new data
- Causal LM, format: `<TASK_FIX><SRC>{text}<SEP>`
- Same data, same eval for fair comparison

---

## 5. Morphology-Aware Representation

### 5.1 Apertium-kaz (rule-based baseline)
- Open-source: `apertium-kaz`
- Output: `балаларға` → `бала|лар|ға`
- Coverage: ~85-90% vocabulary, OOV left unsegmented
- Used in Round 1-2

### 5.2 Qwen-distilled morpheme segmenter
- Pipeline:
  1. 500K unique Kazakh wordforms from MDBKD
  2. Qwen 500M generates morpheme segmentation for each
  3. Sanity filter (morphemes must concatenate back to original)
  4. Train lightweight char-level seq2seq (~5-10M params) or fine-tune mT5-small
- Validation: compare with apertium-kaz, metric = morpheme boundary F1
- Used in Round 3

### 5.3 Integration modes (ablation)
- **Mode A (primary):** Input markers — `бала|лар|ға` — pipe char between morphemes
- **Mode B (ablation):** Auxiliary tags — `<STEM>бала<PL>лар<DAT>ға`

---

## 6. Evaluation Framework

### 6.1 Metrics

**Primary:**
| Metric | Multi-ref |
|--------|-----------|
| Word F0.5 | max across references |
| GLEU | mean across references |
| Exact Match | match any reference |
| CER | min across references |

**Secondary:** Identity Preservation, Over-correction Rate, Per-L1/L2 F0.5, Inference Speed

### 6.2 Baselines
| System | Type |
|--------|------|
| No correction | Identity |
| Qwen 500M (existing) | Causal LM |
| Qwen 500M (retrained) | Causal LM |
| GPT-4o (5-shot) | API LLM |
| NLLB-600M | Seq2seq |
| Edit Tagger | Tagger |
| NLLB + Tagger | Dual |
| NLLB + Tagger + Morph | Dual + morph |

### 6.3 Significance testing
- Bootstrap resampling (1000 iterations)
- Paired tests between systems
- Report p-values for key comparisons

---

## 7. Experiment Plan — 3 Rounds

### Round 1: Baseline (2-3 weeks)

| Step | Task | Compute | Output |
|------|------|---------|--------|
| R1.1 | Define taxonomy schema.py (2 levels initially) | Local | `taxonomy/schema.py` |
| R1.2 | Synthetic data: 10K pairs with L1/L2 tags via GPT-4o | $5-7 | Dataset v0 |
| R1.3 | Fine-tune NLLB-600M on 10K synthetic | RunPod 4090, ~2h | Model v0 |
| R1.4 | Fine-tune Qwen 500M on same 10K (baseline) | RunPod 4090, ~1h | Baseline |
| R1.5 | Single-ref eval on 500 test examples | Local | First metrics |
| R1.6 | GPT-4o few-shot baseline | $3-5 | Baseline numbers |

**Kill criterion:** If NLLB-600M significantly worse than Qwen 500M after fine-tune → reconsider backbone.

### Round 2: Full System (3-4 weeks)

| Step | Task | Compute | Output |
|------|------|---------|--------|
| R2.1 | Extend taxonomy to 3 levels | Local | Final `schema.py` |
| R2.2 | Wikipedia edit extractor → ~25K organic pairs | Local, ~1 day | Organic dataset |
| R2.3 | Social media → GPT-4o correction → ~12K pairs | $15-20 | Social dataset |
| R2.4 | Extend synthetic to 50K with full taxonomy | $25-30 | Synthetic v1 |
| R2.5 | Merge + deduplicate → final ~85K dataset | Local | Full dataset |
| R2.6 | Apertium-kaz morpheme segmentation of full dataset | Local | Segmented data |
| R2.7 | Fine-tune NLLB-600M on full data (± morphemes) | RunPod A100, ~4-6h | NLLB v1 |
| R2.8 | Extract edit tags from training data | Local | Tag vocabulary |
| R2.9 | Train XLM-R edit tagger | RunPod 4090, ~2h | Tagger v1 |
| R2.10 | Build dual pipeline (tagger → NLLB) | Local | Inference pipeline |
| R2.11 | Generate multi-ref benchmark (700 × 5 refs) | $10-15 | Benchmark v1 |
| R2.12 | Full eval: all systems × single/multi-ref | RunPod 4090 | Main results |

### Round 3: Ablation + Paper (2-3 weeks)

| Step | Task | Compute | Output |
|------|------|---------|--------|
| R3.1 | Train Qwen-distilled morpheme segmenter | RunPod 4090, ~1h | Morph model |
| R3.2 | Validate: Qwen-morph vs apertium-kaz (boundary F1) | Local | Comparison |
| R3.3 | Retrain NLLB with Qwen-morph segmentation | RunPod A100, ~4h | NLLB v2 |
| R3.4 | Ablation: data mix (synthetic / organic / mixed) | RunPod 4090, 3 runs | Ablation table |
| R3.5 | Ablation: morphemes (none / apertium / qwen) | RunPod 4090, 3 runs | Ablation table |
| R3.6 | Ablation: single vs dual vs cascade | Local (eval only) | Ablation table |
| R3.7 | Ablation: single-ref vs multi-ref gap | Local | Analysis |
| R3.8 | Train reranker (optional) | RunPod 4090 | Reranker |
| R3.9 | Error analysis: per-L2/L3 breakdown, failures | Local | Paper section |
| R3.10 | Paper write-up: LaTeX | Local | `paper/main.tex` |
| R3.11 | Upload final models + benchmark to HuggingFace | — | Public release |

### Estimated total cost: ~$100-120

---

## 8. Paper Structure

**Working title:** "Morphology-Aware Minimal-Edit GEC for Kazakh: A Dual-Model Pipeline with Multi-Reference Evaluation"

| # | Section | Content |
|---|---------|---------|
| 1 | Introduction | Agglutinative language GEC challenges, Kazakh as case study |
| 2 | Related Work | GECTurk, KoGEC, SCRIPT, Pillars of GEC, multilingual GEC with MT |
| 3 | Kazakh Error Taxonomy | 3-level schema, comparison with Korean/Turkish. **Contribution #1** |
| 4 | Data | 3 sources, pipeline, statistics, taxonomy distribution |
| 5 | Method | 5.1 Morpheme segmentation. 5.2 NLLB fine-tuning. 5.3 Edit tagger. 5.4 Dual pipeline |
| 6 | Multi-Reference Benchmark | Construction, agreement, gap analysis. **Contribution #2** |
| 7 | Experiments | Main results, per-category breakdown, baselines |
| 8 | Ablation Studies | Data mix, morphemes, single vs dual, single vs multi-ref |
| 9 | Analysis | Error analysis, failure modes, morpheme impact |
| 10 | Conclusion | Summary, recommendations for other agglutinative languages, future work |

**5 contributions:**
1. First 3-level error taxonomy for Kazakh GEC with affix-chain annotation
2. First multi-reference GEC benchmark for Kazakh (700 sentences, 2-4 refs each)
3. Morphology-aware dual-model pipeline (NLLB + data-derived tagger + morpheme segmentation)
4. Empirical single-ref vs multi-ref evaluation gap for agglutinative language
5. All artifacts public: taxonomy, dataset (~85K), benchmark, models, code

**Appendices:** Full taxonomy examples, data generation prompts, additional ablations, per-L3 breakdown
