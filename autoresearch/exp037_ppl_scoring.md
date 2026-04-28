# exp037: PPL Quality Scoring of Kazakh Corpus

## Context

The project hit a data quality ceiling. We have ~13.7M cleaned Kazakh texts (sozkz-corpus-clean-v3) but no quality differentiation within the corpus — all texts that pass 9-stage cleaning filters are treated equally. Web crawl (80%+) dominates; literary/news texts are underrepresented. Previous models trained on this flat corpus showed diminishing returns.

**Problem statement:** How to extract more quality from the existing corpus without collecting new data?

## Decision: PPL-based Quality Scoring

After evaluating 7 methods for dataset quality improvement:

1. **PPL scoring & curriculum** (selected as first step)
2. Synthetic data generation (Self-Instruct)
3. Data augmentation without generation (fuzzy dedup, paragraph shuffling)
4. Cross-lingual knowledge transfer (translate more domains)
5. Synthetic instruction data (Evol-Instruct)
6. Data mixing & optimal ratios
7. Multilingual distillation

We chose PPL scoring first because:
- **Zero cost** — uses our own trained model, no API calls
- **Immediate impact** — quality tiers enable curriculum learning and domain reweighting
- **Foundation for other methods** — scored corpus enables quality-aware filtering for all downstream tasks
- **Measurable** — PPL distribution directly shows data quality landscape

## Architecture

### PPL Scorer
- **Model:** Qwen 500M (exp036, `stukenov/sozkz-core-qwen-500m-kk-base-v1`)
  - Best model we have: BPB 0.474, morphbpe-100k tokenizer
  - 447.5M params, Qwen2 architecture, GQA 7:1
- **Tokenizer:** `stukenov/sozkz-morphbpe-100k-kk-v1` (100K vocab, kazakh-optimized)
- **Input:** `saken-tukenov/sozkz-corpus-clean-v3` (13,563,018 train + 137,000 val)
- **Output:** `stukenov/sozkz-corpus-scored-kk-v1` (same texts + `ppl` column)
- **Method:** Batched cross-entropy loss → exp(loss) per text, bf16, torch.compile()
- **Crash recovery:** Intermediate parquet shards (50K rows each) saved to disk

### Dashboard (Streamlit)
- **Overview:** PPL histogram, percentile stats, box plots by source
- **Explorer:** Browse texts with PPL/source/length filters, pagination
- **Quality Tiers:** Configurable gold/silver/bronze/reject boundaries (percentile-based)
- **Export:** Filter and push subsets to HF, or download as parquet

## Files Created

| File | Purpose |
|------|---------|
| `scripts/data/ppl_score_corpus.py` | PPL scoring script (Qwen 500M) |
| `ansible/run_ppl_score_v2.yml` | Ansible playbook for remote deployment |
| `dashboard/app.py` | Streamlit dashboard entry point |
| `dashboard/pages/overview.py` | PPL distribution and stats |
| `dashboard/pages/explorer.py` | Interactive text browser |
| `dashboard/pages/quality_tiers.py` | Quality tier assignment and visualization |
| `dashboard/pages/export.py` | Filtered dataset export |
| `dashboard/requirements.txt` | Dashboard dependencies |
| `dashboard/test_data/` | Mock data for testing |

## Dataset Analysis (pre-scoring)

Sampled 500 texts from clean-v3:

| Metric | Value |
|--------|-------|
| Char length min | 64 |
| Char length median | 1,433 |
| Char length mean | 3,085 |
| Char length max | 49,930 |
| Word count median | 182 |

Length distribution:
- <100 chars: 7% (short sentences)
- 100-500: 21% (paragraphs)
- 500-2K: 33% (articles)
- 2K-10K: 33% (long articles)
- >10K: 6% (documents)

Top sources: culturax (19%), hplt_new (17%), madlad400 (15%), mc4 (14%), md_leipzig (9%), cc100 (8%), kazparc_sync (8%)

## Infrastructure

- **Compute:** CloudRift, 1x RTX 4090 (24GB), ~$0.39-0.48/hr
- **Estimated time:** ~7-8 hours (13.7M texts / ~500 texts/s)
- **Estimated cost:** ~$3-4
- **Dashboard:** Local Streamlit (localhost:8501)

## Timeline

- 2026-04-12: Design and implementation of scorer + dashboard
- 2026-04-12: CloudRift instance provisioning
- Pending: Full corpus scoring run
- Pending: Dashboard exploration of results
- Pending: Export quality-filtered subsets

## Expected Outcomes

1. PPL distribution map of the entire Kazakh corpus
2. Quality tiers (gold/silver/bronze/reject) with configurable boundaries
3. Source-level quality comparison (which sources have cleanest text?)
4. Filtered high-quality subset for future training
5. Foundation for curriculum learning experiments

## Next Steps After Scoring

1. Train new model on gold-tier subset → compare with flat training
2. Implement curriculum learning (easy→hard by PPL)
3. Domain rebalancing based on per-source PPL distributions
4. Fuzzy dedup (MinHash) on gold tier to remove near-duplicates
5. Translate more EN domains (Wikipedia, StackExchange) with quality filtering
