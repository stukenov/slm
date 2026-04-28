# Parameter Golf — Strategy v5 (2026-03-27)

## Executive Summary

The competition has bifurcated into two tracks: **N-gram eval-time caching** (0.03-0.10 BPB) and **pure neural** (1.02 BPB). N-gram PRs are not yet merged but organizers are "leaning towards accepting." Our v4 (1.0222 BPB, PR #745) is competitive on the neural track but **adding n-gram backoff could drop us to ~0.40-0.96 BPB**.

**v5 strategy: keep our strong neural base + add n-gram backoff mixer = best of both worlds.**

## Current Standings

| Rank | BPB | Author | Technique | PR |
|------|-----|--------|-----------|-----|
| 1 (pending) | 0.0308 | THUQiXuan | Order-13 N-gram Oracle | #883 |
| 2 (pending) | 0.0935 | aamodbhatt | Fast Full-Rescore N-gram | #888 |
| ... | ... | ... | ... | ... |
| ~10 (pending) | 0.9642 | unknown | Neural + N-gram Backoff | #889 |
| ~15 (pending) | 1.0222 | **stukenov** | **Our v4** | **#745** |
| ~16 (pending) | 1.0226 | shalyhinpavel | Pure Neural GDN | #875 |
| 1 (merged) | 1.1194 | abaybektursun | LeakyReLU² + TTT | #549 |

## v5 Changes Over v4

### Tier 1: N-gram Backoff Cache (est. -0.3 to -0.6 BPB)

The **single biggest lever**. Based on PR #889's approach:
- Multi-order backoff cache (2-7gram), 4M hash buckets per order
- Entropy-adaptive alpha: `alpha = 0.05 + 0.55 * sigmoid(2*(H-4))`
  - Neural confident (low entropy) → alpha≈0.05 (trust neural)
  - Neural uncertain (high entropy) → alpha≈0.60 (trust n-gram)
- Built causally from already-scored tokens only (legal)
- All-reduce sync across GPUs (1.6s overhead)
- min_count=2 gate, raw count ratios, no smoothing
- Zero artifact bytes — purely eval-time state

**Integration with our existing Hedge Mixer:** Replace our 5-expert Hedge with a hybrid approach:
- Keep neural + entropy experts from Hedge
- Replace trigram expert with full multi-order n-gram backoff
- Entropy-adaptive blending instead of fixed Hedge weights

### Tier 2: Architecture Improvements (est. -0.005 to -0.015 BPB)

| Change | Source | Expected Gain |
|--------|--------|---------------|
| **EMA(0.997)** every step | PR #414, #549 | -0.003 BPB |
| **6-Expert Hedge Mixer** (+4gram) | PR #849 | -0.002 BPB |
| **BI-guided depth recurrence** (15L from 11) | PR #857 | -0.003 BPB |
| **LeakyReLU(0.9)²** (test vs 0.5) | PR #849 | -0.001 BPB |
| **BigramHash(4096-8192)** | PR #849 | -0.001 BPB |
| **3% selective pruning** post-GPTQ | PR #634 | -0.001 BPB |

### Tier 3: Training Improvements (est. -0.003 to -0.008 BPB)

| Change | Source | Expected Gain |
|--------|--------|---------------|
| **Full-Training QAT** (threshold=1.0) | PR #836 | -0.003 BPB |
| **AdamW TTT** with Polyak averaging | PR #849 | -0.002 BPB |
| **Cosine TTT** (20 epochs, per-layer LR) | PR #857 | -0.003 BPB |

## Implementation Plan

### Phase 1: Neural Base Upgrades
1. Add EMA(0.997) to training loop
2. Upgrade depth recurrence: BI-guided, layers 9-13 tied → 15 virtual layers
3. Add 4-gram expert to Hedge Mixer (6 experts total)
4. Test LeakyReLU(0.9)² vs (0.5)²
5. Increase BigramHash to 4096
6. Add 3% selective pruning post-GPTQ
7. Switch to Full-Training QAT

### Phase 2: N-gram Backoff Cache
1. Implement multi-order n-gram backoff (2-7gram)
2. Hash table: 4M buckets per order, all-reduce sync
3. Entropy-adaptive alpha blending
4. Integrate with existing TTT loop (score → update n-gram cache → blend → train)
5. Causal correctness: only use already-scored tokens

### Phase 3: TTT Enhancements
1. Switch SGD → AdamW for TTT with cosine schedule
2. Per-layer learning rates (mlp.proj 3x, mlp.fc 0.5x)
3. Polyak weight averaging during TTT
4. Increase epochs (4-20, time permitting)

## Expected v5 Results

| Scenario | Expected BPB | Confidence |
|----------|-------------|------------|
| Neural only (no n-gram) | ~1.01 | High |
| Neural + TTT (no n-gram) | ~0.99 | Medium |
| Neural + TTT + N-gram backoff | ~0.40-0.60 | Medium (depends on n-gram legality) |

## Risk Assessment

1. **N-gram legality**: Not yet settled. If banned, we fall back to ~0.99 BPB which is still competitive
2. **Time budget**: N-gram eval adds ~200-300s. Must fit training + TTT + n-gram eval in 600s
3. **Hash collision sensitivity**: PR #886 showed 1M buckets > 256M buckets. Needs careful tuning
4. **N-gram + TTT interaction**: Training on tokens with n-gram-boosted scores may cause issues

## Cost Estimate

- 1×A6000 for ablations: ~$1-2
- 8×H100 for 3-seed validation: ~$20
- Buffer for n-gram tuning: ~$15
- **Total: ~$37**

## Files

```
parameter-golf-140/
  train_gpt_v5.py          — v5 with all features
  STRATEGY_V5.md           — this file
  PROGRESS.md              — updated tracker
```

## Timeline

- Deadline: **April 30, 2026** (~34 days left)
- Week 1: Build v5, smoke test on A6000
- Week 2: 8×H100 ablations (neural base, then +TTT, then +n-gram)
- Week 3: 3-seed validation, submit PR
- Week 4: Buffer for iteration based on review
