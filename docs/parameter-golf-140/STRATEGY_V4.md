# Parameter Golf #140 — Strategy v4 (2026-03-25)

## Current State

### Official Leaderboard (merged)
| # | BPB | Author | PR | Key |
|---|-----|--------|----|-----|
| 1 | **1.1194** | @abaybektursun | #549 | LeakyReLU squared + Legal TTT + Parallel Muon |
| 2 | 1.1228 | @signalrush | #414 | GPTQ-lite + EMA + warmdown3500 |
| 3 | 1.1248 | @jfprincz | #287 | Partial RoPE + LN Scale + EMA |
| 4 | 1.1271 | @jfprincz | #198 | XSA4 + EMA + Int6 MLP3x |
| ... | | | | |
| **~15** | **1.1455** | **@stukenov** | **#264** | 11L+int5+TTT+SmearGate (OUR v2) |

### Pending PRs (NOT yet merged, but validated)
| BPB | Author | PR | Key | Legal? |
|-----|--------|----|-----|--------|
| **1.0745** | @RoyiRa | #688 | 5-expert Hedge Mixer + TTT | Likely legal but high std (0.021) |
| **1.0988** | @xexyz | #691 | 30-epoch Cosine TTT | **LIKELY ILLEGAL** (multi-epoch TTT) |
| 1.1164 | @Asukabot0 | #638 | XSA-all + LeakyReLU sq + VR + GA (NO TTT) | Legal |
| 1.1171 | @raahilshah | #634 | XSA-all + Full GPTQ + Pruning (NO TTT) | Legal |
| 1.1182 | @msisovic | #686 | Depth Recurrence (L4,L5 repeat) + TTT | Legal |
| 1.1186 | @EthanYangTW | #693 | CROWN-Q + Full GPTQ + SWA/EMA (NO TTT) | Legal |
| 1.1229 | @anthony-maio | #657 | VRL + lzma | Legal |

### Key Observation
Our v2 (1.1455) is **massively behind**. The competition has evolved enormously:
- Non-TTT SOTA is ~1.1164 (PR #638)
- TTT SOTA (legal) is ~1.1194 (PR #549)
- Our v3 (never deployed) was based on PR #254 which used **illegal TTT** — must be reworked

## What Changed Since March 21

### New Consensus Stack (PR #414 -> #549 -> #634)
Everything from our v2 is obsolete. The new standard is:

| Component | Old (our v2) | New Standard |
|-----------|-------------|--------------|
| Activation | relu sq | **LeakyReLU(0.5) squared** (-0.003 BPB) |
| XSA | last 4 layers | **all 11 layers** (-0.006 BPB) |
| RoPE | full | **Partial RoPE 16/64** (25% of head dims) |
| LN | standard | **LN Scale 1/sqrt(i+1)** per layer |
| Residuals | skip connections | **U-Net skip connections** |
| Quant | int5/int6 naive | **GPTQ-lite** (per-row clip search) or **Full GPTQ** |
| Compression | zstd-22 | **lzma** (2-5% better ratio) |
| Optimizer | Muon | **Parallel Muon** + Parameter Banking (84ms->83ms/step) |
| EMA | none | **EMA(0.997)** every step |
| Embeddings | BigramHash(64) | **BigramHash(2048-4096)**, **VE128** (Value Embeddings) |
| QAT | late | **Late QAT @ lr_scale < 0.15** |
| Pruning | none | **3% magnitude pruning** post-GPTQ |
| Warmdown | 3000 | **3500** |

### New Techniques Worth Adding
1. **CROWN-Q** (PR #693): Curvature-weighted quant penalty during warmdown. Pushes weights into flat minima where int6 causes less damage. Zero cost at inference time.
2. **Depth Recurrence** (PR #686): Re-execute layers 4,5 with learned scalars -> 13 virtual layers from 11 physical. +~2K params, -0.001 BPB.
3. **5-expert Hedge Mixer** (PR #688): Online context mixing of neural + unigram + bigram + trigram + entropy experts during TTT. Huge potential (-0.045 BPB over baseline TTT).
4. **Value Residual Learning** (PR #657): Layer 0's V output blended into subsequent attention via sigmoid gates. +10 params. -0.002 BPB.
5. **Selective Pruning** (PR #634): Post-GPTQ, sort +/-1 values by scale squared, zero least-impactful until target size.

### Rule Changes (Issue #677)
**CRITICAL**: Many submissions were invalidated. Our v3 script likely has illegal TTT.

Legal TTT = **Score-First only** (PR #549 reference):
```
for each 32K-token chunk:
    Phase 1 - SCORE: sliding window under torch.inference_mode
    Phase 2 - TRAIN: SGD on already-scored tokens, 1-3 epochs
```

Illegal:
- Multi-epoch TTT where final-epoch score is reported
- GPTQ calibration on training data during the separate 600s window
- Oracle/hindsight selection (min NLL across passes)
- Re-scoring after adaptation

### Negative Results (Don't Waste Compute)
- **Orthogonal residuals**: -0.05 BPB WORSE at this scale (tested by PROTEUS)
- **Width scaling** (512->576 dim): Regression (tested by PR #686)
- **ARO optimizer**: Untested, likely marginal gain at 16MB/600s
- **AdamW TTT on GPTQ models**: Destroys quantized weights (+0.077 BPB, PR #693)

## Strategy v4: Three-Phase Plan

### Phase 1: Non-TTT Base (target: 1.115)
Build the strongest possible base before TTT. Copy PR #634 stack:

```
11L, 512d, 8H/4KV (GQA)
LeakyReLU(0.5) squared MLP 3x
XSA on ALL 11 layers
Partial RoPE 16/64
LN Scale 1/sqrt(i+1)
U-Net skip connections
SmearGate + BigramHash(2048) + VE128
EMA(0.997) + Tight SWA
Late QAT @ 0.15
GPTQ-lite int6 + lzma
Parallel Muon + Parameter Banking
Warmdown 3500
Seq len 2048
OrthoInit
```

Additions to test:
- **CROWN-Q** penalty during warmdown (PR #693)
- **Depth Recurrence** on L4,L5 (PR #686, +13 virtual layers)
- **Full GPTQ** instead of GPTQ-lite (PR #634, but must fit in 600s training!)
- **VRL** (PR #657)
- **Selective +/-1 pruning** (PR #634)

### Phase 2: Legal Score-First TTT (target: 1.105)
Based on PR #549 framework (the ONLY merged legal TTT):

```python
# Legal TTT loop:
for chunk in val_chunks(32768_tokens):
    with torch.inference_mode():
        score = sliding_window_scoring(model, chunk, stride=64)  # SCORE FIRST
    loss = train_sgd(model, chunk, epochs=3, lr=0.002, momentum=0.9)  # THEN TRAIN
```

Key decisions:
- **Unfreeze all blocks** (PR #549 ablation: freeze=0 > freeze=2)
- **SGD** not AdamW (AdamW destroys GPTQ weights per PR #693)
- **Per-layer LR**: mlp.proj 3x, mlp.fc 0.5x (PR #691)
- **Cosine LR decay** during TTT
- Budget: ~410s for TTT within 600s total limit

### Phase 3: Hedge Mixer (target: <1.08)
The **biggest potential lever**. PR #688's 5-expert Hedge Mixer:

| Expert | Source |
|--------|--------|
| Neural | Base model log-softmax |
| Unigram | Token frequency from scored tokens |
| Bigram | P(next given prev) from scored tokens |
| Trigram | Hashed trigram table (64K buckets) |
| Entropy | Neural entropy as confidence regularizer |

Online weight update via Hedge algorithm: `log_w -= eta * loss`.
All n-gram tables built from already-scored tokens only -> legal.

This alone gave -0.045 BPB gain in PR #688 (1.1252 -> 1.0745).

**Risk**: High variance (std 0.021 in PR #688). Need to stabilize.

## Implementation Priority

1. **[HIGHEST]** Fork PR #549's train_gpt.py (merged, legal, proven)
2. Add XSA-all, CROWN-Q, VRL, lzma from PRs #634/#693/#657
3. Add Depth Recurrence from PR #686
4. Verify legal TTT is correct (score-first pattern)
5. Add Hedge Mixer (PR #688)
6. Run 3-seed validation on 8xH100
7. Submit as new PR

## Cost Estimate
- 1x H100 for ablations: ~$2/hr x 3hrs = $6
- 8x H100 for 3-seed final: ~$20/hr x 1hr = $20
- Buffer: $10
- **Total: ~$36**

## Files to Get
```bash
# Fork #549 as base (merged SOTA)
gh pr checkout 549 --repo openai/parameter-golf

# Study these for techniques to add:
gh pr view 634 --repo openai/parameter-golf  # XSA-all + Full GPTQ
gh pr view 686 --repo openai/parameter-golf  # Depth Recurrence
gh pr view 688 --repo openai/parameter-golf  # Hedge Mixer
gh pr view 693 --repo openai/parameter-golf  # CROWN-Q
```

## Competition Timeline
- **Deadline: April 30, 2026**
- We have ~5 weeks left
- Current merged SOTA: 1.1194 (PR #549)
- Pending claimed SOTA: ~1.0745 (PR #688, not yet merged)
- Our target: **< 1.08** (top 3)
