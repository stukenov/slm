# Parameter Golf #140 — Detailed Analysis & What to Improve Next

## Our Submission: PR #264
- **BPB: 1.1455** (seed 1337, single seed)
- Rank: ~2-3 place among pending submissions
- PR: https://github.com/openai/parameter-golf/pull/264

## Technique Breakdown: What Each Piece Contributed

### Model Architecture (11L, 512d, 26.67M params)
| Component | Params | BPB impact | Notes |
|-----------|--------|------------|-------|
| 11 transformer blocks | ~25.9M | ~0.008 vs 9L | 2 extra layers = more capacity |
| MLP 3x (1536 hidden) | ~16.9M of above | ~0.01-0.02 vs 2x | Biggest single gain from int6 era |
| SmearGate | ~512 | ~0.01 vs no-SmearGate | Cheap bigram injection |
| BigramHash (2048×64→512) | ~164K | part of SmearGate gain | Token-pair context |
| U-Net skip connections | ~2.8K | unknown | Part of baseline |
| GQA (8h/4kv) | saves ~1M | neutral BPB | Smaller K/V = more room |

### Quantization Stack
| Technique | Artifact size impact | BPB impact (quant gap) |
|-----------|---------------------|----------------------|
| Int5 for MLP [-16,15] | **saves ~1.9MB** vs int6 | ~0.002 more quant gap |
| Int6 for attention [-32,31] | standard | standard |
| tok_emb in int8 (not fp16) | **saves ~0.5MB** | ~0.001 more quant gap |
| zstd-22 compression | **~1.5x** ratio on int6, ~1.88x on int5 | none |
| Total artifact | **15.94 MB** | quant gap ~0.008 |

### Training Techniques
| Technique | BPB impact | Cost |
|-----------|------------|------|
| Muon WD=0.04 | ~0.005 vs WD=0 | free |
| SWA (30 checkpoints) | ~0.003-0.005 | free |
| OrthoInit + muP | ~0.002-0.005 | free |
| Muon momentum warmup 0.92→0.99 | ~0.002 | free |
| Grad clip 0.3 | stabilizes training | free |
| Warmdown 3000 iters | better convergence | free |
| seq_len=2048 (vs 1024) | ~0.01-0.02 | 2x memory |

### Eval Techniques
| Technique | BPB impact | Time cost |
|-----------|------------|-----------|
| Sliding window stride=64 | **~0.03** vs standard eval | +250s |
| Full-model SGD TTT (2 epochs) | **~0.005** (our run) | +422s |
| Total eval time | | ~696s (within 10 min) |

## Where We Lose to #1 (PR #198, 1.1326 BPB)

Gap: **0.013 BPB**

| Factor | Our value | PR #198 value | Est. BPB difference |
|--------|-----------|---------------|-------------------|
| **FlashAttention 3** | No (SDPA) | Yes | ~0.005 (more steps: ~81ms vs 115ms/step) |
| **Bigger bigram** | 2048×64 | 2048×128(?) | ~0.001-0.003 |
| **QAT (STE int6)** | No | Configurable | ~0.003-0.005 (closes quant gap) |
| **TTT** | Yes (0.005 gain) | No | We're ahead here |
| **More training steps** | 5197 | ~7412 (81ms/step) | ~0.005 |

### Total estimated gap explained: ~0.013-0.018 (matches observed 0.013)

## What to Improve Next (priority order)

### 1. FlashAttention 3 (est. +0.005 BPB)
- **Why**: FA3 on H100 SXM gives ~81ms/step vs our 115ms with SDPA. That's 7400 steps vs 5200 in 600s — 42% more training.
- **How**: `pip install flash-attn`, import `flash_attn_interface`, replace SDPA call.
- **Risk**: Low — well-tested by #198, #164, #173.
- **Priority**: HIGH — biggest single improvement available.

### 2. QAT with STE (est. +0.003-0.005 BPB)
- **Why**: Our quant gap is 0.008 BPB (1.1583 pre-quant → 1.1507 post-quant before sliding). QAT can close this to near-zero.
- **How**: Activate int5/int6 simulation in forward pass during last 15-25% of training. STE passes gradients through rounding.
- **Risk**: Medium — STE + Muon momentum can conflict (PR#141 found +0.007 degradation). Need late activation (75%+).
- **Priority**: HIGH — direct BPB gain.

### 3. Bigger bigram (est. +0.001-0.003 BPB)
- **Why**: We use dim=64, PR#162 uses 128, PR#180 uses 128. Smaller bigram = less expressive.
- **How**: If FA3 gives faster steps → smaller artifact possible → room for bigger bigram.
- **Risk**: Low.
- **Priority**: MEDIUM — depends on artifact headroom.

### 4. NorMuon optimizer (est. +0.001-0.002 BPB)
- **Why**: Per-neuron adaptive LR on top of Muon. Used by several top submissions.
- **How**: Add row-wise normalization after Newton-Schulz.
- **Risk**: Low — well-tested.
- **Priority**: LOW — small gain.

### 5. More aggressive TTT (est. +0.005-0.01 BPB)
- **Why**: Our TTT gave only 0.005 BPB (vs 0.033 from PR#152). Possible we're under-tuning.
- **How**: Try lr=0.003, 3 epochs, or LoRA TTT for per-document adaptation.
- **Risk**: Medium — catastrophic forgetting if too aggressive.
- **Priority**: MEDIUM — needs sweep.

### 6. 3-seed validation (required for record)
- **How**: Run seeds 1337, 42, 2025 on 8xH100. Cost: ~$12 total.
- **Priority**: REQUIRED for leaderboard.

## Theoretical ceiling with all improvements

| Current | +FA3 | +QAT | +bigger bigram | +TTT tuning | Estimated |
|---------|------|------|----------------|-------------|-----------|
| 1.1455 | 1.140 | 1.137 | 1.135 | 1.130 | **~1.13** |

This would match or beat the current #1 (1.1326).

## Cost Summary

| Run | GPU | Time | Cost | Result |
|-----|-----|------|------|--------|
| v1 experiments (5 runs) | 1xRTX4090 | 50 min | ~$0.35 | Size/code validation |
| v2 experiments (4 runs) | 1xA40 | 40 min | ~$0.30 | 11L+int5+TTT validation |
| 8xH100 run 1 (artifact too big) | 8xH100 | 25 min | ~$10 | 1.1457 BPB, 16.48MB |
| 8xH100 run 2 (fixed) | 8xH100 | 25 min | ~$10 | **1.1455 BPB, 15.99MB** |
| **Total** | | | **~$21** | |
