# Parameter Golf — $1000 Compute Grant Application

## Detailed Approach (2500 chars max)

My submission (PR #745) combines six architectural and evaluation innovations on top of the PR #549 merged SOTA stack, achieving a 3-seed mean val_bpb of 1.0222 (std 0.0067) with 600s training and 507s eval on 8xH100 SXM.

**Architecture (training-time):**
1. XSA on all layers — Exclusive Self-Attention applied to every layer instead of last 4, forcing cross-position mixing from layer 0 (-0.006 BPB).
2. Depth Recurrence — Layers 4 and 5 are re-executed with independent scalar parameters, creating 13 virtual layers from 11 physical. Banks are indexed via a virtual-to-physical mapping. Near-zero parameter overhead (~2K scalars), but captures richer representations through repeated computation.
3. Value Residual Learning (arXiv:2410.17897) — Layer 0's V output is blended into all subsequent attention layers via learned sigmoid gates, combating attention concentration.
4. Gated Attention — Per-head sigmoid gates on attention output with learned bias initialization.
5. CROWN-Q — Curvature-weighted quantization variance penalty applied during warmdown: lambda * mean(w^2) * (row_max/15)^2 / 12. This pushes weights into flat minima where int6 quantization causes less damage, at zero eval-time cost.

**Evaluation (eval-time):**
6. 5-Expert Hedge Mixer — GPU-vectorized online context mixing during legal score-first TTT. Five experts (neural model, unigram, bigram, hashed trigram, neural entropy) blend predictions in log-probability space. Expert weights are updated online via the Hedge/multiplicative-weights algorithm. All n-gram tables are built incrementally from already-scored tokens only, maintaining full compliance with score-first TTT rules.

The TTT loop processes 32K-token chunks: score under torch.inference_mode(), update mixer statistics, then train via SGD (1 epoch, all blocks unfrozen, cosine LR decay). Depth recurrence layers are untied before TTT so each virtual layer gets independent weight updates.

The entire stack is designed for reproducibility: `torchrun --nproc_per_node=8 train_gpt.py` with no environment variables reproduces the submitted results exactly.

## Current Best Leaderboard Result

1.0222 (PR #745, pending review)

## Expected Improvement from Additional Compute (1500 chars max)

Three concrete directions would benefit from additional compute:

1. **TTT epoch scaling** — Our current submission uses 1 TTT epoch (507s eval) to stay within the 600s limit. With optimized TTT implementation (larger chunks, fused kernels, reduced DDP overhead), I can fit 2-3 epochs in 600s. Our internal run with 3 epochs achieved 1.0278 mean BPB in 763s — the gap is purely engineering, not algorithmic. Estimated gain: -0.005 to -0.01 BPB.

2. **Hedge Mixer expert expansion** — The current 5-expert mixer leaves room for higher-order n-gram experts (4-gram, 5-gram with larger hash tables) and adaptive per-document expert reweighting. Each variant requires a full 3-seed validation run (~$20). I plan to sweep mixer configurations (eta, hash size, expert count) systematically. Estimated gain: -0.005 to -0.015 BPB.

3. **Architecture search on depth recurrence** — Currently recurring layers 4,5 was chosen from PR #686. Sweeping which layers to recur (e.g., 3,4,5 or 5,6,7) and how many repetitions, combined with the full training + TTT + mixer pipeline, requires ~$20 per configuration. A systematic 10-config sweep could find the optimal recurrence pattern. Estimated gain: -0.002 to -0.005 BPB.

Combined, these directions could push below 1.00 BPB. The $1000 grant would fund approximately 50 full 3-seed runs, enabling rigorous ablation studies rather than guesswork.
