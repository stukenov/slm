# Parameter Golf #140 — Progress Tracker

## Current Status: v4 ALL FEATURES compiled & running, ready for 8xH100

## v4 Smoke Test Results (2026-03-25, 1xA6000 48GB)
- **Code runs end-to-end**: training + EMA + SWA + quant + sliding eval
- XSA on ALL 11 layers: confirmed working
- SDPA fallback (no FA3): 625ms/step on A6000 (vs 83ms on H100)
- 500 steps, train_loss: 3.00, val_bpb: 1.78 (expected for 500 steps)
- Artifact: 5.1MB (under budget, room for more params)
- EMA/quant degrade on short runs (expected, OK on 7000+ steps)
- Cost: ~$0.10 (6 min on A6000 @ $0.33/hr)

## v4 Full Feature Test (2026-03-25, 1xA6000 48GB)
- **ALL features running**: XSA-11, VRL, GatedAttn, CROWN-Q, DepthRecur(4,5), HedgeMixer
- `model_params:27051758` (13 virtual layers from 11 physical)
- `recurrence:layers=[4, 5] physical=11 virtual=13` confirmed
- 843ms/step on A6000 (CROWN-Q adds ~35% overhead vs base)
- On 8xH100 with FA3: expect ~90-95ms/step -> ~6300 steps in 600s
- Artifact: 5.2MB (huge headroom under 16MB)
- train_loss: 2.95, val_bpb: 1.79 at 500 steps (same ballpark as v4 smoke)
- EMA/quant degrades on short runs (expected, NOT a bug — needs 3000+ steps)
- Cost: ~$0.15 (10 min on A6000 @ $0.33/hr)
- **Total test spend: ~$0.25**

## PR #264 (v2, live): https://github.com/openai/parameter-golf/pull/264
- **val_bpb: 1.1455** (seed 1337), ~2-3 place
- 11L + int5-MLP + TTT-SGD + SmearGate + SWA

## v3 (ready to deploy, NOT yet run on 8xH100)
**File:** `train_gpt_v3.py` — based on PR #254 (SOTA, 1.1303)

### v3 changes vs v2:
| Change | Why | Source |
|--------|-----|--------|
| **Int5→int6-all** | Int5 quant penalty 0.029 vs int6's 0.010 | #236, #238 |
| **Batch 786K→524K** | More steps: 67ms vs 115ms = +70% steps | #236 |
| **FA3 (optional)** | Faster attention on H100, fallback to SDPA | #254 |
| **LR 0.04→0.025** | More stable with more steps | #236 |
| **Momentum 0.95→0.99** | Better convergence at lower LR | #236 |
| **Warmup 500→1500 steps** | Match momentum warmup to longer training | #236 |
| **WD 0.02→0.04** | Better quant compression | #236 |
| **Warmdown 1200→3000** | More warmdown for better SWA | #236 |
| **NTK-RoPE** | Auto-scale RoPE for longer eval seqs | #254 |
| **TTT freeze first 2 blocks** | Stability during TTT | #254 |
| **TTT 3 epochs** | More adaptation | #254 |
| **Bigram 4096×128** | More expressive bigram features | #254 |

### Expected v3 on 8xH100:
- ~8900 steps in 600s (vs v2's 5197)
- Pre-TTT: ~1.13-1.14 (vs v2's 1.1507)
- Post-TTT: ~1.10-1.12 (vs v2's 1.1455)
- Could match or beat #254's 1.1303

### v3 test results (1xA6000, 500 steps):
- Code works end-to-end (training + SWA + quant + TTT + eval)
- FA3 gracefully falls back to SDPA on non-H100
- 327ms/step on A6000

## 8xH100 Deploy Issues (2026-03-21 night)
- Attempt 1: Pod died during setup (SSH disconnect during data download)
- Attempt 2: Pod failed to provision (250GB disk too large)
- Attempt 3: Pod booted but crashed after ~230s
- **Root cause**: RunPod 8xH100 machines unstable tonight
- **Action**: Retry during daytime when machines are more stable

## To deploy v3 on 8xH100:
```bash
# 1. Launch pod
python scripts/runpod_launch.py launch --gpu 8xH100

# 2. SSH in, then run setup in screen (prevents SSH disconnect):
screen -S setup
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install sentencepiece datasets huggingface-hub zstandard tiktoken
python data/cached_challenge_fineweb.py
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn --no-build-isolation  # FA3, optional

# 3. Upload v3 script
# From local machine:
scp -P PORT train_gpt_v3.py root@HOST:/workspace/parameter-golf/train_gpt.py

# 4. Run (in screen!):
screen -S train
cd /workspace/parameter-golf
MAX_WALLCLOCK_SECONDS=600 TTT_EPOCHS=3 SEED=1337 \
torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/run_v3.log
```

## All Experiments Log

| # | Date | Config | GPU | Steps | BPB post-quant | BPB post-TTT | Artifact | Status |
|---|------|--------|-----|-------|----------------|--------------|----------|--------|
| 1 | 03-20 | v2 11L int5 wd0.04 | 1x4090 | 500 | - | - | 18.90 MB | TOO BIG |
| 2 | 03-20 | v2 9L int6 wd0.04 | 1x4090 | 500 | 1.5122 | - | 16.03 MB | over 34KB |
| 3 | 03-20 | v2 9L bg2048d96 | 1x4090 | 500 | 1.5138 | - | 15.61 MB | v1 OK |
| 4 | 03-20 | v2 11L int5 bg2048d64 | 1xA40 | 500 | 1.5074 | 1.5030 | 16.01 MB | v2 test OK |
| 5 | 03-20 | v2 11L int5 no-fp16 | 8xH100 | 5197 | 1.1507 | **1.1455** | 15.94 MB | **PR #264** |
| 6 | 03-21 | v3 11L int6 batch524K | 1xA6000 | 500 | 1.5963 | 1.5963 | 11.35 MB | v3 test OK |
| 7 | 03-21 | v3 on 8xH100 | 8xH100 | - | - | - | - | RunPod crashed ×3 |

## Cost Tracking
| Date | GPU | Duration | Est Cost | Purpose |
|------|-----|----------|----------|---------|
| 03-20 | 1xRTX4090 | ~50 min | ~$0.35 | v1/v2 ablations |
| 03-20 | 1xA40 | ~40 min | ~$0.30 | v2 11L test |
| 03-20 | 8xH100 ×2 | ~50 min | ~$20 | v2 official runs |
| 03-21 | 1xA6000 | ~30 min | ~$0.25 | v3 test |
| 03-21 | 8xH100 ×3 | ~15 min (crashed) | ~$6 | v3 failed deploys |
| **Total** | | | **~$27** | |

## Competition Landscape (as of 03-21)
| Rank | BPB | Author | Key | PR |
|------|------|--------|-----|-----|
| 1 | **1.1303** | @timowhite88 | TTT+11L+FA3+int6+batch524K | #254 |
| 2 | 1.1326 | @jfprincz | 11L+SmearGate+WD0.04+SWA+FA3 | #198 |
| 3 | 1.1400 | @saml212 | 11L+SmearGate+batch524K optimization | #236 |
| 4 | 1.1453 | @thwu1 | Mixed int5/int6, 10L | #180 |
| **5** | **1.1455** | **@stukenov** | **11L+int5+TTT+SmearGate** | **#264** |
| 6 | 1.1472 | @devin-cog | 11L+int6+WD0.038 | #179 |

## Files
```
parameter-golf-140/
  PROGRESS.md              — this file
  train_gpt_v3.py          — v3: PR#254 base + #236 hyperparams (READY)
  train_gpt.py             — v2: current PR #264 (deployed)
  scripts/
    runpod_launch.py       — RunPod pod manager
    setup_pod.sh           — pod setup
    run_train.sh           — training runner
    deploy_to_pod.sh       — deploy files
  configs/
    v1_9L_final.env        — v1 (archived)
```
