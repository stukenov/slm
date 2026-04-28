# nanochat speedrun: ideas we may have missed

Snapshot: 2026-03-21.

Scope:
- source: `karpathy/nanochat`
- focus: ideas from the GPT-2 speedrun and experiment log
- target: only ideas that might transfer to `parameter-golf`

## High-confidence ideas

### 1. BOS-aligned packing is probably more important than masking document leakage

nanochat changed the dataloader so every training row starts with BOS, instead of flattening tokens and letting rows start mid-document.

Why this matters:
- they explicitly say mid-document starts can confuse the model
- they kept BOS-aligned packing as default
- then they tested varlen attention to stop cross-doc leakage and found it basically useless

Transfer guess:
- if `parameter-golf` still trains on flat token streams with arbitrary row starts, a BOS-aligned packer is worth trying
- do not spend time on fancy cross-document attention masks first

What nanochat found:
- BOS-aligned packing: useful enough to keep
- varlen boundary masking: `~0.0002` bpb, basically noise

### 2. Disable Python GC during train

nanochat disabled Python GC after step 1 and did manual collect every 5000 steps.

Why this matters:
- they call out `~500ms` pauses
- this is a free systems trick, not a modeling change

Transfer guess:
- very likely worth copying if your loop is still pure Python around `torch.compile`

### 3. Batch size should scale with training horizon, not stay fixed forever

nanochat moved from a fixed `524K` token batch to an auto-scaled rule using:
- `B_opt ∝ D^0.383`

where `D` is target training tokens.

Why this matters:
- same repo found d12 liked `524K`
- d26 liked `1M`
- they turned this into a principled scaling rule

Transfer guess:
- for current `parameter-golf` 11L runs, `524K` still looks right
- but if you try deeper or longer-horizon variants, do not assume `524K` remains optimal
- use this as the prior for the next batch sweep

### 4. Data quality beat many architecture tweaks

nanochat reports the biggest single improvement came from switching pretraining data:
- `FineWeb-EDU 100B -> ClimbMix 400B`
- speedrun time dropped from `2h46m -> 2h01m`

Transfer guess:
- only relevant if `parameter-golf` lets you change the train corpus
- if the train set is effectively fixed, ignore
- if not fixed, this is bigger than many small model hacks

### 5. Softcap tuning is cheap and real

nanochat tuned logit softcap and found:
- `20` was best in the tested range
- small but repeatable gain

Transfer guess:
- if you are still using a default or inherited softcap, this is a cheap sweep

## Medium-confidence ideas

### 6. Data order / shard order may matter more than it should

nanochat explicitly notes that different shard orders gave noticeably different results, even though the data had already been shuffled during construction.

Transfer guess:
- if your train loader iterates shards in one fixed order, try a few deterministic shard-order seeds
- this is cheap and may explain noisy single-seed wins

### 7. Backout + smear may require careful integration, not bolt-on ablations

nanochat first found smear/backout mostly negative in a simple sweep.
Later, autoresearch round 2 found a way to incorporate backout + smear so they helped and became part of the faster leaderboard path.

Transfer guess:
- a weak ablation does not prove the idea is dead
- but the useful version may require changing both training and inference behavior together

## Low-confidence / probably not worth it for parameter-golf

### 8. FP8 training

nanochat found:
- fp8 on larger models helped
- capability-matched gain was only about `~5%`
- d12-scale models did not benefit much
- `torch.compile` was mandatory
- tensorwise scaling was much better than rowwise

Transfer guess:
- current `parameter-golf` models are probably too small for this to be a top-priority lever
- maybe worth testing only after easier speed wins are exhausted

### 9. Value embeddings

nanochat liked value embeddings because they add lots of capacity at low FLOP cost.

Transfer guess:
- probably bad fit for `parameter-golf`
- cheap FLOPs, yes; cheap artifact bytes, probably no

## Negative results worth importing directly

- Varlen attention to stop cross-document leakage: not worth it
- MoE: better per-step loss, worse wallclock
- MTP: worse wallclock
- FP8 only on lm_head: not worth it
- Simple smear/backout/skip tweaks: mostly negative unless integrated carefully

## What I would actually test in parameter-golf

1. BOS-aligned document packing with best-fit crop, no padding.
2. Disable Python GC during train.
3. Small softcap sweep.
4. Shard-order / data-order seed sweep.
5. Batch-size scaling rule only if moving off the current 11L regime.

## Sources

- nanochat README: <https://github.com/karpathy/nanochat>
- speedrun script: <https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh>
- leaderboard: <https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md>
- experiment log: <https://github.com/karpathy/nanochat/blob/master/dev/LOG.md>
