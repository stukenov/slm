# Parameter Golf #140: short submission map

Snapshot: 2026-03-20.

Goal: quickly see what the strong submissions used, and what to copy first.

## What almost everyone used

Core stack:
- sliding-window eval
- int6 post-train quantization
- MLP 3x expansion
- fp16 tied embeddings
- zstd-22 compression

Near-consensus training knobs:
- Muon-family optimizer
- Muon momentum warmup `0.92 -> 0.99`
- matrix/scalar LR around `0.02`
- warmdown `3000`
- grad clip `0.3`

Fast-rising add-ons:
- SmearGate
- BigramHash
- OrthoInit / muP-scaled init
- seq_len `2048`
- SWA

## Ranked submissions

### Official merged top 5

1. `#60` `@notapplica` `1.1748`
   - Used: sliding window `stride=64`, fp16 embeddings, 10 layers, decoupled Muon WD, overtone/spectral init, residual mixing.
   - Repeat: start from baseline, add sliding eval, keep tied emb in fp16, go `9L -> 10L`, add Muon WD `0.02`.

2. `#50` `@mattqlf` `1.1925`
   - Used: sliding window eval only.
   - Repeat: keep training unchanged, change eval to sliding window `stride=64`.

3. `#77` `@samacqua` `1.1928`
   - Used: doc-isolated eval + sliding window + LoRA test-time training.
   - Repeat: split validation by document, use overlapping windows, train rank-8 LoRA per document chunk, reset LoRA between docs.

4. `#52` `@spokane-way` `1.2014`
   - Used: long context (`4096`) + tuned Muon momentum/LR/warmdown.
   - Repeat: keep baseline architecture family, train at `seq_len=4096`, sweep Muon schedule.

5. `#49` `@spokane-way` `1.2060`
   - Used: `seq_len=2048`.
   - Repeat: baseline + `seq_len=2048` and tuned hparams.

### Pending validated top 5

1. `#179` `@devin-cog` `1.1472`
   - Used: 11 layers, GQA (`kv_heads=4`), int6 + zstd, fp16 embeddings, sliding eval `stride=64`, decoupled Muon WD `0.038`, LR `0.025`.
   - Repeat: take consensus stack, fund `11L` with int6+zstd, add stronger WD.

2. `#162` `@raahilshah` `1.1483`
   - Used: int6, MLP 3x, SmearGate, BigramHash, OrthoInit, Muon WD, SWA, seq2048.
   - Repeat: this is the cleanest modern baseline to copy.

3. `#164` `@jfprincz` `1.1538`
   - Used: same family as `#162` plus seq2048 + FA3, stride `256`.
   - Repeat: copy `#162`, then raise train seq to `2048` and switch attention to FA3.

4. `#173` `@tamoghnokandar` `1.1546`
   - Used: int6, MLP 3x, selective precision, seq2048, NorMuon, FlashAttention 3.
   - Repeat: copy `#114`, replace Muon with NorMuon, use FA3.

5. `#114` `@saml212` `1.1575`
   - Used: int6, MLP 3x, selective precision, long-context training.
   - Repeat: important claim was `train@2048` is enough; no need to pay for `4096`.

### Useful but less standard

- `#180` `@thwu1` `1.1453`
  - Mixed `int5` MLP + `int6` attention, 10L, WD `0.04`, SWA/50, SmearGate, BigramHash.
  - Best “compression buys depth” recipe.

- `#135` `@unnir` `1.1539`
  - OrthoInit + int6 + MLP 3x + SmearGate + BigramHash.
  - Good minimal version of the modern stack.

- `#174` `@Julz19` `1.1537`
  - ContextFuse-2048 + BigramSmear + mixed int6 + SWA + Muon WD.
  - Alternative to SmearGate/BigramHash family.

- `#144` `@DJLougen` `1.0156`
  - MPK architecture (`8x384`, `k=2`, `m=4`), int8+zlib.
  - Huge score, but architectural outlier; not the easiest thing to copy.

- `#168` `@spokane-way` `1.0238`
  - Paid prefix (`8.75MB`) + 7L 384d.
  - Not the first thing to reproduce unless you want the prefix trick specifically.

## What to copy first

If you want the shortest path:
1. Copy the consensus stack.
2. Make `#162` your base.
3. Then try one branch at a time:
   - `#179`: add depth + stronger WD
   - `#173`: swap in NorMuon + FA3
   - `#180`: try mixed int5/int6 to buy another layer

## Your PR

`#118` is a non-record compat smoke:
- 1x RTX 4090
- 50 steps
- reproducibility/compat fix for SDPA+GQA
- not leaderboard-relevant

## Source links

- Issue `#140`: <https://github.com/openai/parameter-golf/issues/140>
- PR `#118`: <https://github.com/openai/parameter-golf/pull/118>
- PR `#60`: <https://github.com/openai/parameter-golf/pull/60>
- PR `#50`: <https://github.com/openai/parameter-golf/pull/50>
- PR `#77`: <https://github.com/openai/parameter-golf/pull/77>
- PR `#52`: <https://github.com/openai/parameter-golf/pull/52>
- PR `#49`: <https://github.com/openai/parameter-golf/pull/49>
- PR `#179`: <https://github.com/openai/parameter-golf/pull/179>
- PR `#162`: <https://github.com/openai/parameter-golf/pull/162>
- PR `#164`: <https://github.com/openai/parameter-golf/pull/164>
- PR `#173`: <https://github.com/openai/parameter-golf/pull/173>
- PR `#114`: <https://github.com/openai/parameter-golf/pull/114>
- PR `#180`: <https://github.com/openai/parameter-golf/pull/180>
- PR `#135`: <https://github.com/openai/parameter-golf/pull/135>
- PR `#174`: <https://github.com/openai/parameter-golf/pull/174>
- PR `#144`: <https://github.com/openai/parameter-golf/pull/144>
- PR `#168`: <https://github.com/openai/parameter-golf/pull/168>
