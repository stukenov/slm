# Parameter Golf #140: speed map vs PR #264

Snapshot: 2026-03-21.

Reference:
- your PR `#264`
- train speed: `115 ms/step`
- train steps: `5197` in `600s`
- eval time: `~696s` total

Goal of this note:
- list every submission or ablation I found with explicit speed data
- mark everything that is faster than `#264`
- keep only useful info

## Faster than PR #264 in training

### Clear wins

- `#236` `67 ms/step`, `~8900 steps / 600s`
  - Faster by: `1.72x`
  - Why: smaller batch `524K` instead of `786K`
  - Extra: batched sliding eval `172s` instead of `943s`

- `#194` `74-77 ms/step`, `7772-8052 steps / 600s`
  - Faster by: `1.49x-1.54x`
  - Why: 11L int6 QAT + per-dim SmearGate + SWA, batch `524K`

- `#215` `77.1-79.0 ms/step`, `7597-7821 steps`
  - Faster by: `1.46x-1.49x`
  - Why: low-rank Q (`512 -> 192 -> 512`) cuts step time from `108ms` to `77ms`

- `#76` `~79 ms/step` inferred from `~7600 steps / 600s`
  - Faster by: `~1.46x`
  - Why: 11L int6 + SmearGate + BigramHash + SWA + OrthoInit + WD

- `#198` `81 ms/step`, `7412 steps / 600s`
  - Faster by: `1.42x`
  - Why: 11L int6 + MLP3x + SmearGate + BigramHash + WD 0.04 + SWA + FA3

- `#179` `~73 ms/step` on normal runs, one seed at `86.15 ms/step`
  - Faster by: `~1.58x` normally
  - Why: 11L int6 + decoupled WD, simpler stack than `#198`

- `#180` `89.5 ms/step`, `6694 steps / 600s`
  - Faster by: `1.28x`
  - Why: 10L int5-MLP + SmearGate + BigramHash + SWA

- `#219` `107.35 ms/step`, `5588-5590 steps / 600s`
  - Faster by: `1.07x`
  - Why: still faster than `#264`, but not enough to justify 12L
  - Important: this is a negative result despite being faster than you

### Earlier but useful

- `#164` `68 ms/step`, `8390 steps / 600s`
  - Faster by: `1.69x`
  - Why: older 9-layer base carried into `#198`

- `#70` `48 ms/step`, `12485 steps / 600s`
  - Faster by: `2.40x`
  - Why: much smaller older base, not directly frontier-comparable

## Faster than PR #264 in eval

- `#236` batched sliding eval: `172s`
  - Your `#264`: sliding eval alone `273s`
  - Faster by: `1.59x`
  - Why: process `32` windows at once

- `#241` `Stride-OGD`
  - Claimed: `16x` faster feedback than TTT LoRA
  - Why it matters: if you want eval-time adaptation without paying full TTT cost

## Speed ideas that are not full submissions yet

- `#203` context curriculum
  - early phase: `~50 ms/step` at `seq1024`
  - late phase: `~81 ms/step` at `seq2048`
  - claim: about `60%` more optimizer steps early in the same wallclock

## Speed findings from ablations

- `#145`
  - int8 QAT overhead: `64 ms -> 77 ms`
  - cost: about `~2000` training steps lost in 600s
  - takeaway: int8 QAT was too slow for the gain

- `#215`
  - low-rank Q: `108 ms -> 77 ms`
  - gain: `28%` more steps in the same wallclock
  - takeaway: if a change saves compute without hurting model quality too much, it can beat fancy modeling ideas

- `#219`
  - 12L int5: `107 ms` vs 11L reference `81 ms`
  - takeaway: extra layer lost to slower throughput

- `#236`
  - batch sweep found `524K` tokens beats `786K`
  - takeaway: in fixed-time training, more updates mattered more than more tokens

## What this means for PR #264

Your stack is not mainly losing because of missing tricks.
It is losing because it is too slow.

The raw picture:
- `#264`: `115 ms/step`, `5197` steps
- strong frontier: `67-81 ms/step`, roughly `7400-8900` steps

That is the real gap.

## Best speed-first copy order

1. Copy `#236` batch regime.
2. Copy `#198` or `#194` train-speed envelope.
3. If needed, test `#215` low-rank Q.
4. Only then add back TTT.

## Sources

- Issue `#140`: <https://github.com/openai/parameter-golf/issues/140>
- PR `#264`: <https://github.com/openai/parameter-golf/pull/264>
- PR `#236`: <https://github.com/openai/parameter-golf/pull/236>
- PR `#194`: <https://github.com/openai/parameter-golf/pull/194>
- PR `#215`: <https://github.com/openai/parameter-golf/pull/215>
- PR `#76`: <https://github.com/openai/parameter-golf/pull/76>
- PR `#198`: <https://github.com/openai/parameter-golf/pull/198>
- PR `#179`: <https://github.com/openai/parameter-golf/pull/179>
- PR `#180`: <https://github.com/openai/parameter-golf/pull/180>
- PR `#219`: <https://github.com/openai/parameter-golf/pull/219>
- PR `#164`: <https://github.com/openai/parameter-golf/pull/164>
- PR `#70`: <https://github.com/openai/parameter-golf/pull/70>
- PR `#241`: <https://github.com/openai/parameter-golf/pull/241>
- PR `#203`: <https://github.com/openai/parameter-golf/pull/203>
- PR `#145`: <https://github.com/openai/parameter-golf/pull/145>
