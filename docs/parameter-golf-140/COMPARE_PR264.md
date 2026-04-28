# PR #264 vs newer stronger PRs

Snapshot: 2026-03-21.

Your PR:
- `#264` `1.1455`
- stack: `11L + int5-MLP/int6-attn + TTT-SGD + SmearGate + BigramHash + SWA + OrthoInit + WD 0.04`
- train: `5197 steps`, `115 ms/step`
- pre-TTT sliding score: `1.1507`

## Short answer

The newer PRs are better mostly because their base model is much stronger before eval tricks.

Your TTT works.
Your base is the problem.

## Why they beat you

### 1. Your train-time throughput is too low

Compared to stronger PRs:
- `#264`: `5197` steps at `115 ms/step`
- `#198`: `7412` steps at `81 ms/step`
- `#194`: `7772-8052` steps at `74-77 ms/step`
- `#236`: `~8900` steps at `67 ms/step`

In this contest, more updates in the same 600s mattered a lot.

### 2. Int5 MLP is probably hurting more than helping at 11L

The strongest direct warning is `#236`:
- int6-all quant penalty: `0.010`
- int5-MLP quant penalty: `0.029`

Their claim is simple: at 11 layers, the extra space from int5 is not worth the quality loss.

Your result is consistent with that:
- your post-quant + sliding score is `1.1507`
- despite TTT, you only get to `1.1455`

So the base after quantization is already too weak.

### 3. TTT is stacking on a weaker base than the best TTT PR

Best recent TTT PR:
- `#254`: mean `1.1313`
- issue summary says pre-TTT sliding was `1.1447`
- TTT gain was about `0.014`

Your TTT gain:
- `1.1507 -> 1.1455`
- gain `0.0052`

So the gap is not “they used TTT and you didn’t”.
The gap is “their base before TTT is already much better”.

### 4. The current frontier moved to the 11L int6 stack

Reference PR now:
- `#198`: `11L + int6 + MLP3x + SmearGate + BigramHash + WD 0.04 + SWA + FA3`
- mean `1.1326`

Key deltas vs your PR:
- int6, not int5-MLP
- FA3
- faster step time
- stronger base without needing TTT

### 5. Batch-size optimization matters more than it looked

`#236` found smaller batch was better in fixed wallclock:
- fewer tokens processed
- more optimizer steps
- better BPB

If your run is at `115 ms/step`, you are likely on the wrong side of that tradeoff.

## What is probably *not* the main issue

- `SmearGate`
- `BigramHash`
- `SWA`
- `OrthoInit`
- `WD 0.04`
- `sliding window stride=64`

Those are mostly already on the right path.

## What to change first

1. Drop `int5-MLP`; try `int6-all`.
2. Optimize for step count, not just tokens.
3. Copy the `#198/#236` training regime before adding TTT.
4. Only after that, re-add TTT from `#254`.

## Best next experiment order

1. `#264` minus TTT, minus int5, plus `int6-all`.
2. Tune batch to maximize steps in 600s.
3. Add FA3 if not already present.
4. Re-run TTT only on the stronger base.

## Sources

- Issue `#140`: <https://github.com/openai/parameter-golf/issues/140>
- PR `#264`: <https://github.com/openai/parameter-golf/pull/264>
- PR `#254`: <https://github.com/openai/parameter-golf/pull/254>
- PR `#236`: <https://github.com/openai/parameter-golf/pull/236>
- PR `#198`: <https://github.com/openai/parameter-golf/pull/198>
- PR `#194`: <https://github.com/openai/parameter-golf/pull/194>
