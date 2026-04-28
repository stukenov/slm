# Parameter Golf #140: other PRs with useful insights

Snapshot: 2026-03-21.

Use this as an addendum to [README.md](/Users/sakentukenov/slm/docs/parameter-golf-140/README.md).

## Frontier updates

- `#198` `@jfprincz` `1.1326`
  - Stack: `11L + int6 + MLP3x + SmearGate + BigramHash + WD 0.04 + SWA + FA3`.
  - Insight: this is the current reference stack. If you copy one PR now, copy this family, not the older `#162`.

- `#236` `@saml212` `1.1400`
  - Stack: `11L int6 + SmearGate + BigramHash + 524K batch + SWA + WD 0.04`.
  - Insight: batch size itself is a real lever. Very large effective batch still helped at this scale.

- `#194` `@baudrillardsgh0st` `1.1480`
  - Stack: `11L + int6 QAT + per-dim SmearGate + SWA + WD 0.038`.
  - Insight: QAT is competitive enough to keep on the table; per-dimension gating is worth testing.

- `#192` `@baudrillardsgh0st` `1.1502`
  - Stack: `11L + int6 QAT + SmearGate + WD 0.038 + MLP3x`.
  - Insight: cleaner earlier QAT variant; useful if you want a simpler starting point than `#194`.

- `#64` `@yesbhautik` `1.1465`
  - Stack: `SmearGate + BigramHash + int6/int8 + WD 0.04 + SWA/50`.
  - Insight: mixed precision/quantization can work without the full latest stack.

- `#254` `@timowhite88` `1.1303` `1 seed`
  - Stack: `11L int6 MLP3x + SmearGate + TTT (3 epochs SGD) + SWA + WD 0.04`.
  - Insight: TTT still stacks on top of SmearGate, just with smaller gain than on weaker bases.

## Eval-time ideas

- `#206` `@dexhunter` `1.1507`
  - Idea: `RoPE base 50K`.
  - Insight: zero-artifact, low-risk tweak. Easy to compose with the strong 11-layer stacks.

- `#241` `@kellyvv`
  - Idea: `Stride-OGD` over a vocab-bias vector during eval.
  - Insight: cheap online adaptation candidate if TTT is too slow.

- `#108` `@kellyvv`
  - Idea: `error correction table`.
  - Insight: store only model mistakes, not the full prefix. More artifact-efficient than brute-force paid prefix.

## High-risk directions worth remembering

- `#203` `@LexHarie`
  - Idea: context-length curriculum, `1024 -> 2048`.
  - Insight: more optimizer steps early, more context later. Still worth trying if wallclock is the bottleneck.

- `#224` `@Complexity-ML`
  - Idea: small MoE with deterministic routing.
  - Insight: first serious MoE attempt. Routing was weak, but the class of idea is still open.

- `#139` `@ksang123`
  - Idea: ternary / BitNet-style model.
  - Insight: raw ternary lagged behind int6, but the issue summary suggests pairing it with sliding eval + SmearGate if someone wants to rescue it.

## Negative results that actually matter

- `#201` `@machdragon`
  - Result: `LAWA-EMA` lost badly to periodic `SWA`.
  - Rule: keep SWA; do not replace it with EMA just because it looks smoother.

- `#219`
  - Result: `12L + int5` lost to `11L`.
  - Rule: extra depth is not free; slower steps can erase the gain.

- `#123`
  - Result: larger vocab `4096` with fewer layers underperformed.
  - Rule: at this artifact budget, depth beats vocab breadth.

- `#200`
  - Result: `SP4096 + int6 QAT + NorMuon` also underperformed.
  - Rule: same lesson as `#123`; bigger vocab was not the win.

- `#212`
  - Result bundle:
  - `SmearGate + BigramHash` hurts without `OrthoInit`
  - `SWA` accumulation must be `fp32`, not `bf16`
  - `MTP` gave no gain
  - content curriculum gave no gain
  - Rule: this is the most useful ablation PR to read before wasting runs.

- `#238`
  - Result: with many enough SWA checkpoints, quantized BPB can beat pre-quant BPB.
  - Rule: do not assume quantization gap always stays negative; SWA can flip it.

## Short copy order

1. Read `#198`.
2. Read `#236`.
3. Read `#212`.
4. If you want eval tricks, read `#254`, `#206`, `#241`, `#108`.
5. If you want weird ideas, read `#203`, `#224`, `#139`.

## Sources

- Issue `#140`: <https://github.com/openai/parameter-golf/issues/140>
- PR `#198`: <https://github.com/openai/parameter-golf/pull/198>
- PR `#236`: <https://github.com/openai/parameter-golf/pull/236>
- PR `#194`: <https://github.com/openai/parameter-golf/pull/194>
- PR `#192`: <https://github.com/openai/parameter-golf/pull/192>
- PR `#64`: <https://github.com/openai/parameter-golf/pull/64>
- PR `#254`: <https://github.com/openai/parameter-golf/pull/254>
- PR `#206`: <https://github.com/openai/parameter-golf/pull/206>
- PR `#241`: <https://github.com/openai/parameter-golf/pull/241>
- PR `#108`: <https://github.com/openai/parameter-golf/pull/108>
- PR `#203`: <https://github.com/openai/parameter-golf/pull/203>
- PR `#224`: <https://github.com/openai/parameter-golf/pull/224>
- PR `#139`: <https://github.com/openai/parameter-golf/pull/139>
- PR `#201`: <https://github.com/openai/parameter-golf/pull/201>
- PR `#219`: <https://github.com/openai/parameter-golf/pull/219>
- PR `#123`: <https://github.com/openai/parameter-golf/pull/123>
- PR `#200`: <https://github.com/openai/parameter-golf/pull/200>
- PR `#212`: <https://github.com/openai/parameter-golf/pull/212>
- PR `#238`: <https://github.com/openai/parameter-golf/pull/238>
