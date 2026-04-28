# EXP-037: Synthetic Kazakh GEC 5K via CloudRift Qwen3.5

**Date**: 2026-04-12  
**Status**: In progress  
**Goal**: Build a high-precision synthetic Kazakh GEC dataset of ~5K examples using CloudRift-hosted Qwen, with strict verification to minimize noisy or meaning-changing pairs.

## Why This Experiment Exists

We need a compact but reliable grammar-correction dataset where the final correction model uses a **single universal prompt** and learns to:

- fix grammar
- fix spelling
- fix punctuation
- fix word usage
- preserve meaning
- leave already-correct text unchanged

The main risk is synthetic noise: if the teacher model introduces semantic drift or low-quality corruptions, the downstream GEC model will learn bad behavior. Because of that, this experiment uses **generate -> verify -> filter** rather than trusting one LLM pass.

## Final Universal Training Prompt

Every training row uses the same instruction:

```text
Мәтіндегі грамматикалық, орфографиялық, пунктуациялық және сөз қолданысындағы қателерді түзет. Мағынаны өзгертпе. Егер мәтін дұрыс болса, оны өзгеріссіз қайтар. Тек түзетілген мәтінді қайтар.
```

This is the prompt the final GEC model should internalize.

## Generation Scheme

We use a separate teacher prompt to convert clean Kazakh text into natural-looking erroneous text while preserving meaning.

Pipeline:

1. Collect clean Kazakh seed sentences.
2. Generate an erroneous version with Qwen.
3. Run deterministic filters.
4. In strict mode, run a second Qwen judge pass.
5. In strict mode, run a round-trip correction pass.
6. Keep only accepted pairs.
7. Add identity examples so the final model learns not to overcorrect.

## Key Design Decisions

### 1. Single-task downstream format

We do **not** train the final model on multiple tags like `<септік>` or `<шақ>`.  
We only use one universal correction instruction, because that matches the desired product behavior.

### 2. Teacher prompt is separate from student prompt

The teacher is not asked to correct text.  
It is asked to create realistic erroneous text from a correct sentence.

### 3. Identity examples are mandatory

To avoid "always rewrite" behavior, we include clean examples:

- `input == output`
- same universal correction prompt

### 4. Strict verification

Synthetic pairs are accepted only if they pass:

- structural validation
- semantic drift checks
- optional Qwen judge verification
- optional round-trip correction recovery

## Files

### New / updated files in this experiment

- [scripts/data/gec_generate_qwen_cloudrift.py](/Users/sakentukenov/slm/scripts/data/gec_generate_qwen_cloudrift.py:1)
  Main generator for synthetic GEC pairs through CloudRift Qwen.

- [scripts/data/gec_merge_upload.py](/Users/sakentukenov/slm/scripts/data/gec_merge_upload.py:1)
  Merge, deduplicate, and optionally upload generated chunks.

- [scripts/data/gec_validate_candidates_cloudrift.py](/Users/sakentukenov/slm/scripts/data/gec_validate_candidates_cloudrift.py:1)
  Strict second-pass validator for candidate pairs.

### Existing related files

- [scripts/data/gec_generate_v2.py](/Users/sakentukenov/slm/scripts/data/gec_generate_v2.py:1)
  Earlier local GPT-OSS generation pipeline for typed GEC data.

- [scripts/data/gec_generate_v3.py](/Users/sakentukenov/slm/scripts/data/gec_generate_v3.py:1)
  Earlier optimized local generation pipeline.

- [scripts/inference/serve_gec_600m.py](/Users/sakentukenov/slm/scripts/inference/serve_gec_600m.py:1)
  Current single-tag GEC serving path; useful as reference for final user-facing behavior.

- [autoresearch/EXPERIMENT_LOG.md](/Users/sakentukenov/slm/autoresearch/EXPERIMENT_LOG.md:370)
  Notes from earlier GEC phases showing why noisy data and overcorrection are dangerous.

## Generator Behavior

The generator currently supports:

- CloudRift API access
- Qwen3.5-122B teacher generation
- profile-based corruption:
  - morphology
  - spelling
  - punctuation
  - word order
  - function words
  - mixed light
  - mixed hard
- clean identity examples
- progress saving
- cost tracking
- dry-run mode
- strict verification mode

## Strict Verification Logic

### Deterministic filters

Reject if:

- `incorrect == correct`
- low Cyrillic ratio
- strong length shift
- low lexical overlap
- edit distance too large
- polarity / negation shift

### Judge verification

A second Qwen pass checks:

- incorrect really contains an error
- correct is actually a correction
- meaning is preserved
- pair is not a strong paraphrase

Only `accept` with high confidence is kept.

### Round-trip verification

The incorrect text is sent through the universal correction prompt again.
If the model does not recover the expected correct text closely enough, the pair is rejected.

## Expected Dataset Shape

Target production mix for `~5K` rows:

- `~65-70%` synthetic error pairs
- `~30-35%` identity pairs

This is intentionally conservative to reduce false positives in downstream correction.

## Example Output Row

```json
{
  "instruction": "Мәтіндегі грамматикалық, орфографиялық, пунктуациялық және сөз қолданысындағы қателерді түзет. Мағынаны өзгертпе. Егер мәтін дұрыс болса, оны өзгеріссіз қайтар. Тек түзетілген мәтінді қайтар.",
  "input": "Компания нарыққа жаңа өнімдерін биыл шығарады.",
  "output": "Компания биыл жаңа өнімдерін нарыққа шығарады.",
  "source": "synthetic_gec_qwen35_cloudrift",
  "meta": {
    "profile": "word_order"
  }
}
```

## Commands

### Strict dry run

```bash
python3 scripts/data/gec_generate_qwen_cloudrift.py \
  --output_dir /tmp/gec_qwen_strict_dryrun \
  --seeds_file /tmp/gec_seeds_manual.txt \
  --target_pairs 12 \
  --limit 13 \
  --dry-run \
  --strict \
  --concurrency 4
```

### Production run

```bash
python3 scripts/data/gec_generate_qwen_cloudrift.py \
  --output_dir ./gec_qwen_5k \
  --target_pairs 12000 \
  --concurrency 12 \
  --budget_usd 15
```

### Strict validation of candidate pool

```bash
python3 scripts/data/gec_validate_candidates_cloudrift.py \
  --input ./gec_qwen_5k/chunks/synthetic.jsonl \
  --output_dir ./gec_qwen_5k_validated \
  --concurrency 12
```

### Merge and upload

```bash
python3 scripts/data/gec_merge_upload.py \
  --input_dir ./gec_qwen_5k \
  --repo stukenov/sozkz-corpus-synthetic-kk-gec-qwen35-v1 \
  --upload
```

## Notes From This Session

Main decisions made during implementation:

1. Teacher-only generation is not reliable enough by itself.
2. Identity examples must be baked in from the start.
3. Semantic drift is the most dangerous failure mode.
4. Negation shifts are especially toxic and must be filtered.
5. A strict mode with judge + round-trip verification is worth the extra token cost.

## Run Log

### Dry run v1

- mode: non-strict
- outcome: usable pairs, but some generations returned identity and one semantic drift case appeared
- conclusion: add judge + round-trip + negation-shift filter

### Dry run v2

- mode: non-strict with retry and negation filter
- outcome: better yield and cleaner pairs
- conclusion: strict mode should be default for production generation

### Dry run v3

- mode: strict
- command:

```bash
python3 scripts/data/gec_generate_qwen_cloudrift.py \
  --output_dir /tmp/gec_qwen_strict_dryrun \
  --seeds_file /tmp/gec_seeds_manual.txt \
  --target_pairs 12 \
  --limit 13 \
  --dry-run \
  --strict \
  --concurrency 4
```

- result:
  - `synthetic=0`
  - `identity=3`
  - `req=38`
  - cost about `$0.019`
  - failures:
    - `identity_as_error: 3`
    - `judge_bad_json: 1`
    - `judge_reject: 1`
    - `roundtrip_mismatch: 4`

- conclusion:
  - current strict mode is **too strict** for production as implemented
  - the main blocker is round-trip mismatch, not API instability
  - next iteration should relax round-trip acceptance from exact-or-near match to semantic-equivalence style checks
  - judge should also be made more schema-stable, because one rejection came from bad JSON formatting rather than clearly bad content

### Dry run v4

- mode: strict-v2
- change:
  - round-trip verification relaxed from near-exact text match to semantic equivalence check
  - deterministic filters kept
  - judge kept

- command:

```bash
python3 scripts/data/gec_generate_qwen_cloudrift.py \
  --output_dir /tmp/gec_qwen_strict_dryrun_v2 \
  --seeds_file /tmp/gec_seeds_manual.txt \
  --target_pairs 12 \
  --limit 13 \
  --dry-run \
  --strict \
  --concurrency 4
```

- result:
  - `synthetic=2`
  - `identity=3`
  - `req=43`
  - cost about `$0.0193`
  - failures:
    - `judge_reject: 1`
    - `roundtrip_low_cyr: 2`
    - `identity_as_error: 1`
    - `negation_shift: 1`
    - `roundtrip_low_overlap: 2`

- interpretation:
  - acceptance is now non-zero
  - accepted pairs are noticeably cleaner than teacher-only mode
  - precision improved, but recall is still too low for an efficient 5K run
  - main remaining bottleneck is round-trip output formatting and overlap checks

### Dry run v5

- mode: strict-v3
- change:
  - added round-trip text extraction from JSON-like responses

- result:
  - `synthetic=1`
  - `identity=3`
  - `req=33`
  - cost about `$0.0166`
  - failures:
    - `identity_as_error: 3`
    - `judge_reject: 2`
    - `negation_shift: 1`
    - `roundtrip_low_overlap: 1`
    - `roundtrip_low_cyr: 1`

- interpretation:
  - parser fix removed one technical failure mode, but overall strict recall did not improve on this tiny sample
  - the remaining issue is not parsing but teacher generation quality under strict semantic constraints
  - strict mode is good as a high-precision validator, but too expensive and too low-recall to be the only generation path

### Validator check on candidate pool

- input:
  - teacher-only candidate pool from `/tmp/gec_qwen_dryrun4/chunks/synthetic.jsonl`
  - `7` candidate rows

- command:

```bash
python3 scripts/data/gec_validate_candidates_cloudrift.py \
  --input /tmp/gec_qwen_dryrun4/chunks/synthetic.jsonl \
  --output_dir /tmp/gec_validated_from_candidates \
  --concurrency 4
```

- result:
  - `accepted=1`
  - `rejected=6`
  - extra validation spend about `$0.0032`

- interpretation:
  - the two-stage pipeline works mechanically
  - current strict validator behaves like a high-precision narrow filter
  - production should therefore generate a much larger candidate pool than the final desired 5K size

## Next Step

Current recommendation before launching full 5K:

1. Keep deterministic filters and negation-shift rejection.
2. Keep retry generation.
3. Use teacher-only generation to produce a large candidate pool.
4. Use strict mode only as a second-pass validator on that pool, not as the sole online generation path.
5. Expect low validator recall; generate at least `2-4x` more candidates than the final target.
6. Increase seed pool substantially before the full run.
7. Only then launch the full 5K generation.
