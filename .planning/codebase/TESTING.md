# Testing

## Framework

- **pytest** (>=8.0) in `dev` optional dependencies
- No structured test suite — this is a research codebase

## Existing Tests

| File | Purpose |
|------|---------|
| `test_gec.py` | GEC pipeline evaluation (root level) |

## Test Patterns

- **Smoke tests via config:** `smoke_cloud.yaml` — minimal training run (10 steps) to verify pipeline works
- **Manual evaluation:** `python -m slm.evaluate` with prompt files from `eval/`
- **Cloud dry-run:** `--dry-run` flag on cloud pipeline to verify GPU selection without creating instances
- **vast.ai smoke test:** Documented as passing (2026-02-07) — RTX 3060, 50 steps

## Coverage

- No coverage tooling configured
- No CI/CD pipeline
- Testing is manual/script-based, typical for ML research projects

## Evaluation

Evaluation is separate from unit testing:
- `eval/` contains Kazakh text prompts
- `scripts/eval/` contains evaluation pipelines
- `results/` stores inference outputs, judge results, benchmarks
- Benchmarks: arena-style, MC-based, judge-based evaluation
