# Morphology-Aware Minimal-Edit GEC for Kazakh

Research subproject: dual-model pipeline (NLLB seq2seq + edit tagger) with 3-level error taxonomy and multi-reference evaluation.

## Setup

```bash
cd gec-paper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Structure

- `src/gecpaper/` — library code
- `scripts/` — CLI entry points
- `configs/` — experiment YAML configs
- `tests/` — unit tests
- `paper/` — LaTeX source
