# Rules for AI Agents and Contributors

## Project: SLM — Small Language Model for Kazakh

### Core Principles

1. **Reproducibility**: Every experiment must be fully reproducible from a YAML config.
2. **Modularity**: `train`, `data`, `tokenizer`, `evaluate`, `publish` are independent modules.
3. **Logging**: All experiment results go into `WHITEPAPER.md` with metrics, hyperparameters, and conclusions.
4. **No hardcoded paths**: Use configs and CLI args. No absolute paths in code.
5. **Minimal dependencies**: Only add dependencies that are truly needed.

### Code Style

- Python 3.10+, type hints where practical
- Use `ruff` for linting
- Keep functions focused and small
- Docstrings for public functions only

### Experiment Workflow

1. Create or modify a YAML config in `configs/experiments/`
2. Run training: `python -m slm.train --config <path>`
3. Evaluate: `python -m slm.evaluate --model_path <path> --prompts eval/prompts_kk.txt`
4. Record results in `WHITEPAPER.md`
5. If results are good, publish: `python -m slm.publish --model_path <path> --repo_name <name>`

### Config Inheritance

Experiment configs use `inherits: base` to inherit from `configs/base.yaml`. Experiment-specific values override the base.

### Experiment Naming and Config Creation

**Every new experiment MUST have its own YAML config file.** Never reuse or modify an existing experiment config for a new run.

- Naming convention: `configs/experiments/exp{NNN}_{short_description}.yaml`
- Increment the experiment number sequentially (check existing configs first)
- The `experiment_name` field inside the YAML must match the filename (without `.yaml`)
- Each config is a permanent record of what was run — treat them as immutable after launch

Examples:
```
configs/experiments/exp004_scratch_kk.yaml        # 50M Llama from scratch
configs/experiments/exp005_balanced_50m.yaml       # 50M Llama on balanced dataset
configs/experiments/exp006_balanced_150m.yaml      # 150M Llama on balanced dataset
```

### Cloud Training (vast.ai)

Launch experiments on vast.ai using the cloud pipeline:

```bash
PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud launch \
  --config configs/experiments/exp005_balanced_50m.yaml \
  --hf-repo saken-tukenov/<model-name> \
  --max-price 0.50 --num-gpus 1 --disk 60
```

- Always do a `--dry-run` first to check GPU prices
- Use `pretokenized_dataset` in config when the dataset is already tokenized on HF Hub
- The `dataset_name` field is still required (used for logging) even with pretokenized datasets
- Monitor: `python -m slm.cloud monitor --instance-id <ID>`
- Destroy: `python -m slm.cloud destroy --instance-id <ID>`

#### GPU Instance Selection Rules

When selecting GPU instances on vast.ai, **always** enforce these criteria:

1. **Network speed**: Minimum **500 Mbps real download** (vast.ai advertised speeds are often inflated 10-50x). Prefer instances with `inet_down >= 1000` in API.
2. **Disk space**: Minimum **60 GB** (Docker image ~35GB + dataset + checkpoints).
3. **Location**: Prefer **Western Europe (NL, DE, FR, FI)** or **US East Coast** — best peering to HuggingFace Hub CDN (AWS us-east). Avoid Turkey, CIS, Asia — slow HF downloads (~5 MB/s real vs 50+ MB/s in EU/US).
4. **GPU compatibility**: Do NOT use RTX 5090/5080 (Blackwell sm_120) — Docker image `pytorch/pytorch:2.4.1-cuda12.4` lacks Blackwell kernels. Stick to Ampere/Ada/Hopper: A100, H100, H200, RTX 4090, L40S, A6000.
5. **Reliability**: `reliability > 0.95` (already in code).
6. **Price efficiency**: Rank by **total estimated cost** (price/hr × estimated hours), not just price/hr. A faster GPU at higher $/hr can be cheaper overall.

#### Known Issues
- `ssh -f` with nohup times out after 30s on slow instances — if provisioner crashes at "Starting training", the instance is likely alive. Check with `ssh` manually and start `run_cloud.sh` if needed.
- `EADDRINUSE` error on multi-GPU: kill leftover torchrun processes before restarting.

### What NOT to do

- Do not modify `base.yaml` for experiment-specific changes
- Do not push models to HuggingFace Hub without evaluation
- Do not add large files to git (use `.gitignore`)
- Do not install packages outside of `pyproject.toml`

---

## HuggingFace Naming Standard (SozKZ)

All repos under `saken-tukenov/` follow the SozKZ naming convention.

### Brand Lines

| Line | Prefix | Scope |
|------|--------|-------|
| Core | `sozkz-core` | Base / foundation models (from-scratch, pretrained) |
| Vocab | `sozkz-vocab` | Tokenizers, vocabularies |
| Corpus | `sozkz-corpus` | Datasets, corpora, preprocessing |
| Fix | `sozkz-fix` | GEC / error correction models |
| Seq | `sozkz-seq` | Seq2seq / T5-like models |
| MoE | `sozkz-moe` | Mixture-of-Experts models |

### Slug Format

```
sozkz-<artifact>-<arch>-<size>-<lang>-<variant>-v<major>[.<minor>]
```

Fields:
- **artifact**: `core | vocab | corpus | fix | seq | moe`
- **arch**: `llama | gpt2 | t5 | mt5 | pythia | mix | bpe | sp` (extensible)
- **size**: `50m | 200m | 3b | 32k` etc.
- **lang**: `kk` or `kk-ru`
- **variant**: `base | instruct | clean | synthetic | sft | balanced | domain` etc.
- **version**: `v1`, `v2`, minor: `v1.1`

### Type-Specific Formats

**Models:**
`sozkz-{core|fix|seq|moe}-<arch>-<size>-<lang>-<variant>-v<major>`

**Tokenizers:**
`sozkz-vocab-<type>-<vocabsize>-<lang>-<notes>-v<major>`

**Datasets:**
`sozkz-corpus-<stage>-<lang>-<source>-<notes>-v<major>`
Stages: `raw | clean | dedup | tokenized | balanced | synthetic`

**Spaces:**
`sozkz-demo-<line>-<task>-<lang>-v<major>`

### Uniqueness Check (mandatory before publishing)

1. Check `https://huggingface.co/saken-tukenov/<slug>` returns 404
2. Search `site:huggingface.co "<slug>"` for conflicts
3. If collision found, add uniquifier: `-st`, `-kz`, or specific tag

### Versioning

- **Major** (`v1` → `v2`): breaking changes (weights, tokenizer, architecture, data format)
- **Minor** (`v1.1`): non-breaking (README, metadata, scripts)

### Deprecation

Old repos: add `DEPRECATED` tag + link to successor in README. Do not delete.

### README Requirements

Every repo README must start with:
1. Purpose (1–2 sentences)
2. Input/output format
3. License
4. Date/version
5. Related repos (links to tokenizer/dataset/model)

### Tags (mandatory)

- `language`: `kk` (and `ru` if bilingual)
- `task`: `text-generation`, `translation`, `text2text-generation`, `grammatical-error-correction` etc.
- `library`: `transformers`, `tokenizers`, `datasets`

### Adding New Lines

A new brand line requires:
- Covers a distinct task class not fitting existing 6 lines
- At least 2 repos planned
- Passes uniqueness check
- Documented in a short RFC in the collection README

### Pre-Publish Checklist

1. Pick line (core/vocab/corpus/fix/seq/moe)
2. Build slug from template
3. Check uniqueness (see above)
4. Create repo
5. Fill README (see requirements)
6. Add to HF collection
7. Set tags and cross-links
