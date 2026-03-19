# Conventions

## Code Style

- **Python 3.10+**, type hints where practical
- **Linter:** ruff (line-length=120, target py310)
- **Docstrings:** public functions only
- **Imports:** standard library → third-party → local

## Naming

| Entity | Convention | Example |
|--------|-----------|---------|
| Experiment | `exp{NNN}_{description}` | `exp023_llama_600m` |
| Config file | matches experiment name | `exp023_llama_600m.yaml` |
| HF model | SozKZ standard | `sozkz-core-llama-600m-kk-base-v1` |
| Python module | snake_case | `train_sft.py`, `data_gec.py` |
| Functions | snake_case | `load_and_tokenize_dataset()` |
| Classes | PascalCase | `GrammarCorrectionModel` |

## Patterns

### Config Inheritance
All experiment configs use `inherits: base` to pull from `configs/base.yaml`. Experiment-specific values override base. Never modify base for experiment-specific changes.

### DDP Safety
Data processing uses rank-aware coordination:
```python
# Only rank 0 tokenizes; other ranks wait
if local_rank == 0:
    tokenized = dataset.map(tokenize_fn, ...)
torch.distributed.barrier()
```

### Float Casting
YAML `safe_load` returns strings for scientific notation — all numeric hyperparams need explicit `float()`:
```python
learning_rate = float(config.get("learning_rate", 1e-4))
```

### Model Config Dict
Custom architectures defined inline in YAML via `model_config` dict, loaded with `AutoConfig.for_model("llama")`.

## Error Handling

- Training scripts: minimal — let HF Trainer handle errors
- Cloud pipeline: defensive — instance stays alive on failure for debugging, self-destructs only on success
- Data loading: DDP barrier pattern prevents race conditions

## Documentation

- All experiment results → `WHITEPAPER.md` (mandatory)
- Project rules → `agents.md`
- No inline documentation beyond docstrings on public functions
