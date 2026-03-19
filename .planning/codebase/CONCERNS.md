# Concerns

## Security

### Hardcoded Secrets (FIXED)
- ~~HF tokens in `scripts/update_600m_card.py` and `scripts/inference/eval_gec_pipeline.py`~~ — replaced with `os.environ.get("HF_TOKEN")` before GitHub push
- `.env` file with HF_TOKEN is properly gitignored

### Server Credentials
- `ansible/inventory.ini` contains server IPs and SSH details — in git but acceptable for private research repo
- SSH key paths referenced but keys themselves not committed

## Technical Debt

### Duplicate Training Scripts
- `src/slm/train.py` — main training entrypoint via config
- `autoresearch/train_300m_ddp.py`, `autoresearch/train_600m_ddp.py` — standalone DDP scripts that bypass main entrypoint
- These evolved independently and have diverged in features

### Transformers 5.1 Incompatibilities
- HF tokenizer loading broken on server — forced local path workaround
- `_tied_weights_keys` is now dict (not list)
- MoE API changed (`block_sparse_moe` → `.mlp`, `experts` no longer subscriptable)
- `overwrite_output_dir` and `tokenizer` args deprecated

### Failed Experiments Left In-Tree
- `kz-calm/` — TTS experiments (failed, postmortem documented)
- `omniaudio/` — audio experiments
- `nanochat-kazakh/` — chat experiments
- These add size but serve as reference/documentation

## Fragile Areas

### SSH Connectivity
- kaznu server connection is flaky — needs `ConnectTimeout=120`
- `pkill -f` pattern matching kills SSH process itself — must use `ps | grep | awk | xargs kill`

### vast.ai Provisioning
- `ssh -f` with nohup times out on slow instances
- `EADDRINUSE` on multi-GPU — leftover torchrun processes
- RTX 5090/5080 incompatible with current Docker image

### YAML Float Parsing
- `safe_load` returns strings for scientific notation (e.g., `1e-4`)
- Missing `float()` cast causes silent type errors in training

## Performance

### Tokenization Bottleneck
- Full dataset tokenization takes hours with 4 workers
- Mitigation: pre-tokenized datasets on HF Hub
- `num_proc` should be increased for future raw dataset runs

### No Caching Strategy
- Each training run re-downloads datasets if cache missing
- Pre-tokenized datasets on HF Hub partially solve this

## Architecture Concerns

### No Abstraction Layer
- Each training variant (train, train_sft, train_gec, train_seq2seq) is a separate module
- Common patterns (config loading, model init, data prep) duplicated across modules
- Acceptable for research code but would need refactoring for production use
