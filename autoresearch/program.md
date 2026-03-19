# autoresearch-kazakh

Autonomous research for Kazakh LLM pretraining. Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Context

You are optimizing a **Kazakh language model** (Llama architecture, BPE 50K tokenizer) trained from scratch on 17.8B tokens of Kazakh text. The starting config is exp019: Llama 897M params, Chinchilla-optimal.

**Hardware**: 1× NVIDIA A100 80GB.

## Setup

1. **Read the files**: The repo is small. Read these for full context:
   - `README.md` (if exists) — repository context
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
2. **Verify data exists**: Check that `~/.cache/autoresearch-kazakh/data/` contains `train.bin` and `val.bin`. If not, run `uv run prepare.py`.
3. **Create `results.tsv`** with just the header row.
4. **Create a git branch**: `git checkout -b autoresearch/kazakh-exp019`

## Experimentation

Each experiment runs on a single A100 80GB GPU. Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/eval). Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — everything is fair game: architecture, optimizer, hyperparameters, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed evaluation, data loading, and constants.
- Install new packages or add dependencies.
- Modify the evaluation harness.

**The goal: get the lowest val_bpb** (bits per byte on Kazakh validation text).

**Time budget**: You have **1 hour total** = **12 experiment slots** of 5 minutes each. Use them wisely.

**VRAM budget**: A100 has 80GB. Some increase is OK for meaningful val_bpb gains.

**Simplicity criterion**: simpler is better, all else being equal.

## Key facts about this setup

- **Language**: Kazakh (Cyrillic script, agglutinative morphology)
- **Tokenizer**: BPE 50K vocab, ~5.2 bytes/token on Kazakh text
- **Data**: 200M tokens cached (subset of 9B available), seq_len=1024
- **Baseline arch**: Llama 897M (1536d, 24L, 24h, SwiGLU inter=5376)
- **A100 80GB**: ~312 TFLOPS bf16 peak. In 5 min you process roughly:
  - 897M model @ BS=4: ~7M tokens, ~1700 steps
  - 150M model @ BS=16: ~65M tokens, ~4000 steps
  - Consider the tradeoff: bigger model learns more per token, smaller model sees more tokens

## Experiment ideas (starting points)

1. **Baseline**: Run train.py as-is to get baseline val_bpb
2. **Model size sweep**: Try 150M, 300M, 500M, 900M — find the sweet spot for 5-min training
3. **Learning rate**: Explore 1e-4 to 1e-3 range
4. **Batch size**: A100 80GB allows large batches. Try 8, 16, 32
5. **Warmup**: More aggressive warmup for short training
6. **Architecture**: GQA (n_kv_head < n_head), different depth/width ratios
7. **Optimizer**: Try different betas, higher weight decay
8. **torch.compile**: Enable/disable, measure overhead vs speed

## Output format

The script prints a summary block:

```
---
val_bpb:          X.XXXXXX
training_seconds: 300.X
total_seconds:    XXX.X
peak_vram_mb:     XXXXX.X
mfu_percent:      XX.XX
total_tokens_M:   XXX.X
num_steps:        XXXX
num_params_M:     XXX.X
depth:            XX
```

Extract key metrics: `grep "^val_bpb:\|^peak_vram_mb:" run.log`

## Logging results

Log to `results.tsv` (tab-separated). Header + 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

- commit: short git hash (7 chars)
- val_bpb: e.g. 1.234567 (0.000000 for crashes)
- memory_gb: peak VRAM in GB (0.0 for crashes)
- status: `keep`, `discard`, or `crash`
- description: what this experiment tried

## The experiment loop

LOOP (12 times, 1 hour total):

1. Look at git state
2. Edit `train.py` with an experimental idea
3. `git commit -am "description of change"`
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If empty (crash): `tail -n 50 run.log`, attempt fix
7. Record in results.tsv (do NOT commit this file)
8. If val_bpb improved → keep commit, advance branch
9. If val_bpb same or worse → `git reset --hard HEAD~1` (revert)

**Timeout**: If a run exceeds 10 minutes, kill it (crash).

**Strategy**: Start with baseline, then systematically explore. Don't change too many things at once. Keep notes in the description column.
