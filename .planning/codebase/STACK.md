# Stack

## Languages & Runtime

| Language | Version | Usage |
|----------|---------|-------|
| Python | 3.10+ | Primary — training, evaluation, data processing, cloud pipeline |
| YAML | — | Experiment configs with inheritance |
| Bash | — | Training launch scripts, Ansible automation |

## Core Frameworks

| Framework | Version | Purpose |
|-----------|---------|---------|
| PyTorch | >=2.1 (2.10+cu128 on server) | Training runtime, DDP multi-GPU |
| Transformers | >=4.40 (5.1 on server) | Model architectures, tokenizers, Trainer |
| Datasets | >=2.18 | Data loading, streaming, pre-tokenized datasets |
| Accelerate | >=0.28 | Distributed training coordination |
| Tokenizers | >=0.15 | BPE tokenizer training (ByteLevel) |
| HuggingFace Hub | >=0.22 | Model/dataset upload and download |

## Additional Dependencies

| Package | Purpose |
|---------|---------|
| PyYAML | Config loading with safe_load |
| TensorBoard | Training metrics visualization |
| SentencePiece | Alternative tokenizer backend |
| vastai | vast.ai CLI for cloud GPU provisioning |

## Build & Package

- **Build system:** Hatchling (`pyproject.toml`)
- **Package:** `src/slm/` (wheel target)
- **Optional groups:** `dev` (ruff, pytest), `cloud` (vastai)
- **Lock file:** `uv.lock` (uv package manager)

## Configuration

- **Base config:** `configs/base.yaml` — shared hyperparameters
- **Experiment configs:** `configs/experiments/exp{NNN}_{name}.yaml` — inherit from base via `inherits: base`
- **Config resolution:** experiment YAML overrides base; `model_config` dict for custom architectures
- **Known issue:** YAML `safe_load` returns strings for floats — `float()` cast needed for `learning_rate`, `warmup_ratio`, `weight_decay`

## Compute Environments

| Environment | Hardware | Purpose |
|-------------|----------|---------|
| kaznu server | 2× NVIDIA A10 (23GB) | Training, inference API |
| vast.ai | Variable (A100, H100, RTX 4090) | Large-scale training |
| Local (macOS) | CPU only | Development, config creation |

## Key Constraints

- **bf16 required** on A10 GPUs (fp16 causes instability)
- **Local tokenizer path** on server — HF loading broken with transformers 5.1
- **DDP rank coordination** — only rank 0 tokenizes, others wait
- **No RTX 5090/5080** on vast.ai — Docker image lacks Blackwell kernels
