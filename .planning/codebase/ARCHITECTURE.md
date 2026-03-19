# Architecture

## Pattern

**Script-oriented ML research pipeline.** Not a web application or service — a collection of training scripts, data processing utilities, and deployment automation organized around YAML experiment configs.

## Core Data Flow

```
YAML Config → train.py → Trainer → Model Checkpoints → publish.py → HuggingFace Hub
     ↑                       ↑
  base.yaml              data.py (tokenization, dataset loading)
  (inherited)            tokenizer.py (BPE training)
```

## Key Modules (`src/slm/`)

| Module | Responsibility |
|--------|---------------|
| `train.py` | Main training entrypoint — loads config, builds model/tokenizer, runs HF Trainer. Supports `model_config` for custom architectures via `AutoConfig.for_model` |
| `train_sft.py` | Supervised fine-tuning (Alpaca/ChatML format) |
| `train_gec.py` | GEC (grammatical error correction) training |
| `train_seq2seq.py` | Sequence-to-sequence training |
| `data.py` | Dataset loading, tokenization, DDP-safe caching (rank 0 tokenizes, others wait) |
| `data_gec.py` / `data_gec_filtered.py` | GEC-specific data pipelines |
| `data_seq2seq.py` | Seq2seq data pipeline |
| `tokenizer.py` | ByteLevel BPE tokenizer training |
| `evaluate.py` | Model evaluation with text generation |
| `publish.py` | Upload models to HuggingFace Hub |
| `moe_upcycle.py` | Convert dense model → MoE (Mixtral-style) with shared router |
| `moe_trainer.py` | MoE-specific training logic |
| `utils.py` | Shared utilities |

## Cloud Pipeline (`src/slm/cloud/`)

| File | Responsibility |
|------|---------------|
| `__main__.py` | CLI entrypoint (launch, monitor, status, destroy) |
| `vastai.py` | vast.ai API wrapper |
| `gpu_selector.py` | GPU selection with filters (speed, disk, location, compatibility) |
| `remote_script.py` | Generates `run_cloud.sh` for remote execution |
| `provisioner.py` | Instance lifecycle: create → wait SSH → deploy → train → upload → destroy |

## Autonomous Research (`autoresearch/`)

Standalone DDP training scripts for large runs (300M, 600M) that bypass `src/slm/train.py`:
- `train_300m_ddp.py`, `train_600m_ddp.py` — direct PyTorch DDP training
- `upload_600m_to_hf.py` — manual HF upload
- `run_*.sh` — bash launchers for screen sessions

## Entry Points

| Entrypoint | Command |
|------------|---------|
| Training | `python -m slm.train --config <yaml>` |
| Evaluation | `python -m slm.evaluate --model_path <path> --prompts <file>` |
| Publishing | `python -m slm.publish --model_path <path> --repo_name <name>` |
| Cloud | `python -m slm.cloud launch --config <yaml> --hf-repo <repo>` |
| SFT | `python -m slm.train_sft --config <yaml>` |

## Config Inheritance

```
configs/base.yaml          ← shared defaults
  └── configs/experiments/expXXX_name.yaml  ← experiment overrides (inherits: base)
```

Resolution: experiment YAML merged over base. `model_config` dict enables custom Llama architectures without modifying code.
