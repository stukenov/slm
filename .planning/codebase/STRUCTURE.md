# Structure

## Directory Layout

```
slm/
├── src/slm/                    # Core Python package
│   ├── train.py                # Main training entrypoint
│   ├── train_sft.py            # SFT training
│   ├── train_gec.py            # GEC training
│   ├── train_seq2seq.py        # Seq2seq training
│   ├── data.py                 # Dataset loading & tokenization
│   ├── data_gec.py             # GEC data pipeline
│   ├── data_gec_filtered.py    # Filtered GEC data
│   ├── data_seq2seq.py         # Seq2seq data pipeline
│   ├── tokenizer.py            # BPE tokenizer training
│   ├── evaluate.py             # Model evaluation
│   ├── publish.py              # HuggingFace upload
│   ├── moe_upcycle.py          # Dense → MoE conversion
│   ├── moe_trainer.py          # MoE training
│   ├── utils.py                # Shared utilities
│   └── cloud/                  # vast.ai cloud pipeline
│       ├── __main__.py         # CLI (launch/monitor/destroy)
│       ├── vastai.py           # API wrapper
│       ├── gpu_selector.py     # GPU filtering/ranking
│       ├── remote_script.py    # Remote script generation
│       └── provisioner.py      # Instance lifecycle
│
├── configs/
│   ├── base.yaml               # Shared training defaults
│   ├── collect.yaml             # Data collection config
│   ├── clean_corpus*.yaml       # Corpus cleaning configs
│   └── experiments/             # 35 experiment configs (exp001–exp026 + smoke)
│
├── scripts/                     # Utility scripts
│   ├── inference/               # Inference & evaluation scripts
│   ├── eval/                    # Evaluation pipelines
│   ├── hub/                     # HuggingFace hub operations
│   ├── tokenizer/               # Tokenizer training/testing
│   ├── training/                # Training helpers
│   ├── data/                    # Data preparation
│   └── translate/               # Translation utilities
│
├── autoresearch/                # Autonomous training scripts
│   ├── train_300m_ddp.py        # 300M DDP training
│   ├── train_600m_ddp.py        # 600M DDP training
│   ├── run_*.sh                 # Launch scripts
│   └── 5090/                    # RTX 5090 benchmarks
│
├── ansible/                     # Server automation
│   ├── deploy.yml               # Code deployment
│   ├── run_experiment.yml       # Training launch
│   └── fetch_results.yml        # Results download
│
├── eval/                        # Evaluation prompts & benchmarks
├── tokenizers/kazakh-bpe-32k/   # Trained tokenizer files
├── results/                     # Evaluation outputs & logs
├── docs/                        # Model cards, papers, plans
├── kz-calm/                     # TTS experiments (failed)
├── nano/                        # Custom architecture experiments
├── nanochat-kazakh/             # Chat model experiments
├── omniaudio/                   # Audio model experiments
├── notebooks/                   # Jupyter notebooks
├── spaces/                      # HF Spaces
├── translation-demo/            # Translation pipeline (CTranslate2)
│
├── pyproject.toml               # Package definition
├── uv.lock                      # Dependency lock file
├── WHITEPAPER.md                # Experiment log & results
├── agents.md                    # AI agent rules & naming standards
└── README.md                    # Project overview
```

## Naming Conventions

- **Experiments:** `exp{NNN}_{short_description}` (exp013_llama_50m_9b)
- **Configs:** match experiment name + `.yaml`
- **HuggingFace models:** `sozkz-{line}-{arch}-{size}-{lang}-{type}-v{N}` (sozkz-core-llama-600m-kk-base-v1)
- **Scripts:** descriptive snake_case (tokenize_and_push.py, eval_gec_pipeline.py)

## Key File Locations

| What | Where |
|------|-------|
| Training entrypoint | `src/slm/train.py` |
| Experiment configs | `configs/experiments/*.yaml` |
| Base config | `configs/base.yaml` |
| Experiment results | `WHITEPAPER.md` |
| Project rules | `agents.md` |
| Tokenizer files | `tokenizers/kazakh-bpe-32k/` |
| Cloud pipeline | `src/slm/cloud/` |
