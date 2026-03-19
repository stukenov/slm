# SLM — Small Language Models for Kazakh

Training small language models for Kazakh from scratch: from 14M parameter pilots to 600M production models, with custom tokenizers, MoE architectures, and SFT pipelines.

## Published Models (HuggingFace)

All models follow the **SozKZ** naming standard under [`stukenov/`](https://huggingface.co/stukenov).

| Model | Params | Type | HuggingFace |
|-------|--------|------|-------------|
| Llama 600M base | 587M | Pretrained | [stukenov/sozkz-core-llama-600m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-600m-kk-base-v1) |
| Llama 300M base | ~300M | Pretrained | [stukenov/sozkz-core-llama-300m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-300m-kk-base-v1) |
| Llama 150M base | 151.9M | Pretrained | [stukenov/sozkz-core-llama-150m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-150m-kk-base-v1) |
| Llama 150M instruct | 151.9M | SFT ChatML | [stukenov/sozkz-core-llama-150m-kk-instruct-v2](https://huggingface.co/stukenov/sozkz-core-llama-150m-kk-instruct-v2) |
| Llama 50M base | 50.3M | Pretrained | [stukenov/sozkz-core-llama-50m-kk-base-v1](https://huggingface.co/stukenov/sozkz-core-llama-50m-kk-base-v1) |
| MoE 3B init | ~3B | MoE (shared router) | [stukenov/sozkz-moe-mix-3b-kk-base-v1-init](https://huggingface.co/stukenov/sozkz-moe-mix-3b-kk-base-v1-init) |
| Kazakh BPE 50K | — | Tokenizer | [stukenov/sozkz-vocab-bpe-50k-kk-v3](https://huggingface.co/stukenov/sozkz-vocab-bpe-50k-kk-v3) |

## Dataset

Training corpus: **~9B tokens** of Kazakh text, deduplicated from multiple public sources.

| Dataset | Description |
|---------|-------------|
| [kz-transformers/multidomain-kazakh-dataset](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) | 23.6M samples, multi-domain |
| [stukenov/sozkz-corpus-dedup-kk-web-v1](https://huggingface.co/datasets/stukenov/sozkz-corpus-dedup-kk-web-v1) | Deduplicated web corpus |
| [stukenov/sozkz-corpus-tokenized-kk-llama50k-v3](https://huggingface.co/datasets/stukenov/sozkz-corpus-tokenized-kk-llama50k-v3) | Pre-tokenized (BPE 50K) |

## Quick Start

```bash
# Install
pip install -e .

# Train (example: 50M Llama from scratch)
python -m slm.train --config configs/experiments/exp013_llama_50m_9b.yaml

# Evaluate
python -m slm.evaluate --model_path outputs/<experiment_name> --prompts eval/prompts_kk.txt

# Publish to HuggingFace
python -m slm.publish --model_path outputs/<experiment_name> --repo_name stukenov/<model-name>
```

## Cloud Training (vast.ai)

```bash
# Dry run — check GPU prices
PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud launch \
  --config configs/experiments/<config>.yaml \
  --hf-repo stukenov/<model-name> \
  --max-price 0.50 --num-gpus 1 --disk 60 --dry-run

# Launch training
PYTHONPATH=src .venv-cloud/bin/python -m slm.cloud launch \
  --config configs/experiments/<config>.yaml \
  --hf-repo stukenov/<model-name> \
  --max-price 0.50 --num-gpus 1 --disk 60
```

## Project Structure

```
src/slm/           Core package (train, data, tokenizer, evaluate, publish, cloud)
configs/           YAML experiment configs with base.yaml inheritance
  experiments/     Individual experiment configs (exp001–exp026)
scripts/           Utilities: inference, eval, hub, tokenizer, SFT prep
autoresearch/      Autonomous training scripts (DDP, multi-GPU)
ansible/           Playbooks for remote GPU server deployment
eval/              Evaluation prompts and benchmarks
kz-calm/           TTS experiments (Kazakh speech synthesis)
nano/              Custom architecture experiments
tokenizers/        Trained tokenizer files
results/           Evaluation outputs, inference logs, judge results
docs/              Model cards, papers, planning docs
```

## Experiments

26 experiments tracked in [WHITEPAPER.md](WHITEPAPER.md), including:

- **DAPT** on Pythia-14m/31m (pilots)
- **From-scratch Llama** at 50M, 150M, 300M, 600M scales
- **SFT** with Alpaca and ChatML formats
- **MoE** upcycling with shared router (3B)
- **GEC** (grammatical error correction)
- **Sentiment** fine-tuning
- **TTS** experiments (Mimi/mel-spectrogram)

## Key Design Decisions

- **bf16** training (A10/H100 optimized)
- **Config inheritance**: experiment configs extend `configs/base.yaml`
- **Pre-tokenized datasets** on HF Hub for fast training
- **Automated cloud pipeline**: vast.ai GPU selection, training, HF upload, self-destruct
- **SozKZ naming standard** for all HuggingFace publications

## License

MIT
