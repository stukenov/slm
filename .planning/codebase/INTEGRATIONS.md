# Integrations

## HuggingFace Hub

| Integration | Details |
|-------------|---------|
| Model upload | `src/slm/publish.py`, `autoresearch/upload_600m_to_hf.py` |
| Dataset hosting | Pre-tokenized datasets on HF Hub |
| Token locations | `~/.cache/huggingface/token`, `.env` (HF_TOKEN), `/root/.HUGGINGFACE_HUB_TOKEN` |
| Org/user | `stukenov/` (models, datasets, tokenizers) |
| Naming standard | SozKZ convention (sozkz-core-, sozkz-vocab-, sozkz-corpus-, sozkz-fix-) |
| Gating | Models 300M+ use gated access (manual approval), MIT license |

## vast.ai Cloud

| Integration | Details |
|-------------|---------|
| Module | `src/slm/cloud/` (6 files) |
| CLI | `python -m slm.cloud {launch,monitor,status,destroy}` |
| Auth | `vastai` CLI + API key at `~/.config/vastai/vast_api_key` |
| Docker image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` |
| Flow | Select GPU → create instance → SSH → scp tarball → pip install → train → upload HF → self-destruct |

## Ansible (Remote Server)

| Playbook | Purpose |
|----------|---------|
| `ansible/deploy.yml` | rsync code + venv + pip install to kaznu |
| `ansible/run_experiment.yml` | Launch training in detached screen session |
| `ansible/fetch_results.yml` | Download results from server |
| Inventory | `ansible/inventory.ini` — kaznu server (164.138.46.36:15126) |

## Datasets (External)

| Dataset | Usage |
|---------|-------|
| `kz-transformers/multidomain-kazakh-dataset` | Primary training corpus (23.6M train samples) |
| `stukenov/sozkz-corpus-dedup-kk-web-v1` | Deduplicated web corpus |
| `stukenov/sozkz-corpus-tokenized-kk-llama50k-v3` | Pre-tokenized training data |
| `stukenov/sozkz-corpus-tokenized-enkk-fineweb-edu-v1` | English-Kazakh parallel data |

## No External APIs

- No REST API consumers (models are self-hosted)
- No database dependencies
- No auth providers beyond HF token
- No webhooks or event systems
