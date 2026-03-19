# Experiment Configs

Each YAML is an immutable record of a training run. Configs inherit from `../base.yaml`.

## Naming

- `expNNN_description.yaml` — sequential numbering
- When one experiment number has multiple configs (e.g. exp005a/b/c), they share the same logical experiment but differ in data, infra, or tokenizer

## Experiment Map

### Phase 1: DAPT Baselines (exp001–003)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp001_dapt_pythia14m | Pythia | 14M | DAPT on multidomain |
| exp002_dapt_pythia31m | Pythia | 31M | DAPT on multidomain |
| exp003_custom_tok_14m | Pythia | 14M | DAPT + extended tokenizer (+5K) |

### Phase 2: From-Scratch Training (exp004–006)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp004_scratch_14m | Pythia | 14M | From scratch, Kazakh BPE 32K |
| exp004_scratch_kk | Llama | 50M | First custom Llama, server kaznu |
| exp005_balanced_50m | Llama | 50M | Balanced dataset |
| exp005_cloud_50m | Llama | 50M | Cloud (pretokenized, multi-GPU) |
| exp005_scratch_50m_v3 | Llama | 50M | 9B tokens, GPT2-50K tokenizer |
| exp006_balanced_150m | Llama | 150M | Balanced dataset, 32K vocab |
| exp006_cloud_150m | Llama | 150M | Cloud, Chinchilla-optimal arch |

### Phase 3: Chinchilla Scaling (exp007–010)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp007_chinchilla_8m | GPT-2 | 8M | Scaling law baseline |
| exp007_llama_50m_v2 | Llama | 50M | Clean v2 corpus, GPT2-50K |
| exp008_chinchilla_30m | GPT-2 | 30M | Mid-point scaling |
| exp009_chinchilla_60m | GPT-2 | 60M | Largest GPT-2 scaling point |
| exp010_llama_30m | Llama | 30M | Llama vs GPT-2 architecture ablation |

### Phase 3b: MoE Experiments (exp008b–010b)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp008_moe_upcycle | MoE | 200M | Dense→MoE upcycling test |
| exp009_moe_domain_pretrain | MoE | 330M | Domain curriculum training |
| exp010_moe_instruction_sft | MoE | 160M | Instruction SFT on domain MoE |

### Phase 4: Task-Specific (exp011–012)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp011_gec_finetune | Llama | 50M | GEC fine-tuning |
| exp011_t5_50m | T5 | 50M | Seq2seq span corruption pretraining |
| exp012_gec_morphology | Llama | 50M | GEC + morphological data |

### Phase 5: Definitive Runs (exp013–014)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp013_llama_50m_9b | Llama | 50M | Full 9B corpus, best 50M |
| exp014_llama_150m_9b | Llama | 152M | Full 9B corpus, best 150M (flagship) |

### Phase 6: Instruction Tuning (exp015–016)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp015_sft_llama_150m | Llama | 150M | SFT v1 (synthetic instructions) |
| exp016_sft_chatml_150m | Llama | 150M | SFT v2 (ChatML, 8x RTX 4090) |

### Phase 7: Scale-Up (exp017–018)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp017_moe_shared_router_3b | MoE | 3B | Shared router, 128 experts |
| exp018_llama_500m_200k | Llama | 500M | 200K vocab, 3 corpora combined |

### Phase 8: Chinchilla-Optimal Scale-Up (exp019)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp019_llama_900m_chinchilla | Llama | 897M | Chinchilla-optimal on all 50K data (17.8B tokens) |

### Phase 8b: Cost-Optimal Training (exp020)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp020_llama_300m | Llama | 325M | Best quality/cost on 9B tokens, 8×RTX 4090 |

### Phase 9: SFT & GEC (exp021–022)
| Config | Arch | Size | Approach |
|--------|------|------|----------|
| exp021_sft_alpaca_300m | Llama | 325M | Instruction SFT, 52K Kazakh Alpaca, 4×RTX 4090 |
| exp022_gec_300m | Llama | 325M | GEC tag-based, 200K examples, 1ep. Accuracy 5-31%, FP 78% — needs retraining |

### Utility
| Config | Purpose |
|--------|---------|
| smoke_cloud | Pipeline smoke test (50 steps, 1% data) |
