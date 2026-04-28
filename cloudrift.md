# CloudRift Notes

This file is the working operator note for OmniAudio training on CloudRift.

## Purpose

- Track active CloudRift instances used for `omniaudio`
- Document how to connect, monitor, launch, and stop jobs
- Record where logs, checkpoints, tokens, and configs live
- Capture the practical workflow that was actually used

## Current State

Source of truth for active rented instances:
[omniaudio/scripts/cloudrift_state.json](/Users/sakentukenov/slm/omniaudio/scripts/cloudrift_state.json)

Current entries:

| Key | Instance ID | Host | User | Screen | Log | Status |
|---|---|---|---|---|---|---|
| `150m_morphbpe_long15` | `2a3add2c-340b-11f1-9fdd-b7684ea2946b` | `217.138.104.172` | `riftuser` | — | `/home/riftuser/slm/logs/cloudrift_150m_morphbpe_long15.log` | terminated 2026-04-10 — WER 47.27%/CER 33.65% on FLEURS (853 samples) |
| `300m_morphbpe_long15` | `2b31ef86-340b-11f1-9fdd-d349a63b20c0` | `176.124.69.204` | `riftuser` | — | `/home/riftuser/slm/logs/cloudrift_300m_morphbpe_long15.log` | terminated 2026-04-10 — WER 49.69%/CER 35.02% on FLEURS |
| `600m_morphbpe_long15_multigpu` | `05f1f420-34ae-11f1-a147-d7d8cedd3f90` | `217.138.104.162` | `riftuser` | — | `/home/riftuser/slm/logs/cloudrift_600m_morphbpe_long15_multigpu.log` | terminated 2026-04-10 — WER 98.18%/CER 78.15% on FLEURS, severely undertrained |

## Credentials And Paths

Do not commit secrets into git.

- CloudRift API key path:
  - `~/.config/cloudrift/api_key`
- Hugging Face token path on local machine:
  - `~/.cache/huggingface/token`
- Hugging Face token path on pods:
  - `/home/riftuser/.cache/huggingface/token`
- Main project path on pods:
  - `/home/riftuser/slm`

## SSH Access

All current pods use:

- user: `riftuser`
- SSH options:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@<HOST>
```

Examples:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@217.138.104.172
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@176.124.69.204
```

## What Is Running

### 50M — TERMINATED (2026-04-10)

- Instance `a314da5c-3278-11f1-9694-27aaefc2e31f` was terminated via CloudRift API after training finished.
- Final FLEURS on `checkpoint-best`: WER 98.95%, CER 81.25% (poor result; weights kept on HF for the record).
- Config: `omniaudio/configs/v2_llm50m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml`
- HF: `stukenov/sozkz-core-omniaudio-50m-kk-asr-v1`

### 150M — TERMINATED (2026-04-10)

- Instance `2a3add2c-340b-11f1-9fdd-b7684ea2946b` was terminated after training completed 5 epochs and FLEURS eval confirmed results.
- Final eval on `google/fleurs kk_kz test` (853 samples, full test set) on `checkpoint-best`: **WER 47.27%, CER 33.65%**.
- Best result in the 150M/300M/600M MorphBPE series — outperforms 300M (WER 49.69%) and 600M (WER 98.18%).
- Config: `omniaudio/configs/v2_llm150m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml`
- HF: `stukenov/sozkz-core-omniaudio-150m-kk-asr-v1`

### 300M — TERMINATED (2026-04-10)

- Instance `2b31ef86-340b-11f1-9fdd-d349a63b20c0` was terminated after training was stopped at `step 92800 / ~107K` (~87% of the planned 5-epoch schedule) to free VRAM for evaluation.
- Final eval on `google/fleurs kk_kz test` (200 samples) at `checkpoint-92000`: WER 49.69%, CER 35.02%.
- Underperformed the 150M sibling (WER 43.07%) because `grad_accum=4` gave the 300M roughly one third of the gradient updates per epoch vs. the 150M's `accum=1`.
- Config: `omniaudio/configs/v2_llm300m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml`
- HF: `stukenov/sozkz-core-omniaudio-300m-kk-asr-v1` (canonical checkpoint: `checkpoint-92000`, updated model card uploaded)

### 600M Multi-GPU — TERMINATED (2026-04-10)

- Instance `05f1f420-34ae-11f1-a147-d7d8cedd3f90` was terminated after eval on `checkpoint-40000` showed WER **98.18%**, CER **78.15%** on `FLEURS kk_kz test` (200 samples) — severely undertrained (repetition collapse).
- Root cause: at the time of the eval the model had seen only ~0.5 epochs (vs ~4.5 epochs for the 150M sibling) because of its smaller effective batch and the mid-run restart from `checkpoint-40000-init`. Projected full-schedule WER (~50–65%) was not expected to beat the 150M (43.07%), so training was stopped to save ~$23.
- Config: `omniaudio/configs/v2_llm600m_e2e_morphbpe100k_fleursnorm_long15_multigpu_localllm_cloudrift.yaml`
- HF: `stukenov/sozkz-core-omniaudio-600m-kk-asr-v1` (model card NOT updated — treat as abandoned run)
- Launch pattern was: `torchrun --nproc_per_node=3 -m omniaudio.train_v2 --config ...`

**Historical: 600M was the last multi-GPU experiment.** As of 2026-04-10 the active 150M training is the only OmniAudio v2 run left.

~~Current runtime:~~
  - `python3 -m torch.distributed.run --nproc_per_node=3 --master_port=29601 -m omniaudio.train_v2 --config omniaudio/configs/v2_llm600m_e2e_morphbpe100k_fleursnorm_long15_multigpu_localllm_cloudrift.yaml`

## Useful Monitoring Commands

### Check active `screen`

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@217.138.104.172 'screen -ls'
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@176.124.69.204 'screen -ls'
```

### Tail logs

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@217.138.104.172 'tail -n 50 /home/riftuser/slm/logs/cloudrift_150m_morphbpe_long15.log'
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@176.124.69.204 'tail -n 50 /home/riftuser/slm/logs/cloudrift_300m_morphbpe_long15.log'
```

### Check GPU load

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@217.138.104.172 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@176.124.69.204 'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

## Launch Pattern Used In Practice

The helper script exists here:
[omniaudio/scripts/cloudrift_train.py](/Users/sakentukenov/slm/omniaudio/scripts/cloudrift_train.py)

But the recent `150m` and `300m` MorphBPE runs were launched manually, because that was faster and more controllable than relying on the older orchestration path.

Practical launch flow:

1. Rent a VM via CloudRift REST API
2. SSH into pod as `riftuser`
3. Ensure Python deps are installed
4. Rsync repo to `/home/riftuser/slm`
5. Copy HF token to pod
6. Download required tokenizer and init checkpoint
7. Launch detached train with `screen`

### Generic launch command

```bash
cd /home/riftuser/slm
screen -dmS <SCREEN_NAME> bash -lc '
  PYTHONPATH=omniaudio/src python3 -m omniaudio.train_v2 \
    --config <CONFIG_PATH> \
    2>&1 | tee <LOG_PATH>
'
```

### 150M command

```bash
cd /home/riftuser/slm
screen -dmS train_150m_morphbpe_long15 bash -lc '
  PYTHONPATH=omniaudio/src python3 -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm150m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml \
    2>&1 | tee /home/riftuser/slm/logs/cloudrift_150m_morphbpe_long15.log
'
```

### 300M command

```bash
cd /home/riftuser/slm
screen -dmS train_300m_morphbpe_long15 bash -lc '
  PYTHONPATH=omniaudio/src python3 -m omniaudio.train_v2 \
    --config omniaudio/configs/v2_llm300m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml \
    2>&1 | tee /home/riftuser/slm/logs/cloudrift_300m_morphbpe_long15.log
'
```

## Downloaded Assets On Pods

### Tokenizer

- Path:
  - `/home/riftuser/slm/tokenizers/sozkz-morphbpe-100k-kk-v1`
- HF repo:
  - `stukenov/sozkz-morphbpe-100k-kk-v1`

### Init checkpoints

150M:
- HF repo:
  - `stukenov/sozkz-core-omniaudio-150m-kk-ctc-v1`
- Pod path:
  - `/home/riftuser/slm/outputs/omniaudio_v2_llm150m_ctc/checkpoint-best/model.pt`

300M:
- HF repo:
  - `stukenov/sozkz-core-omniaudio-300m-kk-ctc-v1`
- Pod path:
  - `/home/riftuser/slm/outputs/omniaudio_v2_llm300m_ctc/checkpoint-best/model.pt`

## Evaluation Commands

### Full FLEURS eval

```bash
cd /home/riftuser/slm
PYTHONPATH=omniaudio/src python3 -m omniaudio.evaluate_v2 \
  --config omniaudio/configs/v2_llm50m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml \
  --dataset fleurs \
  --model-path outputs/omniaudio_v2_llm50m_morphbpe100k_fleursnorm_long15/checkpoint-best/model.pt
```

### Chunked 15s eval

```bash
cd /home/riftuser/slm
PYTHONPATH=omniaudio/src python3 -m omniaudio.evaluate_v2 \
  --config omniaudio/configs/v2_llm50m_e2e_morphbpe100k_fleursnorm_chunk15_eval.yaml \
  --dataset fleurs \
  --model-path outputs/omniaudio_v2_llm50m_morphbpe100k_fleursnorm_long15/checkpoint-26000/model.pt
```

## How To Stop Jobs

### Stop only the train process on a pod

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@217.138.104.172 'screen -S train_150m_morphbpe_long15 -X quit'
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@176.124.69.204 'screen -S train_300m_morphbpe_long15 -X quit'
```

### Kill any leftover Python training processes

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@217.138.104.172 \"pkill -f 'omniaudio.train_v2' || true\"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null riftuser@176.124.69.204 \"pkill -f 'omniaudio.train_v2' || true\"
```

## How To Delete Pods

The helper script has terminate support, but direct API deletion is also possible if needed.

Relevant instance IDs:

- `150m_morphbpe_long15`:
  - `2a3add2c-340b-11f1-9fdd-b7684ea2946b`
- `600m_morphbpe_long15_multigpu`:
  - `05f1f420-34ae-11f1-a147-d7d8cedd3f90`

Terminated on 2026-04-10 (weights and model cards already on HF):

- `50m` — `a314da5c-3278-11f1-9694-27aaefc2e31f`
- `300m_morphbpe_long15` — `2b31ef86-340b-11f1-9fdd-d349a63b20c0`

## Known Practical Issues

- `cloudrift_train.py` is not fully up to date with all new experiments and recent manual runs.
- Some old log paths in `cloudrift_state.json` still refer to earlier stages; use the explicit paths in this file.
- `trust_remote_code` warnings appear during dataset loading. In current runs they are warnings plus fallback, not a stop condition.
- Cold start is slow:
  - downloading parquet shards from HF can take a long time
  - generating train split can also take several minutes before first `Step ...`
- `50m` evals only worked correctly when `--dataset fleurs` was passed explicitly; otherwise eval could try to load the training mel dataset and fail because there is no raw `audio` field.

## Inference API (serverless LLM)

CloudRift также предоставляет serverless LLM inference по OpenAI-совместимому API — отдельно от training pods.

### Endpoint and model

- **Base URL:** `https://inference.cloudrift.ai/v1`
- **Auth:** `Authorization: Bearer <api_key>` (same key as for pods, `~/.config/cloudrift/api_key`)
- **Available model:** `Qwen/Qwen3.5-122B-A10B-FP8`
  - 10B active × 12 experts, 122B total MoE, FP8
  - Context: 262,144 tokens
  - **Input:  $0.25 / 1M tokens**
  - **Output: $1.50 / 1M tokens**

**Note on pricing field:** `/v1/models` endpoint returns `input_price: 25.0, output_price: 150.0` — это **центы** за миллион, не доллары. Реальная цена из dashboard: $0.25 / $1.50 per 1M tokens.

### Critical: disable thinking mode

Qwen3.5 по умолчанию работает в reasoning mode — пишет chain-of-thought в поле `reasoning`, `content` заполняется только после thinking. Это делает простые запросы в ~60× дороже и медленнее.

**Решение:** обязательно передавать `chat_template_kwargs: {"enable_thinking": false}` в body запроса.

НЕ работают следующие варианты (проверено):
- `/no_think` маркер в промпте
- `reasoning_effort: "none"`
- `enable_thinking: false` на top-level body

### Working example

```python
import os, json, urllib.request

KEY = open(os.path.expanduser("~/.config/cloudrift/api_key")).read().strip()
body = json.dumps({
    "model": "Qwen/Qwen3.5-122B-A10B-FP8",
    "messages": [{"role": "user", "content": "Қазақстан астанасы қай қала?"}],
    "temperature": 0.7,
    "max_tokens": 500,
    "chat_template_kwargs": {"enable_thinking": False},  # CRITICAL
}).encode()
req = urllib.request.Request(
    "https://inference.cloudrift.ai/v1/chat/completions",
    data=body, method="POST",
    headers={"Content-Type": "application/json",
             "Authorization": f"Bearer {KEY}"},
)
r = json.loads(urllib.request.urlopen(req, timeout=300).read())
print(r["choices"][0]["message"]["content"])
# → "Астана"
```

### Measured performance on Kazakh (non-thinking mode)

Батарея из 12 разнообразных тестов (фактоид, грамматика, перевод KK↔RU, математика, стих, саммари, код, идиома, рассуждение, формальный текст):

| Metric | Value |
|---|---|
| Tests passed | 12/12 |
| Avg latency | 5.5s |
| Prompt tokens | 775 |
| Completion tokens | 4622 |
| Total cost | **$0.007** |

Качество: native-level казахский, знает культурный контекст, пишет код с казахскими идентификаторами, переводит без ошибок. Слабости: логические задачи-головоломки (в non-thinking mode может ошибаться).

## Related Files

- State:
  - [cloudrift_state.json](/Users/sakentukenov/slm/omniaudio/scripts/cloudrift_state.json)
- Script:
  - [cloudrift_train.py](/Users/sakentukenov/slm/omniaudio/scripts/cloudrift_train.py)
- Whitepaper:
  - [omniaudio_v2_whitepaper.md](/Users/sakentukenov/slm/docs/omniaudio_v2_whitepaper.md)
- Active configs:
  - [v2_llm150m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_llm150m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml)
  - [v2_llm300m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_llm300m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml)
  - [v2_llm50m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml](/Users/sakentukenov/slm/omniaudio/configs/v2_llm50m_e2e_morphbpe100k_fleursnorm_long15_cloudrift.yaml)
