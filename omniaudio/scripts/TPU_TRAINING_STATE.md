# OmniAudio CTC TPU Training — Continuation State

## Active Trainings (2026-04-06 ~00:35 local / 2026-04-05 ~19:35 UTC)

| Model | Node | Zone | Screen | Step | Loss | Total Steps | HF Repo | HF Checkpoints |
|-------|------|------|--------|------|------|-------------|---------|----------------|
| 150m | omniaudio-150m-b-node | europe-west4-a | ctc_150m | 900+ | 74.59 | ~62.5K | stukenov/sozkz-core-omniaudio-150m-kk-ctc-v1 | none (restarted) |
| 1b | omniaudio-1b-b-node | europe-west4-a | ctc_1b | 1350+ | 118.57 | ~125K | stukenov/sozkz-core-omniaudio-1b-kk-ctc-v1 | none (restarted) |
| 600m | omniaudio-600m-g-node | europe-west4-a | ctc_600m | 200+ | 174.42 | ~62.5K | stukenov/sozkz-core-omniaudio-600m-kk-ctc-v1 | none (fresh) |
| 50m | omniaudio-50m-g-node | us-east1-d | ctc_50m | compiling | — | ~15.6K | stukenov/sozkz-core-omniaudio-50m-kk-ctc-v1 | none (restarted) |

## Queued Resources
- omniaudio-150m-b (europe-west4-a, v6e-4) — ACTIVE
- omniaudio-1b-b (europe-west4-a, v6e-8) — ACTIVE
- omniaudio-600m-g (europe-west4-a, v6e-8) — ACTIVE
- omniaudio-50m-g (us-east1-d, v6e-4) — ACTIVE

## Script
- `/Users/sakentukenov/slm/omniaudio/scripts/train_ctc_tpu.py` — xmp.spawn multi-chip, streaming, bf16 model, float32 CTC
- Tokenizer: `/tmp/tokenizer_gpt2_50k.tar.gz` (SCP'd to each VM at `~/tokenizers/kazakh-gpt2-50k`)
- **FIXED (2026-04-05):** save_checkpoint now called by ALL xmp processes (xm.save is collective). Previously only master called it → checkpoints silently empty.
- **FIXED (2026-04-05):** Using `> file 2>&1` instead of `| tee` to avoid pipe buffering. Also `PYTHONUNBUFFERED=1` + `python3 -u`.

## Deploy Recipe (for new/recreated VMs)
```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token)
SCRIPT=/Users/sakentukenov/slm/omniaudio/scripts/train_ctc_tpu.py
# SCP
gcloud compute tpus tpu-vm scp $SCRIPT /tmp/tokenizer_gpt2_50k.tar.gz NODE:~/ --zone=ZONE --project=sozkz-trc
# Install
gcloud compute tpus tpu-vm ssh NODE --zone=ZONE --project=sozkz-trc --command="cd ~ && tar xzf tokenizer_gpt2_50k.tar.gz; pip install --quiet 'torch==2.9.0' 'torch_xla[tpu]==2.9.0' huggingface_hub numpy datasets transformers --timeout 300"
# Launch (IMPORTANT: use > file 2>&1, NOT | tee; use PYTHONUNBUFFERED=1 + python3 -u)
gcloud compute tpus tpu-vm ssh NODE --zone=ZONE --project=sozkz-trc --command="screen -dmS ctc_SIZE bash -c 'export PATH=\$HOME/.local/bin:\$PATH; export PJRT_DEVICE=TPU; export TPU_RUNTIME_METRICS_PORTS=; export TPU_STDERR_LOG_LEVEL=0; export PYTHONUNBUFFERED=1; export TOKENIZER_PATH=\$HOME/tokenizers/kazakh-gpt2-50k; export HF_TOKEN=$HF_TOKEN; export STREAM_DATASET=1; python3 -u ~/train_ctc_tpu.py --size SIZE --stage ctc --resume > ~/ctc_SIZE.log 2>&1'; sleep 2; screen -ls"
```

## RULES
- NEVER pkill python3 (kills SSH daemon)
- NEVER sudo reboot (loses SSH keys forever)
- NEVER screen -X quit on a working training (locks TPU device, requires recreate!)
- After crash: /dev/vfio stays locked → must delete+recreate queued resource
- To kill screen: screen -S name -X quit (but ONLY if training already crashed/finished)
- Project: sozkz-trc

## After CTC Done → E2E
Run: `python3 ~/train_ctc_tpu.py --size SIZE --stage e2e`
This loads frozen LLM (with attn_implementation="eager") + trained encoder. Needs larger TPU.

## IMMEDIATE TODO (for next session)
1. **Monitor all 4 trainings** — check loss descent, verify checkpoints save at 5K steps
2. When any model shows "DONE" → launch E2E: `python3 ~/train_ctc_tpu.py --size SIZE --stage e2e`
3. If preempted → delete+recreate queued resource (increment letter)
4. After E2E → evaluate WER on kzcalm test set

## Version incrementing for preempt recovery
When recreating: omniaudio-{size}-{letter} → increment letter
- 150m: a→b (current)
- 1b: a→b (current)
- 600m: e→f→g (current)
- 50m: e→f→g (current)

## Cron prompt for new session
```
прочитай omniaudio/scripts/TPU_TRAINING_STATE.md и продолжай мониторинг TPU тренировок
```
