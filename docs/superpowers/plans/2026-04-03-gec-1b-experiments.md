# GEC 1B Experiment Series — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `stukenov/sozkz-core-llama-1b-kk-base-v1` for Kazakh grammar error correction with maximum quality, running a systematic experiment series (LoRA variants + full fine-tune) on 4×H100 RunPod pod.

**Architecture:** Simple two-line format (`input\ntarget<eos>`, loss masked on input line). Wave-based parallel experiments — 4 experiments per wave on 4 GPUs. Eval after each wave to pick best config. Final full fine-tune with winning hyperparameters.

**Tech Stack:** PyTorch, transformers, peft (LoRA), RunPod API, HuggingFace Hub

---

## File Structure

| File | Responsibility |
|---|---|
| `autoresearch/exp030_launch_pod.py` | Launch 4×H100 RunPod pod, save connection info |
| `autoresearch/exp030_deploy.sh` | Deploy code to pod, set up environment |
| `autoresearch/exp030_train.py` | Single training script — accepts all config via env vars (LoRA/full, rank, LR, clean ratio, etc.) |
| `autoresearch/exp030_eval.py` | Eval script — test on held-out set, output metrics table |
| `autoresearch/exp030_wave1.sh` | Launch Wave 1: 4 parallel experiments (one per GPU) |
| `autoresearch/exp030_wave2.sh` | Launch Wave 2: 4 more experiments based on Wave 1 results |
| `autoresearch/exp030_final.sh` | Launch final full FT on all 4 GPUs with DDP |
| `autoresearch/exp030_eval_all.sh` | Eval all checkpoints, produce comparison table |

## Experiment Matrix

### Wave 1 (parallel on 4 GPUs)

| GPU | ID | Method | Rank | LR | Clean% | Epochs | Target Modules |
|---|---|---|---|---|---|---|---|
| 0 | 030a | LoRA | 16 | 2e-4 | 80% | 3 | q_proj,v_proj |
| 1 | 030b | LoRA | 64 | 2e-4 | 80% | 3 | q_proj,v_proj |
| 2 | 030c | LoRA | 16 | 2e-4 | 50% | 3 | q_proj,v_proj |
| 3 | 030d | Full FT | — | 1e-5 | 80% | 1 | all |

### Wave 2 (based on Wave 1 winner)

| GPU | ID | What changes | Details |
|---|---|---|---|
| 0 | 030e | All linear modules | q,k,v,o,gate,up,down_proj |
| 1 | 030f | 5 epochs | More training |
| 2 | 030g | 90% clean | Conservative correction |
| 3 | 030h | Full FT lr=5e-5 | Higher LR full |

### Wave 3 (final)

| GPUs | ID | Method | Details |
|---|---|---|---|
| 0-3 DDP | 030-final | Full FT | Best hyperparams from Wave 1+2, 3 epochs |

## Eval Metrics

- **Exact Match (EM)**: output == target
- **CER**: Character Error Rate
- **Word F0.5**: Precision-weighted word-level metric (standard GEC)
- **Identity Preservation (ID%)**: Correct inputs left unchanged
- Test set: 500 held-out examples from `stukenov/sozkz-corpus-synthetic-kk-gec-v1`

---

### Task 1: Create the training script

**Files:**
- Create: `autoresearch/exp030_train.py`

- [ ] **Step 1: Write `exp030_train.py`**

All-in-one training script controlled by environment variables:

```
ENV VARS:
  EXP_ID          — experiment name (e.g. "030a")
  BASE_MODEL      — HF model id
  METHOD          — "lora" or "full"
  LORA_RANK       — 16, 32, 64
  LORA_MODULES    — "qv" or "all"
  LR              — learning rate
  CLEAN_RATIO     — 0.5, 0.8, 0.9
  EPOCHS          — 1, 3, 5
  CUDA_DEVICE     — which GPU (for single-GPU runs)
  HF_TOKEN        — HuggingFace token
  OUTPUT_DIR      — where to save checkpoint
```

Key design:
- Format: `{input}\n{target}<eos>` — loss masked before `\n`
- Data: loads from `stukenov/sozkz-corpus-synthetic-kk-gec-v1`, filters word_edit_distance <= 2, adds clean examples per CLEAN_RATIO
- LoRA: uses `peft` library, applies to specified modules
- Full FT: standard Trainer with bf16
- Saves checkpoint + `results.json` with config and final train loss

- [ ] **Step 2: Verify script parses without errors locally**

Run: `cd /Users/sakentukenov/slm && python3 -c "import ast; ast.parse(open('autoresearch/exp030_train.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add autoresearch/exp030_train.py
git commit -m "feat(exp030): add GEC 1B training script with LoRA/full FT support"
```

---

### Task 2: Create the eval script

**Files:**
- Create: `autoresearch/exp030_eval.py`

- [ ] **Step 1: Write `exp030_eval.py`**

Eval script that:
- Loads a model (supports both LoRA adapter and full FT checkpoints)
- Loads 500 test examples from the GEC dataset
- Generates corrections using the simple `{input}\n` prompt format
- Computes: Exact Match, CER, Word F0.5, Identity Preservation
- Outputs a JSON results file and prints a summary table
- Accepts `--model_dir` and `--exp_id` args

- [ ] **Step 2: Verify script parses**

Run: `python3 -c "import ast; ast.parse(open('autoresearch/exp030_eval.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add autoresearch/exp030_eval.py
git commit -m "feat(exp030): add GEC eval script for two-line format"
```

---

### Task 3: Create the pod launcher

**Files:**
- Create: `autoresearch/exp030_launch_pod.py`

- [ ] **Step 1: Write `exp030_launch_pod.py`**

Based on existing `autoresearch/launch_1b_pod.py` pattern:
- Launch 4×H100 80GB on RunPod (try SPOT community first, then on-demand)
- Pod name: `exp030-gec-1b`
- Docker: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Container disk: 200GB
- Save connection info to `/tmp/exp030_pod.json`
- Print SSH command when ready

- [ ] **Step 2: Commit**

```bash
git add autoresearch/exp030_launch_pod.py
git commit -m "feat(exp030): add RunPod pod launcher for 4xH100"
```

---

### Task 4: Create deploy script

**Files:**
- Create: `autoresearch/exp030_deploy.sh`

- [ ] **Step 1: Write `exp030_deploy.sh`**

Reads `/tmp/exp030_pod.json`, then:
1. Wait for SSH ready
2. SCP `autoresearch/exp030_train.py`, `autoresearch/exp030_eval.py`, wave scripts
3. Install deps: `pip install torch transformers datasets huggingface-hub safetensors accelerate peft`
4. Set HF token
5. Verify: `python3 -c "import peft; print('peft OK')"`
6. Verify: `python3 -c "from huggingface_hub import whoami; print(whoami())"`
7. Print "Deploy complete"

- [ ] **Step 2: Commit**

```bash
git add autoresearch/exp030_deploy.sh
git commit -m "feat(exp030): add deploy script for pod setup"
```

---

### Task 5: Create Wave 1 script

**Files:**
- Create: `autoresearch/exp030_wave1.sh`

- [ ] **Step 1: Write `exp030_wave1.sh`**

Launches 4 experiments in parallel (one per GPU):

```bash
# GPU 0: 030a — LoRA r=16, baseline
CUDA_VISIBLE_DEVICES=0 EXP_ID=030a METHOD=lora LORA_RANK=16 LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=0.8 EPOCHS=3 python3 exp030_train.py &

# GPU 1: 030b — LoRA r=64
CUDA_VISIBLE_DEVICES=1 EXP_ID=030b METHOD=lora LORA_RANK=64 LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=0.8 EPOCHS=3 python3 exp030_train.py &

# GPU 2: 030c — LoRA r=16, clean=50%
CUDA_VISIBLE_DEVICES=2 EXP_ID=030c METHOD=lora LORA_RANK=16 LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=0.5 EPOCHS=3 python3 exp030_train.py &

# GPU 3: 030d — Full FT, 1 epoch
CUDA_VISIBLE_DEVICES=3 EXP_ID=030d METHOD=full LR=1e-5 CLEAN_RATIO=0.8 \
  EPOCHS=1 python3 exp030_train.py &

wait
```

Sends Telegram notification when all 4 finish.

- [ ] **Step 2: Commit**

```bash
git add autoresearch/exp030_wave1.sh
git commit -m "feat(exp030): add Wave 1 parallel launch script"
```

---

### Task 6: Create Wave 2 script

**Files:**
- Create: `autoresearch/exp030_wave2.sh`

- [ ] **Step 1: Write `exp030_wave2.sh`**

Same pattern as Wave 1 but with experiments 030e-030h. The specific configs will be adjusted after Wave 1 eval, but the template uses the planned defaults:

```bash
# GPU 0: 030e — All linear LoRA modules
CUDA_VISIBLE_DEVICES=0 EXP_ID=030e METHOD=lora LORA_RANK=BEST_RANK LORA_MODULES=all \
  LR=2e-4 CLEAN_RATIO=BEST_CLEAN EPOCHS=3 python3 exp030_train.py &

# GPU 1: 030f — 5 epochs
CUDA_VISIBLE_DEVICES=1 EXP_ID=030f METHOD=lora LORA_RANK=BEST_RANK LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=BEST_CLEAN EPOCHS=5 python3 exp030_train.py &

# GPU 2: 030g — 90% clean
CUDA_VISIBLE_DEVICES=2 EXP_ID=030g METHOD=lora LORA_RANK=BEST_RANK LORA_MODULES=qv \
  LR=2e-4 CLEAN_RATIO=0.9 EPOCHS=3 python3 exp030_train.py &

# GPU 3: 030h — Full FT lr=5e-5
CUDA_VISIBLE_DEVICES=3 EXP_ID=030h METHOD=full LR=5e-5 CLEAN_RATIO=BEST_CLEAN \
  EPOCHS=3 python3 exp030_train.py &

wait
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/exp030_wave2.sh
git commit -m "feat(exp030): add Wave 2 parallel launch script"
```

---

### Task 7: Create final full FT script

**Files:**
- Create: `autoresearch/exp030_final.sh`

- [ ] **Step 1: Write `exp030_final.sh`**

Full fine-tune on all 4 GPUs with torchrun DDP:

```bash
EXP_ID=030final METHOD=full LR=BEST_LR CLEAN_RATIO=BEST_CLEAN EPOCHS=3 \
  torchrun --nproc_per_node=4 exp030_train.py
```

After training: upload to `stukenov/sozkz-core-llama-1b-kk-gec-v1` on HuggingFace.

- [ ] **Step 2: Commit**

```bash
git add autoresearch/exp030_final.sh
git commit -m "feat(exp030): add final DDP full fine-tune script"
```

---

### Task 8: Create eval-all script

**Files:**
- Create: `autoresearch/exp030_eval_all.sh`

- [ ] **Step 1: Write `exp030_eval_all.sh`**

Loops through all experiment output dirs, runs `exp030_eval.py` on each, then prints a comparison table:

```
| Exp  | Method | Rank | Clean | EM%  | CER   | F0.5  | ID%  |
|------|--------|------|-------|------|-------|-------|------|
| 030a | LoRA   | 16   | 80%   | 45.2 | 0.032 | 0.567 | 92.1 |
| 030b | LoRA   | 64   | 80%   | 47.8 | 0.028 | 0.601 | 91.5 |
| ...  |        |      |       |      |       |       |      |
```

Sends table via Telegram.

- [ ] **Step 2: Commit**

```bash
git add autoresearch/exp030_eval_all.sh
git commit -m "feat(exp030): add eval-all comparison script"
```

---

### Task 9: Launch pod and deploy

- [ ] **Step 1: Run pod launcher**

```bash
cd /Users/sakentukenov/slm
.venv-runpod/bin/python autoresearch/exp030_launch_pod.py
```

Wait for SSH ready. Verify `/tmp/exp030_pod.json` created.

- [ ] **Step 2: Deploy code to pod**

```bash
bash autoresearch/exp030_deploy.sh
```

Verify output: "peft OK", HF auth confirmed, "Deploy complete".

- [ ] **Step 3: Smoke test on pod**

SSH in and run a 10-step micro train:

```bash
CUDA_VISIBLE_DEVICES=0 EXP_ID=smoke METHOD=lora LORA_RANK=8 LR=2e-4 \
  CLEAN_RATIO=0.8 EPOCHS=1 MAX_STEPS=10 python3 exp030_train.py
```

Verify: loss prints, checkpoint saves, no errors.

---

### Task 10: Run Wave 1

- [ ] **Step 1: Start Wave 1 in screen**

```bash
SSH_CMD="ssh root@IP -p PORT -o StrictHostKeyChecking=no"
$SSH_CMD "cd /root && screen -dmS wave1 bash exp030_wave1.sh"
```

- [ ] **Step 2: Monitor until complete**

Set up cron to check every 5 minutes:
```bash
$SSH_CMD "tail -1 /root/exp030_*/train.log"
```

- [ ] **Step 3: Eval Wave 1**

```bash
$SSH_CMD "cd /root && bash exp030_eval_all.sh"
```

Record results. Pick best rank, best clean ratio.

---

### Task 11: Run Wave 2

- [ ] **Step 1: Update Wave 2 configs with Wave 1 winners**

Edit `exp030_wave2.sh` on pod — replace BEST_RANK, BEST_CLEAN with actual values.

- [ ] **Step 2: Start Wave 2 in screen**

```bash
$SSH_CMD "cd /root && screen -dmS wave2 bash exp030_wave2.sh"
```

- [ ] **Step 3: Monitor and eval Wave 2**

Same monitoring pattern. Eval all 8 experiments (030a-030h).

---

### Task 12: Run Final Full Fine-Tune

- [ ] **Step 1: Update final script with best config**

Edit `exp030_final.sh` with winning LR, clean ratio, epochs.

- [ ] **Step 2: Run final training on 4 GPUs**

```bash
$SSH_CMD "cd /root && screen -dmS final bash exp030_final.sh"
```

- [ ] **Step 3: Verify and upload**

Eval final model. Upload to `stukenov/sozkz-core-llama-1b-kk-gec-v1`.
Verify model loads from HF and produces sane output.

- [ ] **Step 4: Destroy pod (ONLY after HF verification)**

Confirm model works from HF hub. Then destroy pod.

---

### Task 13: Record results

- [ ] **Step 1: Update WHITEPAPER.md**

Add exp030 results with full comparison table.

- [ ] **Step 2: Commit results**

```bash
git add WHITEPAPER.md autoresearch/
git commit -m "feat(exp030): GEC 1B experiment results — LoRA sweep + full FT"
```
