#!/bin/bash
# Parameter Golf v4 — Full pipeline: 3 seeds + Telegram + HF upload + self-destroy
set -uo pipefail

# === HARDCODED TOKENS ===
TG_TOKEN="${TG_TOKEN:?Set TG_TOKEN env var}"
TG_CHAT_ID="47474471"
RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY env var}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
POD_ID="vlfn32qtzg0ht3"
HF_REPO="stukenov/parameter-golf-v4-logs"

tg() {
    python3 -c "
import urllib.request, urllib.parse
data = urllib.parse.urlencode({'chat_id': '$TG_CHAT_ID', 'text': '''$1'''}).encode()
try: urllib.request.urlopen(urllib.request.Request('https://api.telegram.org/bot$TG_TOKEN/sendMessage', data))
except: pass
" 2>/dev/null
}

tg "Parameter Golf v4 started on 8xH100 SXM
Pod: $POD_ID
Features: XSA-11, VRL, GA, CROWN-Q, DepthRecur(4,5), HedgeMixer
Seeds: 1337, 42, 2025"

cd /workspace/parameter-golf

# === Training env ===
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 ITERATIONS=20000
export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=500 WARMDOWN_ITERS=3500
export XSA_LAST_N=11 GATED_ATTENTION=1 VALUE_RESIDUAL=1
export CROWNQ_LAMBDA=0.01 CROWNQ_WARMDOWN_ONLY=1 RECUR_LAYERS="4,5"
export USE_MIXER=1 MIXER_ETA=0.1 LATE_QAT_THRESHOLD=0.15 SWA_ENABLED=1
export TTT_ENABLED=1 TTT_EPOCHS=1 TTT_FREEZE_BLOCKS=0 TTT_CHUNK_TOKENS=32768
export TTT_LR=0.002 TTT_MOMENTUM=0.9
export HF_TOKEN="$HF_TOKEN"

ALL_OK=true
RESULTS=""

for S in 1337 42 2025; do
    LOG="/workspace/seed${S}.log"
    tg "Seed $S starting..."

    SEED=$S RUN_ID=v4_seed$S \
    torchrun --standalone --nproc_per_node=8 train_gpt.py > "$LOG" 2>&1
    EXIT_CODE=$?

    cp final_model.int6.ptz "/workspace/final_seed${S}.ptz" 2>/dev/null

    if [ $EXIT_CODE -ne 0 ]; then
        ERRTAIL=$(tail -5 "$LOG" 2>/dev/null)
        tg "FAIL Seed $S crashed (exit=$EXIT_CODE)
$ERRTAIL"
        ALL_OK=false
        break
    fi

    # Extract results
    BPB_TTT=$(grep "legal_ttt_exact" "$LOG" 2>/dev/null | tail -1 | grep -o "val_bpb:[0-9.]*" | cut -d: -f2)
    BPB_SLIDE=$(grep "final_int8_zlib_roundtrip_exact" "$LOG" 2>/dev/null | tail -1 | grep -o "val_bpb:[0-9.]*" | cut -d: -f2)
    ARTIFACT=$(grep "Total submission" "$LOG" 2>/dev/null | tail -1)

    RESULTS="${RESULTS}
Seed $S: TTT=$BPB_TTT slide=$BPB_SLIDE"

    tg "Seed $S done
BPB (TTT+Mixer): $BPB_TTT
BPB (sliding s64): $BPB_SLIDE
$ARTIFACT"
done

# === Upload to HuggingFace ===
if [ "$ALL_OK" = true ]; then
    tg "Uploading to HuggingFace..."
    pip install huggingface-hub -q 2>/dev/null

    python3 << 'PYEOF'
import os
from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
repo_id = os.environ.get("HF_REPO", "stukenov/parameter-golf-v4-logs")
api = HfApi(token=token)
create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False, token=token)

for seed in [1337, 42, 2025]:
    for path, name in [
        (f"/workspace/seed{seed}.log", f"seed{seed}.log"),
        (f"/workspace/final_seed{seed}.ptz", f"final_seed{seed}.ptz"),
    ]:
        if os.path.exists(path):
            api.upload_file(path_or_fileobj=path, path_in_repo=name,
                          repo_id=repo_id, repo_type="dataset", token=token)
            print(f"Uploaded {name}")

api.upload_file(path_or_fileobj="/workspace/parameter-golf/train_gpt.py",
               path_in_repo="train_gpt_v4.py", repo_id=repo_id,
               repo_type="dataset", token=token)
print("All uploaded")
PYEOF
    UPLOAD_OK=$?

    if [ $UPLOAD_OK -eq 0 ]; then
        tg "ALL DONE - uploaded to HF: $HF_REPO
$RESULTS

Destroying pod..."

        # Self-destroy
        pip install runpod -q 2>/dev/null
        python3 -c "
import runpod
runpod.api_key='$RUNPOD_API_KEY'
runpod.terminate_pod('$POD_ID')
print('Pod terminated')
"
    else
        tg "Upload failed! Pod kept alive for debugging."
    fi
else
    tg "Pipeline failed. Pod kept alive.
$RESULTS"
fi
