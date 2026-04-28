"""Upload OmniAudio v2 ASR model to HuggingFace Hub."""
import os
import sys
import json

from huggingface_hub import HfApi, create_repo

REPO = "stukenov/sozkz-core-omniaudio-70m-kk-asr-v1"
CHECKPOINT_DIR = "outputs/omniaudio_v2_scratch_kzcalm_pure_ce/checkpoint-best"
CONFIG_PATH = "omniaudio/configs/v2_scratch_kzcalm_pure_ce.yaml"
TOKENIZER_DIR = "tokenizers/kazakh-gpt2-50k"
README_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "omniaudio_v2_model_card.md")

token = os.environ.get("HF_TOKEN")
if not token:
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        token = open(token_path).read().strip()
if not token:
    print("ERROR: HF_TOKEN not set and no cached token found")
    sys.exit(1)

api = HfApi()

print(f"Creating repo: {REPO}")
create_repo(REPO, token=token, exist_ok=True, repo_type="model")

# Upload model checkpoint
model_pt = os.path.join(CHECKPOINT_DIR, "model.pt")
if not os.path.exists(model_pt):
    print(f"ERROR: {model_pt} not found")
    sys.exit(1)

size_mb = os.path.getsize(model_pt) / 1e6
print(f"Uploading model.pt ({size_mb:.0f} MB)...")
api.upload_file(
    path_or_fileobj=model_pt,
    path_in_repo="model.pt",
    repo_id=REPO,
    token=token,
)

# Upload config
print("Uploading config...")
api.upload_file(
    path_or_fileobj=CONFIG_PATH,
    path_in_repo="config.yaml",
    repo_id=REPO,
    token=token,
)

# Upload base config too
base_config = "omniaudio/configs/v2_base.yaml"
if os.path.exists(base_config):
    api.upload_file(
        path_or_fileobj=base_config,
        path_in_repo="v2_base.yaml",
        repo_id=REPO,
        token=token,
    )

# Upload tokenizer files
print("Uploading tokenizer...")
for fname in os.listdir(TOKENIZER_DIR):
    fpath = os.path.join(TOKENIZER_DIR, fname)
    if os.path.isfile(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"tokenizer/{fname}",
            repo_id=REPO,
            token=token,
        )
        print(f"  {fname}")

# Upload model source code
print("Uploading source code...")
src_dir = "omniaudio/src/omniaudio"
for fname in ["model_v2.py", "data_v2.py", "evaluate_v2.py", "augment.py"]:
    fpath = os.path.join(src_dir, fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"src/{fname}",
            repo_id=REPO,
            token=token,
        )
        print(f"  {fname}")

# Upload results
results = {
    "wer": 21.28,
    "cer": 12.80,
    "samples": 50,
    "val_loss": 2.3547,
    "checkpoint": "checkpoint-best",
    "config": "v2_scratch_kzcalm_pure_ce",
    "training": {
        "ctc_weight": 0.0,
        "learning_rate": 5e-6,
        "label_smoothing": 0.05,
        "epochs": 1,
        "dataset": "kzcalm-tts-kk-v1 (439h)",
        "gpu": "1x NVIDIA RTX 4090",
    }
}
results_path = "/tmp/omniaudio_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
api.upload_file(
    path_or_fileobj=results_path,
    path_in_repo="results.json",
    repo_id=REPO,
    token=token,
)

# Upload README
if os.path.exists(README_PATH):
    api.upload_file(
        path_or_fileobj=README_PATH,
        path_in_repo="README.md",
        repo_id=REPO,
        token=token,
    )
    print("README uploaded")

print(f"\nDone! https://huggingface.co/{REPO}")
