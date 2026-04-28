"""Upload best E2E checkpoint to HuggingFace."""
import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = "stukenov/sozkz-omniaudio-v2-scratch-kk-v1"

# Create repo if needed
api.create_repo(repo_id, exist_ok=True, private=True)

# Upload best E2E checkpoint
ckpt_path = "outputs/omniaudio_v2_e2e_from_ctc/checkpoint-best/model.pt"
if os.path.exists(ckpt_path):
    api.upload_file(path_or_fileobj=ckpt_path, path_in_repo="model.pt", repo_id=repo_id)
    print(f"Uploaded {ckpt_path} to {repo_id}")

# Also upload CTC pretrain best
ctc_path = "outputs/omniaudio_v2_ctc_cloud/checkpoint-best/model.pt"
if os.path.exists(ctc_path):
    api.upload_file(path_or_fileobj=ctc_path, path_in_repo="ctc_pretrain_best.pt", repo_id=repo_id)
    print(f"Uploaded CTC checkpoint to {repo_id}")

# Upload training logs
for log in ["logs/ctc_cloud.log", "logs/e2e_cloud.log"]:
    if os.path.exists(log):
        api.upload_file(path_or_fileobj=log, path_in_repo=os.path.basename(log), repo_id=repo_id)

print("Done!")
