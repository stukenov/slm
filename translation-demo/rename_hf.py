#!/usr/bin/env python3
"""One-time: rename remaining files 6-9 on HF (remove -of-00010)."""

import os
import shutil
from huggingface_hub import HfApi

REPO = "saken-tukenov/sozkz-corpus-clean-enkk-fineweb-edu-v1"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/datasets--saken-tukenov--sozkz-corpus-clean-enkk-fineweb-edu-v1")

def main():
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")

    for f in sorted(files):
        if f.startswith("data/train-") and "-of-" in f:
            new_name = f.split("-of-")[0] + ".parquet"
            print(f"  {f} → {new_name}")
            local = api.hf_hub_download(REPO, f, repo_type="dataset")
            api.upload_file(path_or_fileobj=local, path_in_repo=new_name, repo_id=REPO, repo_type="dataset")
            api.delete_file(f, REPO, repo_type="dataset")
            # Clean cache after each file
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
            print(f"    done (cache cleared).")

    print("\nAll files renamed.")

if __name__ == "__main__":
    main()
