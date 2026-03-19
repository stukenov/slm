#!/usr/bin/env python3
"""
Incrementally upload translation checkpoints to HuggingFace.
Watches for new checkpoint files and uploads them as dataset shards.
"""

import glob
import os
import time

from huggingface_hub import HfApi, create_repo

REPO_ID = "saken-tukenov/sozkz-fineweb-edu-en-kk-1m"
WATCH_DIR = "/root/slm/translation-demo"
PATTERN = "fineweb_1m_kk_v2_gpu*_ckpt_*.parquet"


def main():
    api = HfApi()

    # Create repo if needed
    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"Repo ready: {REPO_ID}")
    except Exception as e:
        print(f"Repo creation: {e}")

    uploaded = set()

    while True:
        files = sorted(glob.glob(os.path.join(WATCH_DIR, PATTERN)))

        # Also check for final merged file
        final = os.path.join(WATCH_DIR, "fineweb_1m_kk_v2.parquet")
        if os.path.exists(final) and final not in uploaded:
            print(f"Final merged file found! Uploading {final}...")
            api.upload_file(
                path_or_fileobj=final,
                path_in_repo="data/train.parquet",
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            uploaded.add(final)
            print(f"Uploaded final file. Done!")
            break

        for f in files:
            if f in uploaded:
                continue

            # Check file is not being written (wait for stable size)
            size1 = os.path.getsize(f)
            time.sleep(5)
            size2 = os.path.getsize(f)
            if size1 != size2:
                print(f"  {f} still being written, skipping...")
                continue

            basename = os.path.basename(f)
            print(f"Uploading {basename} ({size2 / 1e6:.0f}MB)...")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"data/checkpoints/{basename}",
                repo_id=REPO_ID,
                repo_type="dataset",
            )
            uploaded.add(f)
            print(f"  Uploaded {basename}")

        # Check for final gpu parts (translation done)
        gpu0_final = os.path.join(WATCH_DIR, "fineweb_1m_kk_v2_gpu0.parquet")
        gpu1_final = os.path.join(WATCH_DIR, "fineweb_1m_kk_v2_gpu1.parquet")
        if os.path.exists(gpu0_final) and os.path.exists(gpu1_final):
            for gf in [gpu0_final, gpu1_final]:
                if gf not in uploaded:
                    basename = os.path.basename(gf)
                    print(f"Uploading final part {basename}...")
                    api.upload_file(
                        path_or_fileobj=gf,
                        path_in_repo=f"data/{basename}",
                        repo_id=REPO_ID,
                        repo_type="dataset",
                    )
                    uploaded.add(gf)
            print("Both GPU parts uploaded. Waiting for merged file...")

        print(f"  [{len(uploaded)} files uploaded, watching for more...]")
        time.sleep(60)


if __name__ == "__main__":
    main()
