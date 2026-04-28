"""Upload crawled pages to HuggingFace every 10 min, delete after upload.
Uploads ONE chunk per run (MAX_TAR_MB) so each call finishes in minutes, not hours.
Run repeatedly via cron or upload_loop in run_crawler.sh.
"""

import json
import os
import tarfile
from datetime import datetime
from pathlib import Path

CRAWLED_DIR = "crawled_sites"
UPLOAD_LOG = "upload_log.json"
HF_REPO = "stukenov/kaznet-crawl-raw"
MAX_TAR_MB = 300
MIN_FILES = 100  # skip if less than this many files accumulated


def load_log():
    if os.path.exists(UPLOAD_LOG):
        with open(UPLOAD_LOG) as f:
            return json.load(f)
    return {"batches": 0, "total_pages": 0, "total_bytes": 0, "last_upload": None}


def save_log(log):
    with open(UPLOAD_LOG, "w") as f:
        json.dump(log, f, indent=2)


def upload_tar(tar_name):
    from huggingface_hub import HfApi
    try:
        api = HfApi()
        api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=tar_name,
            path_in_repo=f"data/{tar_name}",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        return True
    except Exception as e:
        print(f"Upload FAILED: {e}")
        return False


def collect_files(crawled: Path):
    """Collect all HTML files, oldest first."""
    files = []
    for site_dir in sorted(crawled.iterdir()):
        if not site_dir.is_dir():
            continue
        html_dir = site_dir / "html"
        if not html_dir.exists():
            continue
        for f in sorted(html_dir.glob("*.html")):
            files.append(f)
    return files


def main():
    crawled = Path(CRAWLED_DIR)
    if not crawled.exists():
        print("No crawled data yet")
        return

    files = collect_files(crawled)
    if not files:
        print("No HTML files on disk")
        return

    total_on_disk = len(files)
    print(f"Files on disk: {total_on_disk}")

    if total_on_disk < MIN_FILES:
        print(f"Less than {MIN_FILES} files, skipping (waiting for more to accumulate)")
        return

    log = load_log()
    batch_num = log["batches"] + 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pack ONE chunk up to MAX_TAR_MB
    chunk_files = []
    chunk_size = 0
    for f in files:
        sz = f.stat().st_size
        chunk_files.append(f)
        chunk_size += sz
        if chunk_size >= MAX_TAR_MB * 1024 * 1024:
            break

    tar_name = f"batch_{batch_num:04d}_{ts}.tar.gz"
    print(f"Packing {tar_name}: {len(chunk_files)} files, {chunk_size / 1024 / 1024:.0f} MB...")

    with tarfile.open(tar_name, "w:gz") as tar:
        for cf in chunk_files:
            arcname = str(cf.relative_to(crawled))
            tar.add(str(cf), arcname=arcname)

    tar_mb = os.path.getsize(tar_name) / 1024 / 1024
    print(f"Compressed: {tar_mb:.1f} MB, uploading...")

    if upload_tar(tar_name):
        print(f"OK! Deleting {len(chunk_files)} files...")
        for cf in chunk_files:
            cf.unlink()
        os.remove(tar_name)
        log["batches"] = batch_num
        log["total_pages"] = log.get("total_pages", 0) + len(chunk_files)
        log["total_bytes"] = log.get("total_bytes", 0) + chunk_size
        log["last_upload"] = ts
        save_log(log)
        remaining = total_on_disk - len(chunk_files)
        print(f"Done. Total sent to HF: {log['total_pages']} pages. Remaining on disk: {remaining}")
    else:
        print(f"Upload failed, keeping {tar_name} for retry")


if __name__ == "__main__":
    main()
