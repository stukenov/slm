"""Parallel crawler orchestrator. Runs N workers as subprocesses."""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

SITES_FILE = "sites_to_crawl.json"
PROGRESS_FILE = "crawl_sites_progress.json"
WORKERS = int(os.environ.get("CRAWL_WORKERS", 6))


def safe_dirname(domain):
    return domain.replace(":", "_").replace("/", "_").replace(".", "_")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(done):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(done), f)


def main():
    with open(SITES_FILE) as f:
        sites = json.load(f)
    print(f"Total sites: {len(sites)}")

    done = load_progress()
    print(f"Already done: {len(done)}")

    remaining = [s for s in sites if s["domain"] not in done]
    random.shuffle(remaining)
    print(f"Remaining: {len(remaining)} (randomized)")
    print(f"Workers: {WORKERS}")
    print()

    venv_python = str(Path(__file__).parent / ".venv" / "bin" / "python")
    worker_script = str(Path(__file__).parent / "crawl_worker.py")

    active = {}  # domain -> subprocess.Popen
    idx = 0

    while idx < len(remaining) or active:
        # Launch workers up to limit
        while len(active) < WORKERS and idx < len(remaining):
            site = remaining[idx]
            idx += 1
            domain = site["domain"]
            url = site["url"]

            print(f"[START] {domain} ({idx}/{len(remaining)})")
            proc = subprocess.Popen(
                [venv_python, worker_script, url, domain],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            active[domain] = proc

        # Check for completed workers
        completed = []
        for domain, proc in active.items():
            ret = proc.poll()
            if ret is not None:
                stdout = proc.stdout.read()
                if stdout.strip():
                    print(stdout.strip())
                if ret != 0:
                    stderr = proc.stderr.read()
                    if stderr.strip():
                        print(f"[ERR] {domain}: {stderr.strip()[:200]}")
                completed.append(domain)

        for domain in completed:
            del active[domain]
            done.add(domain)
            # Count pages
            dirname = safe_dirname(domain)
            html_dir = Path("crawled_sites") / dirname / "html"
            pages = len(list(html_dir.glob("*.html"))) if html_dir.exists() else 0
            print(f"[DONE] {domain}: {pages} pages  ({len(done)}/{len(sites)} total)")

        # Save progress periodically
        if completed:
            save_progress(done)

        if active:
            time.sleep(2)

    save_progress(done)
    print(f"\nAll done! {len(done)}/{len(sites)} sites crawled.")


if __name__ == "__main__":
    main()
