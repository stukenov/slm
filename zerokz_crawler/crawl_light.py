"""Lightweight crawler for zero.kz — uses requests instead of browser.
Suitable for low-RAM servers (1GB).
"""

import json
import os
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

URLS_FILE = "urls.json"
HTML_DIR = "html_pages"
PROGRESS_FILE = "crawl_progress.json"
DELAY = 1.0  # seconds between requests (be polite)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZeroKZCrawler/1.0; research)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "kk,ru,en",
}


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    # Add hash suffix for long paths
    if len(path) > 150:
        h = hashlib.md5(path.encode()).hexdigest()[:8]
        path = path[:100] + "_" + h
    return f"{path}.html"


def crawl_url(session: requests.Session, url: str) -> str | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"ERR  {url}: {e}")
        return None


def main():
    with open(URLS_FILE) as f:
        urls = json.load(f)
    print(f"Total URLs: {len(urls)}")

    html_dir = Path(HTML_DIR)
    html_dir.mkdir(exist_ok=True)

    # Load progress
    done = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            done = set(json.load(f))
        print(f"Resuming — {len(done)} already done")

    remaining = [u for u in urls if u not in done]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    session = requests.Session()
    errors = 0

    for i, url in enumerate(remaining, 1):
        html = crawl_url(session, url)
        if html:
            fname = url_to_filename(url)
            (html_dir / fname).write_text(html, encoding="utf-8")
            done.add(url)
            if i % 50 == 0 or i <= 5:
                print(f"[{i}/{len(remaining)}] OK  {url}")
        else:
            errors += 1

        # Save progress every 100
        if i % 100 == 0:
            with open(PROGRESS_FILE, "w") as f:
                json.dump(list(done), f)
            print(f"  Progress saved: {len(done)} done, {errors} errors")

        time.sleep(DELAY)

    # Final save
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(done), f)
    print(f"\nDone! Crawled: {len(done)}, Errors: {errors}")


if __name__ == "__main__":
    main()
