"""Crawl external sites from zero.kz catalog.
For each site: fetch homepage + follow internal links (up to MAX_PAGES_PER_SITE).
"""

import json
import os
import re
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

SITES_FILE = "external_sites.json"
OUTPUT_DIR = "crawled_sites"
PROGRESS_FILE = "crawl_external_progress.json"
MAX_PAGES_PER_SITE = 50
DELAY = 1.0
TIMEOUT = 20

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; KaznetCrawler/1.0; research)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "kk,ru,en",
}

SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".zip", ".rar", ".exe",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".css", ".js", ".json", ".xml", ".rss",
}


def safe_dirname(domain: str) -> str:
    return domain.replace(":", "_").replace("/", "_")


def safe_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    if len(path) > 120:
        h = hashlib.md5(path.encode()).hexdigest()[:8]
        path = path[:80] + "_" + h
    return path + ".html"


def should_skip_url(url: str) -> bool:
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    return ext in SKIP_EXTENSIONS


def extract_internal_links(html: str, base_url: str, domain: str) -> list[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        full = urljoin(base_url, href)
        parsed = urlparse(full)

        # Only same domain
        if parsed.netloc.lower().replace("www.", "") != domain.replace("www.", ""):
            continue

        if should_skip_url(full):
            continue

        # Normalize
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if not clean.endswith("/") and "." not in parsed.path.split("/")[-1]:
            clean += "/"
        links.add(clean)

    return list(links)


def crawl_site(session: requests.Session, site_url: str, domain: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    visited = set()
    queue = [site_url]
    saved = 0

    while queue and len(visited) < MAX_PAGES_PER_SITE:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue

            html = resp.text
            fname = safe_filename(url)
            (out_dir / fname).write_text(html, encoding="utf-8")
            saved += 1

            # Extract more links
            new_links = extract_internal_links(html, url, domain)
            for link in new_links:
                if link not in visited:
                    queue.append(link)

            time.sleep(DELAY)

        except Exception:
            continue

    return saved


def main():
    with open(SITES_FILE) as f:
        sites = json.load(f)
    print(f"Total sites to crawl: {len(sites)}")

    out_base = Path(OUTPUT_DIR)
    out_base.mkdir(exist_ok=True)

    # Load progress
    done = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            done = set(json.load(f))
        print(f"Resuming — {len(done)} sites already done")

    remaining = [s for s in sites if s["domain"] not in done]
    print(f"Remaining: {len(remaining)}")

    session = requests.Session()
    total_pages = 0

    for i, site in enumerate(remaining, 1):
        domain = site["domain"]
        url = site["url"]
        site_dir = out_base / safe_dirname(domain)

        print(f"[{i}/{len(remaining)}] {domain}", end=" ", flush=True)
        pages = crawl_site(session, url, domain, site_dir)
        total_pages += pages
        done.add(domain)
        print(f"→ {pages} pages")

        # Save progress every 10 sites
        if i % 10 == 0:
            with open(PROGRESS_FILE, "w") as f:
                json.dump(list(done), f)

    # Final save
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(done), f)
    print(f"\nDone! {len(done)} sites, {total_pages} pages total")


if __name__ == "__main__":
    main()
