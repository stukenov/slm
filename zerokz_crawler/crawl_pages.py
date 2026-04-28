"""Crawl pages from urls.json using crawl4ai and save raw HTML."""

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

URLS_FILE = "urls.json"
HTML_DIR = "html_pages"
PROGRESS_FILE = "crawl_progress.json"
CONCURRENCY = 5  # parallel browser tabs


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    return f"{parsed.netloc}_{path}.html"


async def crawl_batch(crawler, urls: list[str], html_dir: Path, done: set):
    config = CrawlerRunConfig(
        wait_until="domcontentloaded",
        page_timeout=30000,
    )

    for url in urls:
        if url in done:
            continue
        fname = url_to_filename(url)
        out_path = html_dir / fname

        try:
            result = await crawler.arun(url=url, config=config)
            if result.success and result.html:
                out_path.write_text(result.html, encoding="utf-8")
                done.add(url)
                print(f"OK  {url} -> {fname} ({len(result.html)} bytes)")
            else:
                print(f"FAIL {url}: {result.error_message}")
        except Exception as e:
            print(f"ERR  {url}: {e}")

    return done


async def main():
    with open(URLS_FILE) as f:
        urls = json.load(f)
    print(f"Total URLs to crawl: {len(urls)}")

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

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Process in batches
        batch_size = 50
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i : i + batch_size]
            print(f"\n--- Batch {i // batch_size + 1} ({len(batch)} URLs) ---")
            done = await crawl_batch(crawler, batch, html_dir, done)

            # Save progress after each batch
            with open(PROGRESS_FILE, "w") as f:
                json.dump(list(done), f)
            print(f"Progress: {len(done)}/{len(urls)}")

    print(f"\nDone! Crawled {len(done)} pages total.")


if __name__ == "__main__":
    asyncio.run(main())
