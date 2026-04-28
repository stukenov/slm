"""Deep autonomous crawler for kaznet sites.
Uses Scrapy to recursively crawl every page of every site.
Each site gets its own folder with all HTML pages saved.
"""

import json
import os
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

SITES_FILE = "sites_to_crawl.json"
OUTPUT_DIR = "crawled_sites"
MAX_PAGES_PER_SITE = 500
CONCURRENT_REQUESTS = 4
DOWNLOAD_DELAY = 0.5


def safe_dirname(domain: str) -> str:
    return domain.replace(":", "_").replace("/", "_").replace(".", "_")


def safe_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    query = parsed.query
    if query:
        path += "_" + hashlib.md5(query.encode()).hexdigest()[:8]
    if len(path) > 150:
        h = hashlib.md5(path.encode()).hexdigest()[:8]
        path = path[:100] + "_" + h
    return path + ".html"


class DeepSiteSpider(CrawlSpider):
    name = "deep_site"

    rules = (
        Rule(
            LinkExtractor(
                deny_extensions=[
                    "pdf", "jpg", "jpeg", "png", "gif", "svg", "webp", "ico",
                    "mp3", "mp4", "avi", "mov", "wmv", "flv", "wav", "ogg",
                    "zip", "rar", "tar", "gz", "7z", "exe", "msi", "dmg",
                    "doc", "docx", "xls", "xlsx", "ppt", "pptx",
                    "css", "js", "json", "xml", "rss", "atom", "woff", "woff2", "ttf", "eot",
                ],
                deny=[
                    r"/wp-admin/", r"/admin/", r"/login", r"/logout",
                    r"/cart", r"/checkout", r"/account",
                    r"\?.*sort=", r"\?.*order=",  # skip sort/filter variants
                ],
            ),
            callback="parse_page",
            follow=True,
        ),
    )

    custom_settings = {
        "CONCURRENT_REQUESTS": CONCURRENT_REQUESTS,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
        "DEPTH_LIMIT": 5,
        "CLOSESPIDER_PAGECOUNT": MAX_PAGES_PER_SITE,
        "ROBOTSTXT_OBEY": True,
        "LOG_LEVEL": "INFO",
        "USER_AGENT": "Mozilla/5.0 (compatible; KaznetResearchBot/1.0; academic research)",
        "DOWNLOAD_TIMEOUT": 20,
        "RETRY_TIMES": 2,
        "HTTPERROR_ALLOWED_CODES": [],
        "COOKIES_ENABLED": False,
        "TELNETCONSOLE_ENABLED": False,
        "MEDIA_ALLOW_REDIRECTS": True,
        # Don't consume too much memory
        "MEMUSAGE_LIMIT_MB": 512,
        "MEMUSAGE_WARNING_MB": 400,
        # DNS cache
        "DNSCACHE_ENABLED": True,
    }

    def __init__(self, site_url=None, site_domain=None, output_dir=None, *args, **kwargs):
        self.site_url = site_url
        self.site_domain = site_domain
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.page_count = 0

        self.start_urls = [site_url]
        self.allowed_domains = [site_domain]
        # Also allow www variant
        if site_domain.startswith("www."):
            self.allowed_domains.append(site_domain[4:])
        else:
            self.allowed_domains.append("www." + site_domain)

        super().__init__(*args, **kwargs)

    def parse_page(self, response):
        # Only save HTML
        content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore")
        if "text/html" not in content_type:
            return

        fname = safe_filename(response.url)
        filepath = self.output_path / fname

        filepath.write_bytes(response.body)
        self.page_count += 1

        if self.page_count % 20 == 0:
            self.logger.info(f"[{self.site_domain}] {self.page_count} pages saved")

    def parse_start_url(self, response):
        return self.parse_page(response)


def load_sites():
    """Load sites list. Can be a simple list of URLs or the extracted JSON."""
    if os.path.exists(SITES_FILE):
        with open(SITES_FILE) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            if isinstance(data[0], str):
                # Simple URL list
                sites = []
                for url in data:
                    parsed = urlparse(url if url.startswith("http") else f"https://{url}")
                    sites.append({"url": f"{parsed.scheme}://{parsed.netloc}/", "domain": parsed.netloc})
                return sites
            elif isinstance(data[0], dict):
                return data
    # Fallback: try external_sites.json
    if os.path.exists("external_sites.json"):
        with open("external_sites.json") as f:
            return json.load(f)
    return []


def main():
    sites = load_sites()
    if not sites:
        print("No sites found! Create sites_to_crawl.json with a list of URLs or domains.")
        print('Example: ["example.com", "another.kz"]')
        return

    print(f"Sites to crawl: {len(sites)}")

    # Load progress
    progress_file = "crawl_sites_progress.json"
    done = set()
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            done = set(json.load(f))
        print(f"Already done: {len(done)}")

    out_base = Path(OUTPUT_DIR)
    out_base.mkdir(exist_ok=True)

    remaining = [s for s in sites if s["domain"] not in done]
    print(f"Remaining: {len(remaining)}")

    # Crawl one site at a time (Scrapy runs its own event loop)
    for i, site in enumerate(remaining, 1):
        domain = site["domain"]
        url = site["url"]
        site_dir = out_base / safe_dirname(domain)

        print(f"\n{'='*60}")
        print(f"[{i}/{len(remaining)}] Crawling {domain}")
        print(f"{'='*60}")

        try:
            process = CrawlerProcess()
            process.crawl(
                DeepSiteSpider,
                site_url=url,
                site_domain=domain,
                output_dir=str(site_dir),
            )
            process.start(stop_after_crawl=True)
            # Note: CrawlerProcess can only be started once per process
            # For multiple sites, we need to use CrawlerRunner or subprocess

            done.add(domain)
            pages = len(list(site_dir.glob("*.html"))) if site_dir.exists() else 0
            print(f"  -> {pages} pages saved")

        except Exception as e:
            print(f"  -> ERROR: {e}")

        # Save progress
        with open(progress_file, "w") as f:
            json.dump(list(done), f)

        # CrawlerProcess can only start once — break and let wrapper restart
        break

    print(f"\nProgress: {len(done)}/{len(sites)} sites")
    if len(done) < len(sites):
        print("Run again to continue with next site (Scrapy limitation).")


if __name__ == "__main__":
    main()
