"""Single-site crawler worker. Run as subprocess for parallelism."""

import json
import sys
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse, urljoin

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

MAX_PAGES = 10_000_000
DOWNLOAD_DELAY = 0.1


def safe_dirname(domain):
    return domain.replace(":", "_").replace("/", "_").replace(".", "_")


def safe_filename(url):
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    query = parsed.query
    if query:
        path += "_" + hashlib.md5(query.encode()).hexdigest()[:8]
    if len(path) > 150:
        h = hashlib.md5(path.encode()).hexdigest()[:8]
        path = path[:100] + "_" + h
    return path


class SiteSpider(CrawlSpider):
    name = "site_worker"

    rules = (
        Rule(
            LinkExtractor(
                deny_extensions=[
                    "pdf", "jpg", "jpeg", "png", "gif", "svg", "webp", "ico",
                    "mp3", "mp4", "avi", "mov", "wmv", "flv", "wav", "ogg",
                    "zip", "rar", "tar", "gz", "7z", "exe", "msi", "dmg",
                    "doc", "docx", "xls", "xlsx", "ppt", "pptx",
                    "css", "js", "json", "xml", "rss", "atom",
                    "woff", "woff2", "ttf", "eot",
                ],
                deny=[r"/wp-admin/", r"/admin/", r"/login", r"/logout",
                      r"/cart", r"/checkout", r"/account"],
            ),
            callback="parse_page",
            follow=True,
        ),
    )

    custom_settings = {
        "CONCURRENT_REQUESTS": 8,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
        "DEPTH_LIMIT": 0,  # unlimited
        "CLOSESPIDER_PAGECOUNT": MAX_PAGES,
        "ROBOTSTXT_OBEY": True,
        "LOG_LEVEL": "WARNING",
        "USER_AGENT": "Mozilla/5.0 (compatible; KaznetResearchBot/1.0; academic research)",
        "DOWNLOAD_TIMEOUT": 20,
        "RETRY_TIMES": 2,
        "COOKIES_ENABLED": False,
        "TELNETCONSOLE_ENABLED": False,
        "MEMUSAGE_LIMIT_MB": 200,
        "MEMUSAGE_WARNING_MB": 150,
        "DNSCACHE_ENABLED": True,
    }

    def __init__(self, site_url, site_domain, html_dir, seen_file=None, md_dir=None, *args, **kwargs):
        self.site_url = site_url
        self.site_domain = site_domain
        self.html_path = Path(html_dir)
        self.html_path.mkdir(parents=True, exist_ok=True)
        self.page_count = 0

        # Load already-seen URLs (shared across all workers and runs)
        self.seen_file = Path(seen_file) if seen_file else Path("seen_urls.txt")
        self.seen_urls = set()
        if self.seen_file.exists():
            self.seen_urls = set(self.seen_file.read_text().splitlines())
        self._seen_buffer = []

        self.start_urls = [site_url]
        self.allowed_domains = [site_domain]
        if site_domain.startswith("www."):
            self.allowed_domains.append(site_domain[4:])
        else:
            self.allowed_domains.append("www." + site_domain)

        super().__init__(*args, **kwargs)

    def _flush_seen(self):
        if self._seen_buffer:
            with open(self.seen_file, "a") as f:
                f.write("\n".join(self._seen_buffer) + "\n")
            self._seen_buffer = []

    def parse_page(self, response):
        # Skip already-seen URLs (dedup across runs)
        url = response.url
        if url in self.seen_urls:
            return

        content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore")
        if "text/html" not in content_type:
            return

        html = response.text
        fname = safe_filename(url)

        # Save raw HTML only (trafilatura post-processing done separately)
        (self.html_path / (fname + ".html")).write_text(html, encoding="utf-8")

        # Track as seen
        self.seen_urls.add(url)
        self._seen_buffer.append(url)
        if len(self._seen_buffer) >= 50:
            self._flush_seen()

        self.page_count += 1
        if self.page_count % 50 == 0:
            print(f"[{self.site_domain}] {self.page_count} pages", flush=True)

    def parse_start_url(self, response):
        return self.parse_page(response)

    def closed(self, reason):
        self._flush_seen()
        print(f"[{self.site_domain}] DONE: {self.page_count} pages, reason={reason}", flush=True)


def main():
    if len(sys.argv) < 3:
        print("Usage: crawl_worker.py <url> <domain>")
        sys.exit(1)

    url = sys.argv[1]
    domain = sys.argv[2]
    dirname = safe_dirname(domain)

    process = CrawlerProcess()
    process.crawl(
        SiteSpider,
        site_url=url,
        site_domain=domain,
        html_dir=f"crawled_sites/{dirname}/html",
    )
    process.start()


if __name__ == "__main__":
    main()
