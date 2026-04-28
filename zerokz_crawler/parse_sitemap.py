"""Parse zero.kz sitemap.xml and extract all URLs."""

import json
import requests
from lxml import etree

SITEMAP_URL = "https://zero.kz/sitemap.xml"
OUTPUT_FILE = "urls.json"


def fetch_sitemap(url: str) -> list[str]:
    print(f"Fetching sitemap from {url}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    root = etree.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Check if it's a sitemap index (contains other sitemaps)
    sitemap_refs = root.findall(".//sm:sitemap/sm:loc", ns)
    if sitemap_refs:
        print(f"Found sitemap index with {len(sitemap_refs)} sub-sitemaps")
        all_urls = []
        for ref in sitemap_refs:
            sub_urls = fetch_sitemap(ref.text.strip())
            all_urls.extend(sub_urls)
        return all_urls

    # Regular sitemap — extract URLs
    urls = [loc.text.strip() for loc in root.findall(".//sm:url/sm:loc", ns)]
    print(f"Found {len(urls)} URLs")
    return urls


def main():
    urls = fetch_sitemap(SITEMAP_URL)
    print(f"\nTotal URLs: {len(urls)}")

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(urls, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")

    # Show sample
    for url in urls[:10]:
        print(f"  {url}")
    if len(urls) > 10:
        print(f"  ... and {len(urls) - 10} more")


if __name__ == "__main__":
    main()
