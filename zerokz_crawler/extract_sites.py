"""Extract external site URLs from crawled zero.kz card pages."""

import json
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

HTML_DIR = "html_pages"
OUTPUT_FILE = "external_sites.json"

SKIP_DOMAINS = {
    "zero.kz", "www.zero.kz",
    "liveinternet.ru", "www.liveinternet.ru",
    "ps.kz", "www.ps.kz",
    "top.mail.ru",
    "counter.yadro.ru",
    "google.com", "www.google.com",
    "facebook.com", "www.facebook.com",
    "twitter.com", "x.com",
    "instagram.com", "www.instagram.com",
    "youtube.com", "www.youtube.com",
    "t.me",
    "vk.com",
    "ok.ru",
}


def extract_site_url(html_path: Path) -> dict | None:
    try:
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")
    except Exception:
        return None

    # Find all external links, take the first one that looks like a real site
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.startswith("http"):
            continue

        parsed = urlparse(href)
        domain = parsed.netloc.lower()

        if any(skip in domain for skip in SKIP_DOMAINS):
            continue

        # Skip if it's just a tracking/counter link
        if not domain or len(domain) < 4:
            continue

        title = a.get_text(strip=True) or ""
        page_title = soup.title.string.strip() if soup.title and soup.title.string else ""

        return {
            "url": f"{parsed.scheme}://{parsed.netloc}/",
            "domain": domain,
            "link_text": title[:100],
            "page_title": page_title[:100],
            "source_file": html_path.name,
        }

    return None


def main():
    html_dir = Path(HTML_DIR)
    # Only process site card pages (site_id_*)
    files = sorted(html_dir.glob("*site_id_*.html"))
    print(f"Found {len(files)} site card pages")

    sites = {}
    for i, f in enumerate(files, 1):
        info = extract_site_url(f)
        if info and info["domain"] not in sites:
            sites[info["domain"]] = info
            if len(sites) % 100 == 0:
                print(f"[{i}/{len(files)}] {len(sites)} unique sites so far")

    print(f"\nTotal unique external sites: {len(sites)}")

    # Save
    result = sorted(sites.values(), key=lambda x: x["domain"])
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")

    # Sample
    for s in result[:10]:
        print(f"  {s['domain']:30s}  {s['link_text'][:40]}")
    if len(result) > 10:
        print(f"  ... and {len(result) - 10} more")


if __name__ == "__main__":
    main()
