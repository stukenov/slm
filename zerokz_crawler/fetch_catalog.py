"""Fetch full site catalog from zero.kz API v2.0."""

import json
import requests

API_URL = "https://api.zero.kz/v2.0/site?limit=10000"
OUTPUT_FILE = "sites_to_crawl.json"


def main():
    print(f"Fetching catalog from {API_URL}...")
    r = requests.get(API_URL, timeout=30, headers={"Accept": "application/json"})
    r.raise_for_status()
    sites_raw = r.json()

    print(f"Got {len(sites_raw)} sites from API")

    sites = []
    for s in sites_raw:
        url = s.get("url", "").strip()
        if not url or not url.startswith("http"):
            continue
        if "zero.kz" in url:
            continue

        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        sites.append({
            "id": s.get("id"),
            "url": url if url.endswith("/") else url + "/",
            "domain": domain,
            "name": s.get("name", ""),
            "desc": s.get("desc", "")[:200],
        })

    # Deduplicate by domain
    seen = set()
    unique = []
    for s in sites:
        if s["domain"] not in seen:
            seen.add(s["domain"])
            unique.append(s)

    print(f"Unique external sites: {len(unique)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")

    for s in unique[:10]:
        print(f"  {s['domain']:35s} {s['name'][:40]}")
    if len(unique) > 10:
        print(f"  ... and {len(unique) - 10} more")


if __name__ == "__main__":
    main()
