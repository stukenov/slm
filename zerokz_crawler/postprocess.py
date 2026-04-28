"""Post-process crawled HTML → markdown via trafilatura. Run after/during crawl."""

import sys
from pathlib import Path

import trafilatura


def process_site(site_dir: Path):
    html_dir = site_dir / "html"
    md_dir = site_dir / "md"

    if not html_dir.exists():
        return 0

    md_dir.mkdir(exist_ok=True)
    converted = 0

    for html_file in html_dir.glob("*.html"):
        md_file = md_dir / html_file.with_suffix(".md").name
        if md_file.exists():
            continue

        try:
            html = html_file.read_text(encoding="utf-8", errors="ignore")
            extracted = trafilatura.extract(
                html,
                include_links=True,
                include_tables=True,
                include_images=True,
                include_formatting=True,
                output_format="markdown",
            )
            if extracted and len(extracted.strip()) > 50:
                md_file.write_text(extracted, encoding="utf-8")
                converted += 1
        except Exception:
            pass

    return converted


def main():
    base = Path("crawled_sites")
    if not base.exists():
        print("No crawled_sites/ directory")
        return

    sites = sorted(d for d in base.iterdir() if d.is_dir())
    total = 0

    for i, site_dir in enumerate(sites, 1):
        n = process_site(site_dir)
        if n > 0:
            print(f"[{i}/{len(sites)}] {site_dir.name}: {n} new markdown files")
        total += n

    print(f"\nDone! {total} new markdown files created")


if __name__ == "__main__":
    main()
