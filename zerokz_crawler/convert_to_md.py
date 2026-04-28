"""Convert crawled HTML pages to Markdown using docling."""

import json
from pathlib import Path

from docling.document_converter import DocumentConverter

HTML_DIR = "html_pages"
MD_DIR = "markdown_output"


def main():
    html_dir = Path(HTML_DIR)
    md_dir = Path(MD_DIR)
    md_dir.mkdir(exist_ok=True)

    html_files = sorted(html_dir.glob("*.html"))
    print(f"Found {len(html_files)} HTML files to convert")

    converter = DocumentConverter()
    success = 0
    failed = 0

    for i, html_file in enumerate(html_files, 1):
        md_file = md_dir / html_file.with_suffix(".md").name
        if md_file.exists():
            continue

        try:
            result = converter.convert(str(html_file))
            markdown = result.document.export_to_markdown()

            if markdown.strip():
                md_file.write_text(markdown, encoding="utf-8")
                success += 1
                print(f"[{i}/{len(html_files)}] OK  {html_file.name} -> {md_file.name}")
            else:
                failed += 1
                print(f"[{i}/{len(html_files)}] EMPTY {html_file.name}")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(html_files)}] ERR  {html_file.name}: {e}")

    print(f"\nDone! Success: {success}, Failed: {failed}")

    # Save summary
    summary = {
        "total_html": len(html_files),
        "converted": success,
        "failed": failed,
    }
    with open(md_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
