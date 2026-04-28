"""Convert HTML to clean Markdown using markdownify + optional docling."""
import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from .base import BaseAgent

# Tags to strip before converting (nav, ads, footers, etc.)
STRIP_TAGS = [
    "script", "style", "noscript", "iframe", "svg",
    "nav", "footer", "aside", "header",
    "form", "button", "input", "select", "textarea",
]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    return str(soup)


def clean_markdown(text: str) -> str:
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


class MarkdownAgent(BaseAgent):
    columns = ["markdown"]

    def process(self, row: dict) -> dict:
        html = row.get("html", "")
        if not html:
            return {"markdown": None}

        cleaned = clean_html(html)
        result = md(
            cleaned,
            heading_style="ATX",
            bullets="-",
            strip=["a", "img"],   # keep text, drop link/image syntax
        )
        result = clean_markdown(result)
        return {"markdown": result if result else None}
