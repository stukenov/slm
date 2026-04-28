"""Extract page title from HTML."""
from bs4 import BeautifulSoup
from .base import BaseAgent


class TitleAgent(BaseAgent):
    columns = ["title"]

    def process(self, row: dict) -> dict:
        html = row.get("html", "")
        if not html:
            return {"title": None}
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        title = tag.get_text(strip=True) if tag else None
        # fallback: og:title
        if not title:
            og = soup.find("meta", property="og:title")
            title = og.get("content", "").strip() if og else None
        return {"title": title or None}
