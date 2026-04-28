"""Extract clean plain text from HTML using trafilatura."""
import trafilatura
from .base import BaseAgent


class TextAgent(BaseAgent):
    columns = ["text", "text_len"]

    def process(self, row: dict) -> dict:
        html = row.get("html", "")
        if not html:
            return {"text": None, "text_len": 0}

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        )
        return {
            "text": text,
            "text_len": len(text) if text else 0,
        }
