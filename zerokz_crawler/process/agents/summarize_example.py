"""
Example: summarization agent using a local model.
Copy this file, adapt, then run:
  python run_agent.py --agent agents/summarize_example.py --filter-lang kk
"""
from .base import BaseAgent


class SummarizeAgent(BaseAgent):
    columns = ["summary"]

    def __init__(self, max_input=1024, max_summary=150):
        self.max_input = max_input
        self.max_summary = max_summary
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                "summarization",
                model="facebook/mbart-large-cc25",  # swap for your model
                device=0,
            )
        return self._pipe

    def process(self, row: dict) -> dict:
        text = row.get("text", "") or ""
        if len(text) < 100:
            return {"summary": None}
        pipe = self._get_pipe()
        result = pipe(text[:self.max_input], max_length=self.max_summary, truncation=True)
        return {"summary": result[0]["summary_text"]}


# Must define AGENT for run_agent.py to pick up
AGENT = SummarizeAgent()
