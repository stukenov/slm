"""Base class for all processing agents (columns)."""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Each agent adds one or more columns to a row dict.

    Input:  row dict with at least {url, domain, html}
    Output: dict of new columns to merge into the row
    """

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        """Column names this agent produces."""

    @abstractmethod
    def process(self, row: dict) -> dict:
        """Process one row, return dict of new columns."""

    def safe_process(self, row: dict) -> dict:
        """Wraps process() with error handling — returns None values on failure."""
        try:
            return self.process(row)
        except Exception as e:
            return {col: None for col in self.columns}
