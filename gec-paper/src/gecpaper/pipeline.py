from __future__ import annotations

import logging
from typing import Callable

from gecpaper.models.edit_tagger import apply_tags, extract_edit_tags

logger = logging.getLogger(__name__)


class DualPipeline:
    def __init__(
        self,
        tagger_fn: Callable[[list[str]], list[tuple[str, float]]] | None = None,
        seq2seq_fn: Callable[[str], str] | None = None,
        morph_fn: Callable[[str], str] | None = None,
        reranker_fn: Callable[[str, list[str]], str] | None = None,
        tagger_threshold: float = 0.9,
        mode: str = "cascade",
    ):
        self.tagger_fn = tagger_fn
        self.seq2seq_fn = seq2seq_fn
        self.morph_fn = morph_fn
        self.reranker_fn = reranker_fn
        self.tagger_threshold = tagger_threshold
        self.mode = mode

    def correct(self, text: str) -> str:
        if self.mode == "tagger_only":
            return self._run_tagger(text)
        elif self.mode == "seq2seq_only":
            return self._run_seq2seq(text)
        elif self.mode == "cascade":
            return self._run_cascade(text)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _run_tagger(self, text: str) -> str:
        if not self.tagger_fn:
            return text
        words = text.split()
        tag_preds = self.tagger_fn(words)
        tags = []
        for tag, conf in tag_preds:
            if conf >= self.tagger_threshold:
                tags.append(tag)
            else:
                tags.append("$KEEP")
        result = apply_tags(words, tags)
        return " ".join(result)

    def _run_seq2seq(self, text: str) -> str:
        if not self.seq2seq_fn:
            return text
        inp = text
        if self.morph_fn:
            inp = self.morph_fn(text)
        result = self.seq2seq_fn(inp)
        return result.replace("|", "")

    def _run_cascade(self, text: str) -> str:
        tagger_output = self._run_tagger(text)

        if tagger_output == text and self.seq2seq_fn:
            seq2seq_output = self._run_seq2seq(text)
        elif self.seq2seq_fn:
            seq2seq_output = self._run_seq2seq(tagger_output)
        else:
            return tagger_output

        candidates = [tagger_output, seq2seq_output]
        if self.reranker_fn:
            return self.reranker_fn(text, candidates)

        return seq2seq_output
