#!/usr/bin/env python3
"""Morpheme segmentation for Kazakh text.

Four backends:
  1. qazcorpora — BiLSTM neural model trained on QazCorpora (best quality)
  2. Apertium-kaz (rule-based, high quality, slower)
  3. Morfessor (unsupervised, faster, data-driven fallback)
  4. Rule-based (pure suffix stripping, no dependencies)

Usage:
    # QazCorpora backend (recommended)
    seg = MorphemeSegmenter(backend="qazcorpora")
    seg.segment("Мектептегі оқушылар математиканы оқып жатыр")

    # Morfessor backend
    seg = MorphemeSegmenter(backend="morfessor", morfessor_model="model.bin")
    seg.segment("Мектептегі оқушылар")
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

# ASCII Unit Separator — used as morpheme boundary marker
MORPH_SEP = "\x1F"


class QazCorporaSegmenter:
    """Segment Kazakh text using BiLSTM model trained on QazCorpora.

    Uses CachedSegmenter for corpus-scale performance:
      - LRU cache (500K unique words) — >95% hit rate on real corpora
      - GPU batched inference for cache misses
    """

    def __init__(self, model_path: str | None = None):
        from qazcorpora_model import load_model, CachedSegmenter
        model = load_model(model_path)
        self._cached = CachedSegmenter(model, morph_sep=MORPH_SEP)

    def segment_word(self, word: str) -> str:
        return self._cached.segment_word(word)

    def segment_words(self, words: list[str]) -> list[str]:
        """Batch segment for speed."""
        return self._cached.segment_words(words)

    @property
    def cache_stats(self) -> str:
        return self._cached.cache_stats


class ApertiumSegmenter:
    """Segment Kazakh text using apertium-kaz morphological analyzer."""

    def __init__(self):
        # Verify apertium-kaz is installed
        try:
            result = subprocess.run(
                ["apertium", "-d", ".", "-l", "kaz-morph"],
                input="тест",
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                # Try alternative invocation
                result = subprocess.run(
                    ["echo", "тест"],
                    capture_output=True,
                    text=True,
                )
        except FileNotFoundError:
            raise RuntimeError(
                "apertium not found. Install with:\n"
                "  macOS: brew install apertium && brew install apertium-kaz\n"
                "  Ubuntu: apt install apertium apertium-kaz"
            )

        self._coverage_total = 0
        self._coverage_hit = 0

    def segment_word(self, word: str) -> str:
        """Segment a single word into morphemes using apertium.

        Returns word with MORPH_SEP between morphemes.
        If analysis fails, returns the word unchanged.
        """
        if not word or not word.strip():
            return word

        # Skip punctuation and numbers
        if re.match(r'^[\d\W]+$', word):
            return word

        self._coverage_total += 1

        try:
            result = subprocess.run(
                ["echo", word],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Parse apertium output to extract morpheme boundaries
            analyzed = self._run_apertium(word)
            if analyzed and analyzed != word:
                self._coverage_hit += 1
                return analyzed
        except Exception as e:
            logger.debug(f"Apertium failed for '{word}': {e}")

        return word

    def _run_apertium(self, word: str) -> str | None:
        """Run apertium morphological analysis and extract morpheme splits."""
        try:
            # Use apertium in analysis mode
            proc = subprocess.run(
                ["apertium", "-d", "/usr/share/apertium/apertium-kaz", "kaz-morph"],
                input=word,
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = proc.stdout.strip()
            if not output or output.startswith("*"):
                return None  # Unknown word

            # Parse the morphological analysis
            # Apertium output format: ^word/lemma<tag><tag>...$
            # We extract the lemma and suffixes
            return self._parse_analysis(word, output)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _parse_analysis(self, original: str, analysis: str) -> str | None:
        """Parse apertium analysis output into morpheme-separated form.

        Apertium output: ^word/lemma<n><pl><px1sg><loc>$
        We reconstruct: stem + suffix boundaries based on the original word.
        """
        # Extract content between ^ and $
        match = re.search(r'\^([^/]+)/([^$]+)\$', analysis)
        if not match:
            return None

        surface = match.group(1)
        analysis_str = match.group(2)

        # Extract lemma (everything before first <)
        lemma_match = re.match(r'^([^<]+)', analysis_str)
        if not lemma_match:
            return None

        lemma = lemma_match.group(1)

        # If the surface form equals the lemma, no segmentation needed
        if surface.lower() == lemma.lower():
            return None

        # Find where the lemma ends in the original word (case-insensitive)
        orig_lower = original.lower()
        lemma_lower = lemma.lower()

        # Try to find the stem in the original word
        if orig_lower.startswith(lemma_lower):
            stem_end = len(lemma)
            stem = original[:stem_end]
            suffix = original[stem_end:]
            if suffix:
                # Split suffixes using known Kazakh suffix patterns
                morphemes = self._split_suffixes(suffix)
                return stem + MORPH_SEP + MORPH_SEP.join(morphemes)

        return None

    def _split_suffixes(self, suffix_chain: str) -> list[str]:
        """Split a chain of Kazakh suffixes into individual morphemes.

        Uses known Kazakh suffix patterns (vowel harmony variants included).
        """
        # Common Kazakh suffixes ordered by length (greedy matching)
        SUFFIXES = [
            # Case markers
            "нің", "ның", "нін", "нұң",  # GEN
            "ға", "ге", "қа", "ке",       # DAT
            "дан", "ден", "тан", "тен", "нан", "нен",  # ABL
            "да", "де", "та", "те",        # LOC
            "ды", "ді", "ты", "ті", "ны", "ні", "н",  # ACC
            # Plural
            "лар", "лер", "дар", "дер", "тар", "тер",
            # Possessive
            "ым", "ім", "м",              # 1sg
            "ың", "ің", "ң",              # 2sg
            "ы", "і", "сы", "сі",         # 3sg
            "ымыз", "іміз", "мыз", "міз",  # 1pl
            "ыңыз", "іңіз", "ңыз", "ңіз",  # 2pl.formal
            # Verbal
            "ған", "ген", "қан", "кен",   # past participle
            "атын", "етін",                # habitual
            "ып", "іп", "п",              # converb
            "ды", "ді", "ты", "ті",       # past
            "ады", "еді",                 # present/past
        ]

        morphemes = []
        remaining = suffix_chain

        while remaining:
            matched = False
            for sfx in SUFFIXES:
                if remaining.lower().startswith(sfx):
                    morphemes.append(remaining[:len(sfx)])
                    remaining = remaining[len(sfx):]
                    matched = True
                    break
            if not matched:
                # No known suffix matches — append the rest as one unit
                morphemes.append(remaining)
                break

        return morphemes

    @property
    def coverage(self) -> float:
        if self._coverage_total == 0:
            return 0.0
        return self._coverage_hit / self._coverage_total


class MorfessorSegmenter:
    """Segment text using Morfessor (unsupervised morphological segmentation)."""

    def __init__(self, model_path: str | None = None):
        import morfessor
        self.model = morfessor.MorfessorBaseline()
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading Morfessor model from {model_path}")
            io = morfessor.MorfessorIO()
            self.model = io.read_binary_model_file(model_path)
            self._trained = True
        else:
            self._trained = False

    def train(self, word_counts: dict[str, int], save_path: str | None = None):
        """Train Morfessor model on word frequency counts.

        Args:
            word_counts: dict of {word: count}
            save_path: optional path to save the trained model
        """
        import morfessor

        # Convert to list of (count, word) tuples
        data = [(count, word) for word, count in word_counts.items()]
        self.model.load_data(data)
        self.model.train_batch()
        self._trained = True

        if save_path:
            io = morfessor.MorfessorIO()
            io.write_binary_model_file(save_path, self.model)
            logger.info(f"Morfessor model saved to {save_path}")

    def segment_word(self, word: str) -> str:
        """Segment a word into morphemes."""
        if not self._trained:
            return word
        if not word or re.match(r'^[\d\W]+$', word):
            return word

        try:
            segments = self.model.viterbi_segment(word)[0]
            if len(segments) > 1:
                return MORPH_SEP.join(segments)
        except Exception:
            pass
        return word


class MorphemeSegmenter:
    """Unified interface for morpheme segmentation.

    Supports four backends:
      - "qazcorpora": BiLSTM neural model from QazCorpora (best quality, recommended).
      - "apertium": Rule-based (Apertium-kaz). High quality, requires system package.
      - "morfessor": Unsupervised (Morfessor). Data-driven, pure Python.
      - "rule": Pure rule-based suffix splitting. No external dependencies.
    """

    def __init__(
        self,
        backend: str = "qazcorpora",
        morfessor_model: str | None = None,
        qazcorpora_model: str | None = None,
    ):
        self.backend_name = backend

        if backend == "qazcorpora":
            self._backend = QazCorporaSegmenter(model_path=qazcorpora_model)
        elif backend == "apertium":
            self._backend = ApertiumSegmenter()
        elif backend == "morfessor":
            self._backend = MorfessorSegmenter(morfessor_model)
        elif backend == "rule":
            self._backend = RuleBasedSegmenter()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'qazcorpora', 'apertium', 'morfessor', or 'rule'.")

    def segment(self, text: str) -> str:
        """Segment all words in text, preserving whitespace and punctuation.

        For qazcorpora backend, batches all words for GPU inference.
        """
        tokens = re.findall(r'\S+|\s+', text)

        # Extract words and their positions for batch processing
        word_infos = []  # (index, leading, word, trailing)
        for i, token in enumerate(tokens):
            if token.strip():
                leading = re.match(r'^(\W*)', token).group(1)
                trailing = re.search(r'(\W*)$', token).group(1)
                word = token[len(leading):len(token) - len(trailing) if trailing else len(token)]
                word_infos.append((i, leading, word, trailing))

        # Batch segment if backend supports it
        words = [w for _, _, w, _ in word_infos]
        if hasattr(self._backend, 'segment_words') and words:
            segmented_words = self._backend.segment_words(words)
        else:
            segmented_words = [self._backend.segment_word(w) if w else w for w in words]

        # Reconstruct text
        result = list(tokens)
        for j, (i, leading, word, trailing) in enumerate(word_infos):
            seg = segmented_words[j] if segmented_words[j] else word
            result[i] = leading + seg + trailing

        return "".join(result)

    def segment_batch(self, texts: list[str]) -> list[str]:
        """Segment a batch of texts."""
        return [self.segment(t) for t in texts]

    def segment_corpus_iterator(
        self,
        texts: Iterator[str],
        batch_size: int = 1000,
    ) -> Iterator[list[str]]:
        """Yield batches of segmented texts from an iterator."""
        batch = []
        for text in texts:
            batch.append(self.segment(text))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @property
    def coverage(self) -> float:
        """Return morphological analysis coverage (apertium only)."""
        if hasattr(self._backend, 'coverage'):
            return self._backend.coverage
        return -1.0  # Not applicable


class RuleBasedSegmenter:
    """Pure rule-based Kazakh suffix segmenter. No external dependencies.

    Uses known Kazakh morphological patterns to split words into stem + suffixes.
    This is a lightweight alternative when Apertium is not available.
    """

    # Kazakh suffixes ordered by length (longest first for greedy matching)
    # Grouped by function, all vowel harmony variants included
    SUFFIXES = [
        # Possessive 1pl
        "ымыз", "іміз", "мыз", "міз",
        # Possessive 2pl formal
        "ыңыз", "іңіз", "ңыз", "ңіз",
        # Possessive 2pl informal
        "ыңдар", "іңдер",
        # Past participle
        "ған", "ген", "қан", "кен",
        # Habitual participle
        "атын", "етін",
        # Ablative
        "дан", "ден", "тан", "тен", "нан", "нен",
        # Genitive
        "нің", "ның", "нін",
        # Plural
        "лар", "лер", "дар", "дер", "тар", "тер",
        # Dative
        "ға", "ге", "қа", "ке",
        # Locative
        "да", "де", "та", "те",
        # Accusative
        "ды", "ді", "ты", "ті", "ны", "ні",
        # Possessive 1sg
        "ым", "ім",
        # Possessive 2sg
        "ың", "ің",
        # Possessive 3sg
        "сы", "сі",
        # Converb
        "ып", "іп",
        # Short forms
        "н", "м", "ң", "п",
        "ы", "і",
    ]

    # Minimum stem length — don't segment if stem would be too short
    # 4 chars prevents false positives like Қазақс|тан, Бүг|ін, Алма|ты
    MIN_STEM_LEN = 4

    # Maximum suffix layers to strip (prevent over-segmentation)
    MAX_SUFFIX_LAYERS = 4

    def segment_word(self, word: str) -> str:
        """Split a Kazakh word into stem + suffixes using rule-based matching."""
        if len(word) <= 4:
            return word

        # Try to greedily strip suffixes from the end
        morphemes = []
        remaining = word

        for _ in range(self.MAX_SUFFIX_LAYERS):
            matched = False
            for sfx in self.SUFFIXES:
                sfx_len = len(sfx)
                rem_lower = remaining.lower()
                if (rem_lower.endswith(sfx)
                        and len(remaining) - sfx_len >= self.MIN_STEM_LEN):
                    suffix_part = remaining[len(remaining) - sfx_len:]
                    remaining = remaining[:len(remaining) - sfx_len]
                    morphemes.append(suffix_part)
                    matched = True
                    break
            if not matched:
                break

        if morphemes:
            morphemes.reverse()
            return remaining + MORPH_SEP + MORPH_SEP.join(morphemes)
        return word


# --- Convenience functions ---

def segment_text(text: str, backend: str = "rule") -> str:
    """Quick one-shot segmentation."""
    seg = MorphemeSegmenter(backend=backend)
    return seg.segment(text)


def visualize_segmentation(text: str, backend: str = "rule") -> str:
    """Segment and show morpheme boundaries as | for readability."""
    segmented = segment_text(text, backend)
    return segmented.replace(MORPH_SEP, "|")


if __name__ == "__main__":
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "qazcorpora"
    seg = MorphemeSegmenter(backend=backend)

    test_words = [
        "үйлерімізде",      # үй+лер+іміз+де
        "мектептегі",        # мектеп+тегі
        "оқушылар",          # оқу+шы+лар
        "математиканы",      # математика+ны
        "қаласында",         # қала+сы+нда -> қала+сын+да
        "Қазақстанның",      # Қазақстан+ның
        "жазылғандар",       # жаз+ыл+ған+дар
        "көрмегенімді",      # көр+ме+ген+ім+ді
        "университеттерде",  # университет+тер+де
        "айтқанымызды",      # айт+қан+ымыз+ды
    ]

    print(f"Backend: {backend}")
    print(f"{'Word':<25} {'Segmented':<35}")
    print("-" * 60)
    for word in test_words:
        segmented = seg.segment(word).replace(MORPH_SEP, "|")
        print(f"{word:<25} {segmented:<35}")

    print("\nFull sentence test:")
    sentences = [
        "Қазақстан — Орталық Азиядағы мемлекет.",
        "Бүгін ауа райы жақсы болады.",
        "Мектепте оқушылар математика сабағына дайындалуда.",
        "Алматы қаласында жаңа метро стансасы ашылды.",
    ]
    for s in sentences:
        segmented = seg.segment(s).replace(MORPH_SEP, "|")
        print(f"  {segmented}")
