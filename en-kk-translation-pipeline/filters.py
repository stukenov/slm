"""
Pre-translation text filters for FineWeb-Edu EN→KK pipeline.

Filters (cheap to expensive, early exit):
1. Length: skip < 50 chars or > 10,000 chars
2. Exact dedup: xxhash on normalized text
3. Language detection: fasttext lid.176.bin, threshold 0.8
4. Near-dedup: MinHash LSH (datasketch), Jaccard threshold 0.8
"""

import os
import re
import unicodedata
from collections import Counter

import xxhash


BASE = os.path.dirname(os.path.abspath(__file__))

# Precompiled patterns
_WHITESPACE_RE = re.compile(r'\s+')
_WORD_RE = re.compile(r'\w+', re.UNICODE)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    t = unicodedata.normalize("NFKC", text.lower())
    return _WHITESPACE_RE.sub(" ", t).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingles for MinHash."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


class TextFilter:
    """Applies a cascade of filters with statistics tracking."""

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10_000,
        lang_threshold: float = 0.8,
        lang_detect_chars: int = 500,
        fuzzy_dedup: bool = False,
        jaccard_threshold: float = 0.8,
        minhash_perms: int = 128,
        fasttext_model_path: str | None = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.lang_threshold = lang_threshold
        self.lang_detect_chars = lang_detect_chars
        self.fuzzy_dedup = fuzzy_dedup
        self.jaccard_threshold = jaccard_threshold
        self.minhash_perms = minhash_perms

        # Stats
        self.stats: Counter = Counter()

        # Exact dedup set
        self._seen_hashes: set[int] = set()

        # Language detection (lazy load)
        self._ft_model = None
        self._ft_path = fasttext_model_path or os.path.join(BASE, "lid.176.bin")

        # MinHash LSH (lazy init)
        self._lsh = None

    def _get_ft_model(self):
        if self._ft_model is None:
            import numpy as np
            import fasttext
            fasttext.FastText.eprint = lambda x: None
            self._ft_model = fasttext.load_model(self._ft_path)
            # Patch numpy 2.x incompatibility in fasttext predict
            _f = self._ft_model.f
            def _patched_predict(text, k=1, threshold=0.0, on_unicode_error='strict'):
                result = _f.predict(text, k, threshold, on_unicode_error)
                labels = [r[1] for r in result]
                probs = np.asarray([r[0] for r in result])
                return labels, probs
            self._ft_model.predict = _patched_predict
        return self._ft_model

    def _get_lsh(self):
        if self._lsh is None:
            from datasketch import MinHashLSH
            self._lsh = MinHashLSH(
                threshold=self.jaccard_threshold,
                num_perm=self.minhash_perms,
            )
        return self._lsh

    def _make_minhash(self, text: str):
        from datasketch import MinHash
        m = MinHash(num_perm=self.minhash_perms)
        for s in _shingles(text):
            m.update(s.encode("utf-8"))
        return m

    def filter(self, text: str, doc_id: str = "") -> tuple[bool, str]:
        """
        Returns (keep, reason).
        keep=True means text passes all filters.
        reason is empty string if kept, otherwise the filter name that rejected it.
        """
        self.stats["total"] += 1

        # 1. Length
        n = len(text)
        if n < self.min_length or n > self.max_length:
            self.stats["length"] += 1
            return False, "length"

        # 2. Exact dedup
        norm = _normalize(text)
        h = xxhash.xxh64_intdigest(norm)
        if h in self._seen_hashes:
            self.stats["exact_dedup"] += 1
            return False, "exact_dedup"
        self._seen_hashes.add(h)

        # 3. Language detection
        snippet = text[:self.lang_detect_chars].replace("\n", " ")
        model = self._get_ft_model()
        labels, scores = model.predict(snippet)
        lang = labels[0].replace("__label__", "")
        score = float(scores[0])
        if lang != "en" or score < self.lang_threshold:
            self.stats["lang_detect"] += 1
            return False, "lang_detect"

        # 4. Near-dedup (MinHash LSH)
        if self.fuzzy_dedup:
            mh = self._make_minhash(text)
            lsh = self._get_lsh()
            result = lsh.query(mh)
            if result:
                self.stats["near_dedup"] += 1
                return False, "near_dedup"
            key = doc_id or str(self.stats["total"])
            try:
                lsh.insert(key, mh)
            except ValueError:
                pass  # duplicate key, ignore

        self.stats["kept"] += 1
        return True, ""

    def summary(self) -> str:
        total = self.stats["total"]
        if total == 0:
            return "No documents processed."
        kept = self.stats["kept"]
        lines = [
            f"Filter summary: {kept}/{total} kept ({kept / total * 100:.1f}%)",
            f"  length:      {self.stats['length']}",
            f"  exact_dedup: {self.stats['exact_dedup']}",
            f"  lang_detect: {self.stats['lang_detect']}",
            f"  near_dedup:  {self.stats['near_dedup']}",
        ]
        return "\n".join(lines)
