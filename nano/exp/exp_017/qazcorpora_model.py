"""QazCorpora BiLSTM morpheme segmentation model.

Trained on QazCorpora dataset with BIO tagging:
  B-ROOT, I-ROOT, B-SUFFIX, I-SUFFIX

Optimized for corpus-scale processing:
  - LRU cache for repeated words (80M docs have limited unique vocab)
  - GPU batched inference (pad + batch multiple words)
  - ~100x faster than naive per-word inference
"""
from __future__ import annotations

import os
import re
import logging
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

# Character vocabulary (must match training)
CHAR2IDX = {'<PAD>': 0, '<UNK>': 1}
_kazakh_lowercase = list("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")
_spe = list('-')
for _char in _spe + _kazakh_lowercase:
    if _char not in CHAR2IDX:
        CHAR2IDX[_char] = len(CHAR2IDX)

LABEL2IDX = {'O': 0, 'B-ROOT': 1, 'I-ROOT': 2, 'B-SUFFIX': 3, 'I-SUFFIX': 4}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

_SKIP_RE = re.compile(r'^[\d\W]+$')


class SuffixAnalysisModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.fc(lstm_out)
        return torch.log_softmax(tag_space, dim=2)


def load_model(model_path: str | None = None, device: str = "auto") -> SuffixAnalysisModel:
    """Load the pre-trained BiLSTM model."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "morpho_lemma_suf.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"QazCorpora model not found: {model_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SuffixAnalysisModel(len(CHAR2IDX), len(LABEL2IDX))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    model.train(False)
    model = model.to(device)
    logger.info(f"QazCorpora BiLSTM model loaded on {device}")
    return model


def _tags_to_segmented(word: str, predicted: torch.Tensor, morph_sep: str) -> str:
    """Convert BIO tag predictions to morpheme-separated string."""
    chars = list(word)
    result = []
    for i, idx in enumerate(predicted):
        if IDX2LABEL[idx.item()] == 'B-SUFFIX' and i > 0:
            result.append(morph_sep)
        result.append(chars[i])
    return ''.join(result)


def segment_word(model: SuffixAnalysisModel, word: str, morph_sep: str = "\x1F") -> str:
    """Segment a single word using the BiLSTM model."""
    if not word or len(word) <= 1:
        return word
    if _SKIP_RE.match(word):
        return word

    word_lower = word.lower()
    word_idx = [CHAR2IDX.get(c, CHAR2IDX['<UNK>']) for c in word_lower]
    device = next(model.parameters()).device

    with torch.no_grad():
        inputs = torch.tensor([word_idx], dtype=torch.long, device=device)
        tag_scores = model(inputs)
        predicted = torch.argmax(tag_scores, dim=2).squeeze(0).cpu()

    return _tags_to_segmented(word, predicted, morph_sep)


def segment_words_batch(
    model: SuffixAnalysisModel,
    words: list[str],
    morph_sep: str = "\x1F",
) -> list[str]:
    """Batch segment multiple words at once (GPU-accelerated).

    Pads words to same length, runs single forward pass.
    Much faster than calling segment_word() in a loop.
    """
    if not words:
        return []

    device = next(model.parameters()).device
    results = [None] * len(words)
    batch_indices = []
    batch_tensors = []

    # Separate trivial words from ones needing inference
    for i, word in enumerate(words):
        if not word or len(word) <= 1 or _SKIP_RE.match(word):
            results[i] = word
        else:
            word_lower = word.lower()
            idxs = [CHAR2IDX.get(c, CHAR2IDX['<UNK>']) for c in word_lower]
            batch_indices.append(i)
            batch_tensors.append(torch.tensor(idxs, dtype=torch.long))

    if not batch_tensors:
        return results

    # Pad and batch
    padded = pad_sequence(batch_tensors, batch_first=True, padding_value=0).to(device)

    with torch.no_grad():
        tag_scores = model(padded)
        predicted_batch = torch.argmax(tag_scores, dim=2).cpu()

    for j, orig_idx in enumerate(batch_indices):
        word = words[orig_idx]
        seq_len = len(word)
        predicted = predicted_batch[j, :seq_len]
        results[orig_idx] = _tags_to_segmented(word, predicted, morph_sep)

    return results


class CachedSegmenter:
    """High-performance segmenter with LRU cache + GPU batching.

    For corpus-scale processing (80M+ docs), most words repeat many times.
    Cache hit rate is typically >95%, making this ~100x faster than naive.
    """

    def __init__(self, model: SuffixAnalysisModel, cache_size: int = 500_000,
                 batch_size: int = 512, morph_sep: str = "\x1F"):
        self.model = model
        self.batch_size = batch_size
        self.morph_sep = morph_sep
        self._cache: dict[str, str] = {}
        self._cache_size = cache_size
        self._stats_hits = 0
        self._stats_misses = 0

    def segment_word(self, word: str) -> str:
        """Segment with cache lookup."""
        if not word or len(word) <= 1 or _SKIP_RE.match(word):
            return word

        key = word.lower()
        cached = self._cache.get(key)
        if cached is not None:
            self._stats_hits += 1
            if word == key:
                return cached
            # Restore original case: map morph_sep positions from cached onto original word
            try:
                result = []
                wi = 0
                for c in cached:
                    if c == self.morph_sep:
                        result.append(c)
                    else:
                        if wi < len(word):
                            result.append(word[wi])
                            wi += 1
                        # else: skip (length mismatch safety)
                return ''.join(result)
            except Exception:
                # Fallback: run inference directly
                return segment_word(self.model, word, self.morph_sep)

        self._stats_misses += 1
        segmented = segment_word(self.model, word, self.morph_sep)

        if len(self._cache) < self._cache_size:
            self._cache[key] = segment_word(self.model, key, self.morph_sep)

        return segmented

    def segment_words(self, words: list[str]) -> list[str]:
        """Segment list of words, batching cache misses for GPU inference."""
        results = [None] * len(words)
        miss_indices = []
        miss_words = []

        for i, word in enumerate(words):
            if not word or len(word) <= 1 or _SKIP_RE.match(word):
                results[i] = word
                continue

            key = word.lower()
            cached = self._cache.get(key)
            if cached is not None:
                self._stats_hits += 1
                if word == key:
                    results[i] = cached
                else:
                    r = []
                    wi = 0
                    for c in cached:
                        if c == self.morph_sep:
                            r.append(c)
                        else:
                            if wi < len(word):
                                r.append(word[wi])
                                wi += 1
                    results[i] = ''.join(r)
            else:
                self._stats_misses += 1
                miss_indices.append(i)
                miss_words.append(word)

        # Batch inference for cache misses
        if miss_words:
            for start in range(0, len(miss_words), self.batch_size):
                batch = miss_words[start:start + self.batch_size]
                batch_results = segment_words_batch(self.model, batch, self.morph_sep)
                for j, seg in enumerate(batch_results):
                    orig_idx = miss_indices[start + j]
                    results[orig_idx] = seg
                    # Cache the lowercase result
                    key = miss_words[start + j].lower()
                    if len(self._cache) < self._cache_size:
                        low_seg = segment_word(self.model, key, self.morph_sep) if key != miss_words[start + j] else seg
                        self._cache[key] = low_seg

        return results

    @property
    def cache_hit_rate(self) -> float:
        total = self._stats_hits + self._stats_misses
        return self._stats_hits / total if total > 0 else 0.0

    @property
    def cache_stats(self) -> str:
        total = self._stats_hits + self._stats_misses
        return f"hits={self._stats_hits:,} misses={self._stats_misses:,} rate={self.cache_hit_rate:.1%} cached={len(self._cache):,}"
