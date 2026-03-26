# Translation Pipeline v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular EN→KK translation pipeline for FineWeb-Edu with sentence-level confidence scoring, quality filtering, and incremental translation support.

**Architecture:** Modular pipeline with 5 components: config → sentence_splitter → translator → postprocessor → pipeline orchestrator. Each module is independently testable. Data flows through sentence splitting, translation with confidence, post-translation filtering, and document reassembly.

**Tech Stack:** Python 3.10+, CTranslate2, SentencePiece, xxhash, datasets, huggingface_hub

**Spec:** `docs/superpowers/specs/2026-03-26-translation-pipeline-v2-design.md`

---

### Task 1: Create config.py with all constants

**Files:**
- Create: `translation-pipeline-v2/config.py`

- [ ] **Step 1: Create the config file**

```python
"""Configuration constants for translation pipeline v2."""

# Translation model
CT2_MODEL_NAME = "HPLT/translate-en-kk-v2.0-hplt_opus"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 4096
BEAM_SIZE = 1
MAX_INPUT_LENGTH = 128
MAX_DECODING_LENGTH = 200

# Source dataset
SOURCE_DATASET = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample-10BT"
ROWS_PER_CHUNK = 1_000_000

# HuggingFace output
HF_REPO = "stukenov/sozkz-fineweb-edu-kk-v2"

# Pre-translation filters (per sentence)
MIN_WORDS_PER_SENTENCE = 3
MAX_SENTENCE_LENGTH = 512
NON_ALPHA_THRESHOLD = 0.3  # >30% non-alpha chars → skip

# Post-translation filters (per sentence)
DUPLICATE_SIMILARITY_THRESHOLD = 0.9  # translation ≈ original → skip
LENGTH_RATIO_MAX = 3.0  # translation too long vs original
LENGTH_RATIO_MIN = 0.3  # translation too short vs original
NGRAM_REPEAT_THRESHOLD = 3  # 3+ repeats of same n-gram → skip

# Testing
TEST_SAMPLE_SIZE = 100
VALIDATION_SAMPLE_SIZE = 1000

# Checkpoint
CHECKPOINT_EVERY = 100_000
```

- [ ] **Step 2: Commit**

```bash
git add translation-pipeline-v2/config.py
git commit -m "feat(translation-v2): add config with all pipeline constants"
```

---

### Task 2: Create filters.py with sentence-level pre/post filters

**Files:**
- Create: `translation-pipeline-v2/filters.py`

- [ ] **Step 1: Write tests for pre-translation filters**

Create `translation-pipeline-v2/test_filters.py`:

```python
"""Tests for sentence-level filters."""

from filters import (
    is_noisy_sentence,
    is_translation_bad,
    char_similarity,
    has_ngram_repetition,
)


def test_noisy_math_formula():
    assert is_noisy_sentence("∫₀¹ f(x)dx = F(1) - F(0)") is True


def test_noisy_code():
    assert is_noisy_sentence("if (x > 0) { return x * 2; }") is True


def test_clean_sentence():
    assert is_noisy_sentence("The capital of Kazakhstan is Astana.") is False


def test_too_short():
    assert is_noisy_sentence("Hi there") is True  # <3 words


def test_too_long():
    assert is_noisy_sentence("word " * 200) is True  # >512 chars


def test_borderline_non_alpha():
    # Exactly 30% non-alpha should pass
    text = "abc def ghi" + "!@#"  # 8 alpha + 3 spaces + 3 special = 14 chars, 3/14=21%
    assert is_noisy_sentence(text) is False


def test_char_similarity_identical():
    assert char_similarity("hello world", "hello world") == 1.0


def test_char_similarity_different():
    assert char_similarity("hello", "xyzab") < 0.5


def test_translation_duplicate():
    assert is_translation_bad("Hello world", "Hello world") is True


def test_translation_good():
    assert is_translation_bad("Hello world", "Сәлем әлем") is False


def test_ngram_repetition_detected():
    assert has_ngram_repetition("және және және және бұл") is True


def test_ngram_repetition_normal():
    assert has_ngram_repetition("Бүгін ауа-райы жақсы болды") is False


def test_translation_too_long():
    original = "Short text"
    translated = "Бұл өте ұзын мәтін " * 20  # way longer than 3x
    assert is_translation_bad(original, translated) is True


def test_translation_too_short():
    original = "This is a fairly long sentence with many words in it"
    translated = "Қыс"  # way shorter than 0.3x
    assert is_translation_bad(original, translated) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd translation-pipeline-v2 && python -m pytest test_filters.py -v
```

Expected: FAIL — `filters` module not found.

- [ ] **Step 3: Implement filters.py**

```python
"""Sentence-level pre-translation and post-translation filters."""

import re
from difflib import SequenceMatcher

from config import (
    MIN_WORDS_PER_SENTENCE,
    MAX_SENTENCE_LENGTH,
    NON_ALPHA_THRESHOLD,
    DUPLICATE_SIMILARITY_THRESHOLD,
    LENGTH_RATIO_MAX,
    LENGTH_RATIO_MIN,
    NGRAM_REPEAT_THRESHOLD,
)

_WORD_RE = re.compile(r'\b\w+\b', re.UNICODE)


def is_noisy_sentence(sentence: str) -> bool:
    """Pre-translation filter. Returns True if sentence should be skipped.

    Checks:
    - Too short (<MIN_WORDS words)
    - Too long (>MAX_SENTENCE_LENGTH chars)
    - Too many non-alpha characters (>NON_ALPHA_THRESHOLD ratio)
    """
    sentence = sentence.strip()
    if not sentence:
        return True

    # Too long
    if len(sentence) > MAX_SENTENCE_LENGTH:
        return True

    # Too short
    words = _WORD_RE.findall(sentence)
    if len(words) < MIN_WORDS_PER_SENTENCE:
        return True

    # Non-alpha ratio
    alpha_count = sum(1 for c in sentence if c.isalpha())
    total_count = len(sentence.replace(" ", ""))
    if total_count == 0:
        return True
    non_alpha_ratio = 1.0 - (alpha_count / total_count)
    if non_alpha_ratio > NON_ALPHA_THRESHOLD:
        return True

    return False


def char_similarity(a: str, b: str) -> float:
    """Character-level similarity between two strings (0.0 to 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def has_ngram_repetition(text: str, n: int = 2) -> bool:
    """Check if text contains repeated n-grams (sign of model looping).

    Returns True if any n-gram appears NGRAM_REPEAT_THRESHOLD+ times.
    """
    words = text.split()
    if len(words) < n + NGRAM_REPEAT_THRESHOLD:
        return False

    ngram_counts: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        if ngram_counts[ngram] >= NGRAM_REPEAT_THRESHOLD:
            return True
    return False


def is_translation_bad(original: str, translated: str) -> bool:
    """Post-translation filter. Returns True if translation should be skipped.

    Checks:
    - Translation too similar to original (copy-through)
    - N-gram repetition (model looping)
    - Length ratio too extreme
    - Disproportionate special characters
    """
    if not translated or not translated.strip():
        return True

    # Translation ≈ original (copy-through)
    if char_similarity(original, translated) > DUPLICATE_SIMILARITY_THRESHOLD:
        return True

    # N-gram repetition
    if has_ngram_repetition(translated):
        return True

    # Length ratio
    len_orig = len(original)
    len_trans = len(translated)
    if len_orig > 0:
        ratio = len_trans / len_orig
        if ratio > LENGTH_RATIO_MAX or ratio < LENGTH_RATIO_MIN:
            return True

    # Special char ratio: if translated has way more special chars than original
    orig_special = sum(1 for c in original if not c.isalpha() and not c.isspace())
    trans_special = sum(1 for c in translated if not c.isalpha() and not c.isspace())
    orig_ratio = orig_special / max(len(original), 1)
    trans_ratio = trans_special / max(len(translated), 1)
    if trans_ratio > orig_ratio * 2 + 0.1:
        return True

    return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd translation-pipeline-v2 && python -m pytest test_filters.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add translation-pipeline-v2/filters.py translation-pipeline-v2/test_filters.py
git commit -m "feat(translation-v2): add sentence-level pre/post filters with tests"
```

---

### Task 3: Create sentence_splitter.py

**Files:**
- Create: `translation-pipeline-v2/sentence_splitter.py`

- [ ] **Step 1: Write tests**

Add to `translation-pipeline-v2/test_filters.py` (or create new file `test_sentence_splitter.py`):

```python
"""Tests for sentence splitter."""

from sentence_splitter import split_document, reassemble_document


def test_split_simple():
    text = "Hello world. This is a test."
    result = split_document(text)
    assert len(result["sentences"]) == 2
    assert result["sentences"][0]["text"] == "Hello world."
    assert result["sentences"][1]["text"] == "This is a test."
    assert result["sentences"][0]["para_idx"] == 0
    assert result["sentences"][0]["sent_idx"] == 0


def test_split_preserves_paragraphs():
    text = "First paragraph.\n\nSecond paragraph."
    result = split_document(text)
    sents = result["sentences"]
    assert sents[0]["para_idx"] != sents[1]["para_idx"]


def test_split_filters_noisy():
    text = "Good sentence here. ∫₀¹ f(x)dx = 0. Another good one."
    result = split_document(text)
    clean = [s for s in result["sentences"] if not s["skipped"]]
    skipped = [s for s in result["sentences"] if s["skipped"]]
    assert len(clean) == 2
    assert len(skipped) == 1


def test_reassemble():
    text = "Hello world. Nice day.\n\nSecond paragraph here."
    doc = split_document(text)
    translations = {}
    for s in doc["sentences"]:
        if not s["skipped"]:
            translations[s["sent_idx"]] = f"[KK]{s['text']}"
    reassembled = reassemble_document(doc, translations)
    assert "\n" in reassembled  # paragraph break preserved
    assert "[KK]" in reassembled


def test_reassemble_partial():
    text = "Good sentence. Bad: ∫₀¹ f(x)dx. Another good."
    doc = split_document(text)
    translations = {}
    for s in doc["sentences"]:
        if not s["skipped"]:
            translations[s["sent_idx"]] = f"[KK]{s['text']}"
    reassembled = reassemble_document(doc, translations)
    assert "[KK]Good sentence." in reassembled
    assert "[KK]Another good." in reassembled
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd translation-pipeline-v2 && python -m pytest test_sentence_splitter.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement sentence_splitter.py**

```python
"""Split documents into sentences with paragraph preservation and pre-filtering."""

import re
from filters import is_noisy_sentence

SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ\d"])')


def split_document(text: str) -> dict:
    """Split document into sentences preserving paragraph structure.

    Returns dict:
        sentences: list of {text, para_idx, sent_idx, skipped, skip_reason}
        paragraph_count: int
    """
    paragraphs = text.split('\n')
    sentences = []
    global_sent_idx = 0

    for para_idx, para in enumerate(paragraphs):
        para_stripped = para.strip()
        if not para_stripped:
            # Empty paragraph marker — preserved for reassembly
            sentences.append({
                "text": "",
                "para_idx": para_idx,
                "sent_idx": global_sent_idx,
                "skipped": True,
                "skip_reason": "empty_paragraph",
                "is_paragraph_break": True,
            })
            global_sent_idx += 1
            continue

        parts = SENT_RE.split(para_stripped)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            skipped = is_noisy_sentence(part)
            sentences.append({
                "text": part,
                "para_idx": para_idx,
                "sent_idx": global_sent_idx,
                "skipped": skipped,
                "skip_reason": "noisy" if skipped else "",
                "is_paragraph_break": False,
            })
            global_sent_idx += 1

    return {
        "sentences": sentences,
        "paragraph_count": len(paragraphs),
    }


def reassemble_document(doc: dict, translations: dict[int, str]) -> str:
    """Reassemble translated sentences into document preserving paragraph breaks.

    Args:
        doc: output of split_document
        translations: {sent_idx: translated_text} for non-skipped sentences

    Returns: reassembled translated text
    """
    paragraphs: dict[int, list[str]] = {}

    for sent in doc["sentences"]:
        para_idx = sent["para_idx"]
        paragraphs.setdefault(para_idx, [])

        if sent.get("is_paragraph_break"):
            continue

        sent_idx = sent["sent_idx"]
        if sent_idx in translations:
            paragraphs[para_idx].append(translations[sent_idx])
        # Skipped sentences without translation are simply omitted from output

    result = []
    for para_idx in sorted(paragraphs.keys()):
        sents = paragraphs[para_idx]
        if sents:
            result.append(" ".join(sents))
        else:
            result.append("")

    return "\n".join(result)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd translation-pipeline-v2 && python -m pytest test_sentence_splitter.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add translation-pipeline-v2/sentence_splitter.py translation-pipeline-v2/test_sentence_splitter.py
git commit -m "feat(translation-v2): add sentence splitter with paragraph preservation"
```

---

### Task 4: Create translator.py with confidence scoring

**Files:**
- Create: `translation-pipeline-v2/translator.py`

- [ ] **Step 1: Write tests (CPU-only, mock translator for unit tests)**

Create `translation-pipeline-v2/test_translator.py`:

```python
"""Tests for translator module (CPU-safe, no GPU needed)."""

import math
from translator import compute_sentence_confidence, TranslationResult


def test_confidence_from_scores():
    # log-probs: higher (closer to 0) = more confident
    scores = [-0.1, -0.2, -0.3]
    conf = compute_sentence_confidence(scores)
    expected = math.exp(sum(scores) / len(scores))
    assert abs(conf - expected) < 1e-6


def test_confidence_empty_scores():
    conf = compute_sentence_confidence([])
    assert conf == 0.0


def test_translation_result_structure():
    r = TranslationResult(
        text="Сәлем",
        confidence=0.85,
        token_scores=[-0.1, -0.2],
    )
    assert r.text == "Сәлем"
    assert r.confidence == 0.85
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd translation-pipeline-v2 && python -m pytest test_translator.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement translator.py**

```python
"""Translation via CTranslate2 with per-sentence confidence scoring."""

import math
import os
import time
from dataclasses import dataclass

import ctranslate2
import sentencepiece as spm

from config import (
    BATCH_SIZE,
    BEAM_SIZE,
    COMPUTE_TYPE,
    MAX_INPUT_LENGTH,
    MAX_DECODING_LENGTH,
)

BASE = os.path.dirname(os.path.abspath(__file__))
CT2_DIR = os.path.join(BASE, "model_ct2")
SPM_PATH = os.path.join(BASE, "model_cache", "model.en-kk.spm")


@dataclass
class TranslationResult:
    text: str
    confidence: float
    token_scores: list[float]


def compute_sentence_confidence(token_log_probs: list[float]) -> float:
    """Compute sentence confidence from token log-probabilities.

    Returns exp(mean(log_probs)) — a value in (0, 1].
    Higher = more confident.
    """
    if not token_log_probs:
        return 0.0
    mean_log_prob = sum(token_log_probs) / len(token_log_probs)
    return math.exp(mean_log_prob)


class Translator:
    """CTranslate2-based EN→KK translator with confidence scoring."""

    def __init__(self, device: str = "cuda", device_index: int = 0):
        self.ct2 = ctranslate2.Translator(
            CT2_DIR, device=device, device_index=device_index,
            compute_type=COMPUTE_TYPE,
        )
        self.sp = spm.SentencePieceProcessor(SPM_PATH)

    def translate_sentences(
        self,
        sentences: list[str],
        batch_size: int = BATCH_SIZE,
        beam_size: int = BEAM_SIZE,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_decoding_length: int = MAX_DECODING_LENGTH,
        verbose: bool = True,
    ) -> list[TranslationResult]:
        """Translate a list of sentences, returning text + confidence for each.

        Batches are sorted by length for minimal padding, then results
        are reordered to match input order.
        """
        if not sentences:
            return []

        # Tokenize
        all_tokens = []
        for s in sentences:
            toks = self.sp.encode(s, out_type=str)
            if len(toks) > max_input_length:
                toks = toks[:max_input_length]
            all_tokens.append(toks)

        results = [None] * len(sentences)
        total_batches = (len(all_tokens) + batch_size - 1) // batch_size
        t0 = time.time()

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_tokens))

            # Sort by length for efficient batching
            batch_indices = list(range(start, end))
            batch_indices.sort(key=lambda i: len(all_tokens[i]))
            batch_tokens = [all_tokens[i] for i in batch_indices]

            ct2_results = self.ct2.translate_batch(
                batch_tokens,
                beam_size=beam_size,
                max_decoding_length=max_decoding_length,
                return_scores=True,
            )

            for local_idx, global_idx in enumerate(batch_indices):
                hyp = ct2_results[local_idx]
                translated_tokens = hyp.hypotheses[0]
                translated_text = self.sp.decode(translated_tokens)

                # Extract per-token scores if available
                token_scores = []
                if hasattr(hyp, 'scores') and hyp.scores:
                    # scores[0] is the score for the best hypothesis
                    # It's the cumulative log-prob; we approximate per-token
                    score = float(hyp.scores[0])
                    num_tokens = max(len(translated_tokens), 1)
                    # Distribute score evenly (approximation)
                    token_scores = [score / num_tokens] * num_tokens

                confidence = compute_sentence_confidence(token_scores)

                results[global_idx] = TranslationResult(
                    text=translated_text,
                    confidence=confidence,
                    token_scores=token_scores,
                )

            if verbose and ((batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1):
                elapsed = time.time() - t0
                done = end
                sps = done / elapsed if elapsed > 0 else 0
                eta = (len(all_tokens) - done) / sps if sps > 0 else 0
                print(f"  [{done}/{len(all_tokens)}] {sps:.0f} sents/sec, ETA {eta:.0f}s", flush=True)

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd translation-pipeline-v2 && python -m pytest test_translator.py -v
```

Expected: all PASS (these tests don't need GPU).

- [ ] **Step 5: Commit**

```bash
git add translation-pipeline-v2/translator.py translation-pipeline-v2/test_translator.py
git commit -m "feat(translation-v2): add translator with confidence scoring"
```

---

### Task 5: Create postprocessor.py

**Files:**
- Create: `translation-pipeline-v2/postprocessor.py`

- [ ] **Step 1: Write tests**

Create `translation-pipeline-v2/test_postprocessor.py`:

```python
"""Tests for postprocessor module."""

from postprocessor import process_document
from sentence_splitter import split_document
from translator import TranslationResult


def _make_result(text: str, confidence: float = 0.8) -> TranslationResult:
    return TranslationResult(text=text, confidence=confidence, token_scores=[])


def test_process_clean_document():
    text = "The sun is shining brightly today. Birds are singing in the trees."
    doc = split_document(text)
    non_skipped = [s for s in doc["sentences"] if not s["skipped"]]
    translations = {
        s["sent_idx"]: _make_result(f"KK: {s['text']}")
        for s in non_skipped
    }
    result = process_document(doc, translations, text)
    assert result["text_kk"] != ""
    assert result["sentences_translated"] == 2
    assert result["sentences_skipped"] == 0
    assert result["confidence_mean"] > 0


def test_process_bad_translation_filtered():
    text = "Hello world today is great. Another fine sentence here."
    doc = split_document(text)
    non_skipped = [s for s in doc["sentences"] if not s["skipped"]]
    # First translation is copy of original (will be filtered)
    translations = {
        non_skipped[0]["sent_idx"]: _make_result(non_skipped[0]["text"], 0.9),
        non_skipped[1]["sent_idx"]: _make_result("Тағы бір жақсы сөйлем.", 0.8),
    }
    result = process_document(doc, translations, text)
    assert result["sentences_skipped"] >= 1  # copy-through filtered


def test_process_all_skipped():
    text = "x=1"  # too short, will be skipped at split level
    doc = split_document(text)
    result = process_document(doc, {}, text)
    assert result["text_kk"] == ""
    assert result["confidence_mean"] == 0.0
    assert result["confidence_min"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd translation-pipeline-v2 && python -m pytest test_postprocessor.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement postprocessor.py**

```python
"""Post-translation quality checks, filtering, and document reassembly."""

from filters import is_translation_bad
from sentence_splitter import reassemble_document
from translator import TranslationResult


def process_document(
    doc: dict,
    translations: dict[int, TranslationResult],
    original_text: str,
) -> dict:
    """Apply post-translation filters and reassemble document.

    Args:
        doc: output of split_document()
        translations: {sent_idx: TranslationResult} for non-skipped sentences
        original_text: original English text

    Returns dict with:
        text_kk, confidence_mean, confidence_min,
        sentences_total, sentences_translated, sentences_skipped
    """
    sentences = doc["sentences"]
    real_sentences = [s for s in sentences if not s.get("is_paragraph_break")]
    total = len(real_sentences)

    # Apply post-translation filters
    accepted_translations: dict[int, str] = {}
    confidences: list[float] = []
    skipped = 0
    pre_skipped = 0

    for sent in real_sentences:
        sent_idx = sent["sent_idx"]

        if sent["skipped"]:
            # Already skipped at pre-translation stage
            pre_skipped += 1
            skipped += 1
            continue

        if sent_idx not in translations:
            skipped += 1
            continue

        tr = translations[sent_idx]

        # Post-translation filter
        if is_translation_bad(sent["text"], tr.text):
            skipped += 1
            continue

        accepted_translations[sent_idx] = tr.text
        confidences.append(tr.confidence)

    # Reassemble
    text_kk = reassemble_document(doc, accepted_translations)
    text_kk = text_kk.strip()

    # Compute aggregate confidence
    if confidences:
        confidence_mean = sum(confidences) / len(confidences)
        confidence_min = min(confidences)
    else:
        confidence_mean = 0.0
        confidence_min = 0.0

    translated_count = len(accepted_translations)

    return {
        "text_kk": text_kk,
        "confidence_mean": round(confidence_mean, 6),
        "confidence_min": round(confidence_min, 6),
        "sentences_total": total,
        "sentences_translated": translated_count,
        "sentences_skipped": skipped,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd translation-pipeline-v2 && python -m pytest test_postprocessor.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add translation-pipeline-v2/postprocessor.py translation-pipeline-v2/test_postprocessor.py
git commit -m "feat(translation-v2): add postprocessor with quality filters and reassembly"
```

---

### Task 6: Create pipeline.py orchestrator

**Files:**
- Create: `translation-pipeline-v2/pipeline.py`

- [ ] **Step 1: Implement pipeline.py**

```python
#!/usr/bin/env python3
"""
FineWeb-Edu EN→KK translation pipeline v2 — orchestrator.

Streams source dataset in 1M-doc chunks, translates sentence-by-sentence
with confidence scoring, applies quality filters, uploads to HF Hub.

Usage:
    # Smoke test (10 rows, CPU)
    python pipeline.py --smoke-test

    # Single GPU, first 2 chunks
    python pipeline.py --num-gpus 1 --start-chunk 0 --end-chunk 2

    # Multi-GPU, auto-resume
    python pipeline.py --num-gpus 2 --start-chunk auto
"""

import argparse
import glob as globmod
import os
import re
import time
from itertools import islice
from multiprocessing import Process, Queue

import xxhash
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from config import (
    SOURCE_DATASET,
    SOURCE_CONFIG,
    ROWS_PER_CHUNK,
    HF_REPO,
    BATCH_SIZE,
    BEAM_SIZE,
    MAX_INPUT_LENGTH,
    MAX_DECODING_LENGTH,
    CHECKPOINT_EVERY,
)
from sentence_splitter import split_document
from translator import Translator, TranslationResult
from postprocessor import process_document

BASE = os.path.dirname(os.path.abspath(__file__))


def content_hash(text: str) -> str:
    """Compute xxhash of text for deduplication."""
    return xxhash.xxh64_hexdigest(text)


def shard_filename(chunk_idx: int) -> str:
    return f"data/sample-10BT-{chunk_idx:05d}.parquet"


def shard_exists(api: HfApi, chunk_idx: int) -> bool:
    try:
        files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
        return shard_filename(chunk_idx) in files
    except Exception:
        return False


def find_first_missing_chunk(api: HfApi) -> int:
    try:
        files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    except Exception:
        return 0
    idx = 0
    while shard_filename(idx) in files:
        idx += 1
    return idx


def process_rows(
    rows: list[dict],
    translator: "Translator",
    verbose: bool = True,
) -> list[dict]:
    """Process a batch of rows: split → translate → postprocess.

    Returns list of output dicts ready for parquet.
    """
    # Step 1: Split all documents into sentences
    docs = []
    all_sentences = []  # (global_sent_idx, doc_idx, sent_idx_in_doc)
    sentence_texts = []

    for doc_idx, row in enumerate(rows):
        text = row.get("text", "")
        doc = split_document(text)
        docs.append((row, doc))

        for sent in doc["sentences"]:
            if not sent["skipped"]:
                all_sentences.append((len(sentence_texts), doc_idx, sent["sent_idx"]))
                sentence_texts.append(sent["text"])

    if verbose:
        print(f"  Split {len(rows)} docs → {len(sentence_texts)} sentences to translate", flush=True)

    # Step 2: Translate all sentences in batch
    if sentence_texts:
        translation_results = translator.translate_sentences(
            sentence_texts, verbose=verbose,
        )
    else:
        translation_results = []

    # Step 3: Map translations back to documents and postprocess
    # Build per-doc translation maps
    doc_translations: dict[int, dict[int, TranslationResult]] = {}
    for (global_idx, doc_idx, sent_idx), tr_result in zip(all_sentences, translation_results):
        doc_translations.setdefault(doc_idx, {})[sent_idx] = tr_result

    # Step 4: Postprocess each document
    output_rows = []
    for doc_idx, (row, doc) in enumerate(docs):
        text_en = row.get("text", "")
        original_id = row.get("id", "")

        translations = doc_translations.get(doc_idx, {})
        result = process_document(doc, translations, text_en)

        output_rows.append({
            "original_id": original_id,
            "content_hash": content_hash(text_en),
            "text_en": text_en,
            "text_kk": result["text_kk"],
            "confidence_mean": result["confidence_mean"],
            "confidence_min": result["confidence_min"],
            "sentences_total": result["sentences_total"],
            "sentences_translated": result["sentences_translated"],
            "sentences_skipped": result["sentences_skipped"],
        })

    return output_rows


def worker_process_chunk(
    gpu_id: int,
    rows: list[dict],
    output_path: str,
    checkpoint_every: int,
    result_queue: Queue,
):
    """Worker process for one GPU. Translates rows and saves to parquet."""
    print(f"[GPU:{gpu_id}] Loading translator...", flush=True)
    translator = Translator(device="cuda", device_index=gpu_id)

    all_output = []
    t_start = time.time()

    for chunk_start in range(0, len(rows), checkpoint_every):
        chunk_end = min(chunk_start + checkpoint_every, len(rows))
        chunk_rows = rows[chunk_start:chunk_end]

        print(f"[GPU:{gpu_id}] Processing rows {chunk_start}–{chunk_end}...", flush=True)
        output = process_rows(chunk_rows, translator)
        all_output.extend(output)

        # Checkpoint
        if chunk_end < len(rows):
            ckpt_path = output_path.replace(".parquet", f"_gpu{gpu_id}_ckpt_{chunk_end}.parquet")
            Dataset.from_list(all_output).to_parquet(ckpt_path)
            print(f"[GPU:{gpu_id}] Checkpoint at {chunk_end}: {len(all_output)} rows", flush=True)

    elapsed = time.time() - t_start
    print(f"[GPU:{gpu_id}] Done: {len(all_output)} rows in {elapsed:.1f}s", flush=True)

    final_path = output_path.replace(".parquet", f"_gpu{gpu_id}.parquet")
    Dataset.from_list(all_output).to_parquet(final_path)
    result_queue.put((gpu_id, final_path, len(all_output)))


def process_chunk(chunk_idx: int, rows: list[dict], args, api: HfApi):
    """Process one 1M-row chunk: translate, merge GPUs, upload."""
    print(f"\n{'='*60}", flush=True)
    print(f"CHUNK {chunk_idx}: {len(rows)} rows", flush=True)
    print(f"{'='*60}", flush=True)

    chunk_output = os.path.join(BASE, f"chunk_{chunk_idx:05d}.parquet")

    if args.num_gpus == 1:
        result_queue = Queue()
        worker_process_chunk(0, rows, chunk_output, args.checkpoint_every, result_queue)
        _, final_path, count = result_queue.get()
        merged_path = chunk_output
        os.rename(final_path, merged_path)
    else:
        chunk_size = len(rows) // args.num_gpus
        result_queue = Queue()
        processes = []

        for gpu_id in range(args.num_gpus):
            start = gpu_id * chunk_size
            end = start + chunk_size if gpu_id < args.num_gpus - 1 else len(rows)
            gpu_rows = rows[start:end]

            p = Process(
                target=worker_process_chunk,
                args=(gpu_id, gpu_rows, chunk_output, args.checkpoint_every, result_queue),
            )
            p.start()
            processes.append(p)

        results = []
        for _ in processes:
            results.append(result_queue.get())
        for p in processes:
            p.join()

        # Merge GPU outputs
        all_rows = []
        for gpu_id, final_path, count in sorted(results):
            ds = Dataset.from_parquet(final_path)
            all_rows.extend(ds.to_list())
            os.remove(final_path)
            print(f"  GPU:{gpu_id} → {count} rows", flush=True)

        merged_path = chunk_output
        Dataset.from_list(all_rows).to_parquet(merged_path)

        # Cleanup checkpoints
        for f in globmod.glob(chunk_output.replace(".parquet", "_gpu*")):
            os.remove(f)

    # Upload
    remote_path = shard_filename(chunk_idx)
    print(f"Uploading {merged_path} → {HF_REPO}/{remote_path}...", flush=True)
    api.upload_file(
        path_or_fileobj=merged_path,
        path_in_repo=remote_path,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"Chunk {chunk_idx} uploaded.", flush=True)
    os.remove(merged_path)


def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu EN→KK pipeline v2")
    parser.add_argument("--start-chunk", type=str, default="0")
    parser.add_argument("--end-chunk", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Translate 10 rows on CPU, print results")
    args = parser.parse_args()

    if args.smoke_test:
        print("Smoke test: 10 rows, CPU...", flush=True)
        ds = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)
        rows = list(islice(ds, 10))
        translator = Translator(device="cpu", device_index=0)
        results = process_rows(rows, translator)
        for r in results:
            print(f"\n--- {r['original_id']} (conf={r['confidence_mean']:.3f}/{r['confidence_min']:.3f}, "
                  f"translated={r['sentences_translated']}/{r['sentences_total']}) ---")
            print(f"EN: {r['text_en'][:200]}...")
            print(f"KK: {r['text_kk'][:200]}...")
        return

    api = HfApi()
    t_total = time.time()

    if args.start_chunk == "auto":
        start_chunk = find_first_missing_chunk(api)
        print(f"Auto-detected start chunk: {start_chunk}", flush=True)
    else:
        start_chunk = int(args.start_chunk)

    end_chunk = args.end_chunk if args.end_chunk is not None else start_chunk + 10

    print(f"Pipeline v2: chunks {start_chunk}–{end_chunk - 1}", flush=True)
    print(f"Repo: {HF_REPO}", flush=True)
    print(f"Source: {SOURCE_DATASET} ({SOURCE_CONFIG})", flush=True)
    print(f"GPUs: {args.num_gpus}", flush=True)

    chunks_todo = []
    for chunk_idx in range(start_chunk, end_chunk):
        if shard_exists(api, chunk_idx):
            print(f"Chunk {chunk_idx}: exists, skipping.", flush=True)
        else:
            chunks_todo.append(chunk_idx)

    if not chunks_todo:
        print("All chunks already uploaded!", flush=True)
        return

    first_offset = chunks_todo[0] * ROWS_PER_CHUNK
    last_end = (chunks_todo[-1] + 1) * ROWS_PER_CHUNK

    print(f"\nStreaming rows {first_offset}–{last_end}...", flush=True)
    ds_stream = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)
    stream_iter = iter(islice(ds_stream, first_offset, last_end))

    for chunk_idx in range(chunks_todo[0], chunks_todo[-1] + 1):
        rows = list(islice(stream_iter, ROWS_PER_CHUNK))
        if chunk_idx not in chunks_todo:
            continue
        process_chunk(chunk_idx, rows, args, api)

    elapsed = time.time() - t_total
    print(f"\nALL DONE in {elapsed / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add translation-pipeline-v2/pipeline.py
git commit -m "feat(translation-v2): add pipeline orchestrator with chunks, resume, HF upload"
```

---

### Task 7: Create test_pipeline.py for E2E testing on RunPod

**Files:**
- Create: `translation-pipeline-v2/test_pipeline.py`

- [ ] **Step 1: Implement test_pipeline.py**

```python
#!/usr/bin/env python3
"""
E2E test script for translation pipeline v2.

Runs the full pipeline on random rows from FineWeb-Edu and outputs
a detailed report for manual inspection and threshold tuning.

Usage:
    # Test on 100 random rows (GPU)
    python test_pipeline.py --num-rows 100

    # Validation run on 1000 rows
    python test_pipeline.py --num-rows 1000 --seed 42

    # CPU mode for quick testing
    python test_pipeline.py --num-rows 10 --cpu
"""

import argparse
import random
import time
from itertools import islice

from datasets import load_dataset

from config import (
    SOURCE_DATASET,
    SOURCE_CONFIG,
    TEST_SAMPLE_SIZE,
    VALIDATION_SAMPLE_SIZE,
)
from sentence_splitter import split_document
from translator import Translator
from postprocessor import process_document
from pipeline import process_rows


def sample_random_rows(num_rows: int, seed: int) -> list[dict]:
    """Sample random rows from FineWeb-Edu by picking random offsets."""
    rng = random.Random(seed)
    # FineWeb-Edu sample-10BT has ~9.7M rows
    # Sample random offsets and take 1 row each
    total_approx = 9_700_000
    offsets = sorted(rng.sample(range(total_approx), min(num_rows, total_approx)))

    print(f"Sampling {num_rows} random rows (seed={seed})...", flush=True)
    ds = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)

    rows = []
    ds_iter = iter(ds)
    prev_offset = 0
    for offset in offsets:
        # Skip to offset
        skip = offset - prev_offset
        for _ in range(skip):
            next(ds_iter, None)
        row = next(ds_iter, None)
        if row:
            rows.append(row)
        prev_offset = offset + 1

    print(f"Loaded {len(rows)} rows.", flush=True)
    return rows


def print_report(results: list[dict]):
    """Print detailed analysis report for manual inspection."""
    total = len(results)
    translated = sum(1 for r in results if r["text_kk"])
    empty = total - translated

    all_conf_mean = [r["confidence_mean"] for r in results if r["confidence_mean"] > 0]
    all_conf_min = [r["confidence_min"] for r in results if r["confidence_min"] > 0]

    total_sents = sum(r["sentences_total"] for r in results)
    total_translated_sents = sum(r["sentences_translated"] for r in results)
    total_skipped_sents = sum(r["sentences_skipped"] for r in results)

    print(f"\n{'='*70}")
    print(f"TRANSLATION PIPELINE V2 — TEST REPORT")
    print(f"{'='*70}")

    print(f"\n## Document-level stats")
    print(f"  Total documents:       {total}")
    print(f"  With translation:      {translated} ({translated/total*100:.1f}%)")
    print(f"  Empty (text_kk=''):    {empty} ({empty/total*100:.1f}%)")

    print(f"\n## Sentence-level stats")
    print(f"  Total sentences:       {total_sents}")
    print(f"  Translated:            {total_translated_sents} ({total_translated_sents/max(total_sents,1)*100:.1f}%)")
    print(f"  Skipped:               {total_skipped_sents} ({total_skipped_sents/max(total_sents,1)*100:.1f}%)")

    if all_conf_mean:
        print(f"\n## Confidence distribution")
        print(f"  Mean confidence — avg: {sum(all_conf_mean)/len(all_conf_mean):.4f}, "
              f"min: {min(all_conf_mean):.4f}, max: {max(all_conf_mean):.4f}")
        print(f"  Min confidence  — avg: {sum(all_conf_min)/len(all_conf_min):.4f}, "
              f"min: {min(all_conf_min):.4f}, max: {max(all_conf_min):.4f}")

        # Histogram buckets
        buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        print(f"\n  Confidence_mean histogram:")
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            count = sum(1 for c in all_conf_mean if lo <= c < hi)
            bar = "#" * count
            print(f"    [{lo:.1f}, {hi:.1f}): {count:4d} {bar}")

    # Examples: good translations (high confidence)
    good = sorted([r for r in results if r["confidence_mean"] > 0], key=lambda r: -r["confidence_mean"])
    print(f"\n## Top 5 best translations (highest confidence)")
    for r in good[:5]:
        print(f"\n  conf={r['confidence_mean']:.4f} | translated={r['sentences_translated']}/{r['sentences_total']}")
        print(f"  EN: {r['text_en'][:150]}...")
        print(f"  KK: {r['text_kk'][:150]}...")

    # Examples: borderline (low but non-zero confidence)
    borderline = sorted([r for r in results if 0 < r["confidence_mean"] < 0.5], key=lambda r: r["confidence_mean"])
    if borderline:
        print(f"\n## Top 5 borderline translations (lowest confidence)")
        for r in borderline[:5]:
            print(f"\n  conf={r['confidence_mean']:.4f} | translated={r['sentences_translated']}/{r['sentences_total']}")
            print(f"  EN: {r['text_en'][:150]}...")
            print(f"  KK: {r['text_kk'][:150]}...")

    # Examples: empty translations
    empties = [r for r in results if not r["text_kk"]]
    if empties:
        print(f"\n## Examples of fully skipped documents ({len(empties)} total)")
        for r in empties[:5]:
            print(f"\n  total_sents={r['sentences_total']} | skipped={r['sentences_skipped']}")
            print(f"  EN: {r['text_en'][:200]}...")

    # Documents with high skip ratio
    partial = [r for r in results if r["sentences_skipped"] > 0 and r["text_kk"]]
    if partial:
        partial.sort(key=lambda r: r["sentences_skipped"] / max(r["sentences_total"], 1), reverse=True)
        print(f"\n## Top 5 partially translated documents (most skips)")
        for r in partial[:5]:
            skip_pct = r["sentences_skipped"] / max(r["sentences_total"], 1) * 100
            print(f"\n  {r['sentences_skipped']}/{r['sentences_total']} skipped ({skip_pct:.0f}%) | conf={r['confidence_mean']:.4f}")
            print(f"  EN: {r['text_en'][:150]}...")
            print(f"  KK: {r['text_kk'][:150]}...")


def main():
    parser = argparse.ArgumentParser(description="E2E test for translation pipeline v2")
    parser.add_argument("--num-rows", type=int, default=TEST_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--sequential", action="store_true",
                        help="Load rows sequentially from start (faster than random sampling)")
    args = parser.parse_args()

    t_start = time.time()

    if args.sequential:
        print(f"Loading first {args.num_rows} rows sequentially...", flush=True)
        ds = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split="train", streaming=True)
        rows = list(islice(ds, args.num_rows))
    else:
        rows = sample_random_rows(args.num_rows, args.seed)

    device = "cpu" if args.cpu else "cuda"
    print(f"Initializing translator (device={device})...", flush=True)
    translator = Translator(device=device, device_index=0)

    print(f"Processing {len(rows)} rows...", flush=True)
    results = process_rows(rows, translator)

    elapsed = time.time() - t_start
    print(f"\nProcessing completed in {elapsed:.1f}s")

    print_report(results)

    # Save raw results for further analysis
    from datasets import Dataset
    ds_out = Dataset.from_list(results)
    output_path = f"test_results_n{args.num_rows}_s{args.seed}.parquet"
    ds_out.to_parquet(output_path)
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add translation-pipeline-v2/test_pipeline.py
git commit -m "feat(translation-v2): add E2E test script with detailed reporting"
```

---

### Task 8: Create README.md with full documentation

**Files:**
- Create: `translation-pipeline-v2/README.md`

- [ ] **Step 1: Write README**

```markdown
# Translation Pipeline v2 — FineWeb-Edu EN→KK

Modular pipeline for translating FineWeb-Edu from English to Kazakh with sentence-level confidence scoring and quality filtering.

## Architecture

```
FineWeb-Edu (streaming) → sentence_splitter → translator → postprocessor → parquet → HF Hub
```

### Modules

| Module | Purpose |
|--------|---------|
| `config.py` | All constants and thresholds |
| `sentence_splitter.py` | Split docs into sentences, pre-filter noise |
| `translator.py` | CTranslate2 translation with per-sentence confidence |
| `postprocessor.py` | Post-translation quality checks, document reassembly |
| `filters.py` | Sentence-level pre/post filter functions |
| `pipeline.py` | Orchestrator: chunking, multi-GPU, resume, HF upload |
| `test_pipeline.py` | E2E test on random rows with detailed report |

## Translation Model

- **HPLT/translate-en-kk-v2.0-hplt_opus** (Marian NMT)
- CTranslate2 runtime, FP16, greedy decoding
- Model files: `model_ct2/` (CTranslate2 format), `model_cache/model.en-kk.spm`

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `original_id` | str | Original document ID from FineWeb-Edu |
| `content_hash` | str | xxhash of `text_en` (for incremental dedup) |
| `text_en` | str | Original English text |
| `text_kk` | str | Kazakh translation (empty if all sentences failed) |
| `confidence_mean` | float | Mean per-sentence confidence (0-1) |
| `confidence_min` | float | Min per-sentence confidence (0-1) |
| `sentences_total` | int | Total sentences in document |
| `sentences_translated` | int | Successfully translated sentences |
| `sentences_skipped` | int | Skipped sentences (pre + post filter) |

## Filtering Scheme

### Pre-translation (per sentence)

Sentence is skipped if:
- **>30% non-alpha characters** (math formulas, code, special chars)
- **<3 words** (too short to be meaningful)
- **>512 characters** (exceeds CTranslate2 input limit)

### Post-translation (per sentence)

Translation is discarded if:
- **>90% character similarity** with original (copy-through, not translated)
- **3+ n-gram repeats** (model looping: "және және және...")
- **Length ratio >3x or <0.3x** vs original (hallucination or truncation)
- **Disproportionate special characters** vs original (garbage output)

### Document level

- Rows are NEVER deleted — every document stays with `text_en` and `original_id`
- `text_kk` may be empty (all sentences skipped) or partial
- Users filter by `confidence_mean` / `confidence_min` threshold

## Incremental Translation

Designed for incremental translation across FineWeb-Edu splits:

1. Translate `sample-10BT` → upload as split `sample-10BT`
2. Later: load `content_hash` set from sample-10BT
3. Stream `sample-100BT`, skip already-translated rows
4. Translate remaining ~90BT, upload as `sample-100BT`

## Quick Start

```bash
# Install dependencies
pip install ctranslate2 sentencepiece datasets huggingface_hub xxhash

# Download and convert model (from en-kk-translation-pipeline/)
python ../en-kk-translation-pipeline/download_translation_model.py
# Copy model_ct2/ and model_cache/ here

# Run tests
python -m pytest test_filters.py test_sentence_splitter.py test_translator.py test_postprocessor.py -v

# E2E test (CPU, 10 rows)
python test_pipeline.py --num-rows 10 --cpu --sequential

# E2E test (GPU, 100 random rows)
python test_pipeline.py --num-rows 100

# Full pipeline (1 GPU, first chunk)
python pipeline.py --num-gpus 1 --start-chunk 0 --end-chunk 1

# Full pipeline (2 GPUs, auto-resume)
python pipeline.py --num-gpus 2 --start-chunk auto
```

## HuggingFace

- **Repo:** `stukenov/sozkz-fineweb-edu-kk-v2`
- **Split:** `sample-10BT` (mirrors FineWeb-Edu split naming)
```

- [ ] **Step 2: Commit**

```bash
git add translation-pipeline-v2/README.md
git commit -m "docs(translation-v2): add README with architecture, filtering scheme, and usage"
```

---

### Task 9: Model setup script

**Files:**
- Create: `translation-pipeline-v2/setup_model.sh`

- [ ] **Step 1: Create setup script for model download and CT2 conversion**

```bash
#!/usr/bin/env bash
# Download HPLT EN→KK translation model and convert to CTranslate2 format.
# Run this once before using the pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="HPLT/translate-en-kk-v2.0-hplt_opus"
CACHE_DIR="model_cache"
CT2_DIR="model_ct2"

echo "=== Step 1: Download model ==="
if [ -d "$CACHE_DIR" ] && [ -f "$CACHE_DIR/model.en-kk.spm" ]; then
    echo "Model cache already exists, skipping download."
else
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_NAME', local_dir='$CACHE_DIR')
print('Model downloaded to $CACHE_DIR')
"
fi

echo "=== Step 2: Convert to CTranslate2 ==="
if [ -d "$CT2_DIR" ] && [ -f "$CT2_DIR/model.bin" ]; then
    echo "CT2 model already exists, skipping conversion."
else
    ct2-opus-mt-converter --model_dir "$CACHE_DIR" --output_dir "$CT2_DIR"
    echo "CT2 model saved to $CT2_DIR"
fi

echo "=== Done ==="
echo "Model ready. You can now run the pipeline."
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x translation-pipeline-v2/setup_model.sh
git add translation-pipeline-v2/setup_model.sh
git commit -m "feat(translation-v2): add model download and CT2 conversion script"
```

---

### Task 10: E2E validation on RunPod

This task is manual — run on a cheap RunPod GPU.

- [ ] **Step 1: Launch RunPod instance**

Rent a cheap GPU (RTX 3060/3070, ~$0.15-0.25/hr) with pytorch image.

- [ ] **Step 2: Copy pipeline and setup**

```bash
# On RunPod
git clone <repo> && cd slm/translation-pipeline-v2
pip install ctranslate2 sentencepiece datasets huggingface_hub xxhash
bash setup_model.sh
```

- [ ] **Step 3: Run unit tests**

```bash
python -m pytest test_filters.py test_sentence_splitter.py test_translator.py test_postprocessor.py -v
```

Expected: all PASS.

- [ ] **Step 4: Run E2E test on 10 rows (quick sanity)**

```bash
python test_pipeline.py --num-rows 10 --sequential
```

Check output makes sense.

- [ ] **Step 5: Run E2E test on 100 random rows**

```bash
python test_pipeline.py --num-rows 100 --seed 42
```

Review the report:
- Are good sentences being kept?
- Are noisy sentences being filtered?
- Are translations reasonable?
- Is confidence scoring meaningful?

- [ ] **Step 6: Adjust thresholds if needed**

Edit `config.py` based on observations. Re-run with different seed:

```bash
python test_pipeline.py --num-rows 100 --seed 123
```

- [ ] **Step 7: Validation run on 1000 rows**

```bash
python test_pipeline.py --num-rows 1000 --seed 42
```

Confirm thresholds are stable. Commit any threshold changes.

- [ ] **Step 8: Destroy RunPod instance**

Only after confirming everything works and thresholds are committed.
