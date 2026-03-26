"""Tests for translator module (CPU-safe, no GPU needed)."""

import math
from translator import compute_sentence_confidence, TranslationResult


def test_confidence_from_scores():
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
    assert len(r.token_scores) == 2


def test_confidence_high_scores():
    # Scores close to 0 → high confidence (close to 1)
    scores = [-0.01, -0.02, -0.01]
    conf = compute_sentence_confidence(scores)
    assert conf > 0.95


def test_confidence_low_scores():
    # Very negative scores → low confidence
    scores = [-3.0, -4.0, -5.0]
    conf = compute_sentence_confidence(scores)
    assert conf < 0.05
