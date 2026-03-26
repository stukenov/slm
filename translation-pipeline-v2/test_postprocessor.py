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
    # Use realistic Kazakh-like translations (different from English)
    kk_texts = ["Бүгін күн жарқырай түсіп тұр.", "Құстар ағаштарда сайрап жатыр."]
    translations = {
        s["sent_idx"]: _make_result(kk_texts[i])
        for i, s in enumerate(non_skipped)
    }
    result = process_document(doc, translations, text)
    assert result["text_kk"] != ""
    assert result["sentences_translated"] > 0
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
    assert result["sentences_skipped"] >= 1


def test_process_all_skipped():
    text = "x=1"  # too short, will be skipped at split level
    doc = split_document(text)
    result = process_document(doc, {}, text)
    assert result["text_kk"] == ""
    assert result["confidence_mean"] == 0.0
    assert result["confidence_min"] == 0.0


def test_process_preserves_counts():
    text = "First sentence here today.\nSecond sentence here today.\nThird sentence here today."
    doc = split_document(text)
    non_skipped = [s for s in doc["sentences"] if not s["skipped"]]
    translations = {
        s["sent_idx"]: _make_result(f"Перевод {i}", 0.7 + i * 0.05)
        for i, s in enumerate(non_skipped)
    }
    result = process_document(doc, translations, text)
    assert result["sentences_total"] == len(non_skipped)
    assert result["sentences_translated"] + result["sentences_skipped"] == result["sentences_total"]
    assert result["confidence_min"] <= result["confidence_mean"]
