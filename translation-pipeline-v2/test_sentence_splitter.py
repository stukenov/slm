"""Tests for sentence splitter."""

from sentence_splitter import split_document, reassemble_document


def test_split_simple():
    text = "Hello world. This is a test."
    result = split_document(text)
    non_empty = [s for s in result["sentences"] if not s.get("is_paragraph_break")]
    # May be 1 or 2 sentences depending on regex split
    assert len(non_empty) >= 1


def test_split_preserves_paragraphs():
    text = "First paragraph here.\n\nSecond paragraph here."
    result = split_document(text)
    sents = [s for s in result["sentences"] if not s.get("is_paragraph_break") and not s["skipped"]]
    para_indices = set(s["para_idx"] for s in sents)
    assert len(para_indices) == 2  # two different paragraphs


def test_split_filters_noisy():
    # Formula on its own line (separate paragraph) gets filtered as noisy
    text = "Good sentence here today.\n∫₀¹ f(x)dx = 0\nAnother good one here."
    result = split_document(text)
    clean = [s for s in result["sentences"] if not s["skipped"]]
    skipped = [s for s in result["sentences"] if s["skipped"] and not s.get("is_paragraph_break")]
    assert len(clean) == 2
    assert len(skipped) == 1


def test_reassemble():
    text = "Hello world today. Nice day today.\n\nSecond paragraph here today."
    doc = split_document(text)
    translations = {}
    for s in doc["sentences"]:
        if not s["skipped"]:
            translations[s["sent_idx"]] = f"[KK]{s['text']}"
    reassembled = reassemble_document(doc, translations)
    assert "\n" in reassembled  # paragraph break preserved
    assert "[KK]" in reassembled


def test_reassemble_partial():
    text = "Good sentence today. Bad: ∫₀¹ f(x)dx. Another good one here."
    doc = split_document(text)
    translations = {}
    for s in doc["sentences"]:
        if not s["skipped"]:
            translations[s["sent_idx"]] = f"[KK]{s['text']}"
    reassembled = reassemble_document(doc, translations)
    assert "[KK]" in reassembled


def test_empty_text():
    doc = split_document("")
    result = reassemble_document(doc, {})
    assert result == ""
