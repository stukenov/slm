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
    # 8 alpha + 3 spaces + 3 special = 14 chars, non-space=11, special=3, 3/11=27%
    text = "abc def ghi" + "!@#"
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
    translated = "Бұл өте ұзын мәтін " * 20
    assert is_translation_bad(original, translated) is True


def test_translation_too_short():
    original = "This is a fairly long sentence with many words in it"
    translated = "Қыс"
    assert is_translation_bad(original, translated) is True
