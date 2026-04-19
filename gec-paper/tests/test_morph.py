from gecpaper.morph.apertium import is_available, segment_text, segment_word


def test_fallback_when_unavailable():
    text = "Мен мектепке бардым"
    result = segment_text(text, fallback=True)
    if not is_available():
        assert result == text
    else:
        assert len(result.split()) == len(text.split())


def test_word_count_preserved():
    text = "Ол кітап оқыды"
    result = segment_text(text, fallback=True)
    assert len(result.split()) == len(text.split())


def test_segment_word_returns_string():
    result = segment_word("бала")
    assert isinstance(result, str)
    assert len(result) > 0


def test_empty_input():
    assert segment_text("", fallback=True) == ""
