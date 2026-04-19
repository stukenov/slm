from gecpaper.scoring.metrics import (
    char_error_rate,
    word_level_edits,
    compute_word_f05,
    compute_gleu,
    multi_ref_word_f05,
    multi_ref_cer,
)


def test_cer_identical():
    assert char_error_rate("hello", "hello") == 0.0


def test_cer_completely_different():
    assert char_error_rate("abc", "xyz") == 1.0


def test_cer_one_edit():
    assert abs(char_error_rate("helo", "hello") - 0.2) < 0.01


def test_word_edits_no_change():
    assert word_level_edits("a b c", "a b c") == set()


def test_word_edits_substitution():
    edits = word_level_edits("a b c", "a x c")
    assert len(edits) > 0


def test_f05_perfect():
    source = "He go to school"
    target = "He goes to school"
    prediction = "He goes to school"
    result = compute_word_f05(source, prediction, target)
    assert result["f05"] == 1.0


def test_f05_no_correction():
    source = "He go to school"
    target = "He goes to school"
    prediction = "He go to school"
    result = compute_word_f05(source, prediction, target)
    assert result["f05"] == 0.0


def test_multi_ref_f05_takes_max():
    source = "He go to school"
    refs = ["He goes to school", "He went to school"]
    prediction = "He went to school"
    score = multi_ref_word_f05(source, prediction, refs)
    assert score == 1.0


def test_multi_ref_cer_takes_min():
    refs = ["hello world", "hi world"]
    assert multi_ref_cer("hello world", refs) == 0.0


def test_gleu_perfect():
    score = compute_gleu("a b c", "a x c", "a x c")
    assert score > 0.9
