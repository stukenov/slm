from gecpaper.models.edit_tagger import (
    apply_tags,
    build_tag_vocab,
    extract_edit_tags,
)


def test_keep_identical():
    tags = extract_edit_tags(["мен", "бардым"], ["мен", "бардым"])
    assert tags == ["$KEEP", "$KEEP"]


def test_replace_suffix():
    tags = extract_edit_tags(["мектепка"], ["мектепке"])
    assert len(tags) == 1
    assert tags[0].startswith("$REPLACE_")


def test_delete():
    tags = extract_edit_tags(["мен", "өте", "бардым"], ["мен", "бардым"])
    assert "$DELETE" in tags


def test_append():
    tags = extract_edit_tags(["мен", "бардым"], ["мен", "кеше", "бардым"])
    found_append = any(t.startswith("$APPEND_") for t in tags)
    assert found_append


def test_build_vocab_top_k():
    examples = [
        (["a", "b"], ["a", "c"]),
        (["a", "b"], ["a", "c"]),
        (["x"], ["y"]),
    ]
    vocab = build_tag_vocab(examples, top_k=5)
    assert "$KEEP" in vocab
    assert "$DELETE" in vocab
    assert len(vocab) <= 5


def test_apply_tags_keep():
    result = apply_tags(["мен", "бардым"], ["$KEEP", "$KEEP"])
    assert result == ["мен", "бардым"]


def test_apply_tags_replace():
    result = apply_tags(["мектепка"], ["$REPLACE_мектепке"])
    assert result == ["мектепке"]


def test_apply_tags_delete():
    result = apply_tags(["мен", "өте", "бардым"], ["$KEEP", "$DELETE", "$KEEP"])
    assert result == ["мен", "бардым"]


def test_apply_tags_append():
    result = apply_tags(["мен", "бардым"], ["$APPEND_кеше", "$KEEP"])
    assert result == ["мен", "кеше", "бардым"]
