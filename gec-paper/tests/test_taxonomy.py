from gecpaper.taxonomy.schema import (
    Level1, Level2, Level3,
    ErrorAnnotation,
    ALL_L2_FOR_L1,
    ALL_L3_FOR_L2,
    parse_annotation,
)


def test_level1_has_three_categories():
    assert len(Level1) == 3
    assert Level1.ORTHOGRAPHY.value == "orthography"
    assert Level1.MORPHOSYNTAX.value == "morphosyntax"
    assert Level1.SYNTAX_DISCOURSE.value == "syntax_discourse"


def test_level2_parent_mapping():
    l2_ortho = ALL_L2_FOR_L1[Level1.ORTHOGRAPHY]
    assert Level2.SPELLING in l2_ortho
    assert Level2.VOWEL_HARMONY in l2_ortho
    assert Level2.SPACING in l2_ortho
    assert Level2.PUNCTUATION in l2_ortho
    assert Level2.CASE not in l2_ortho


def test_level3_morphosyntax_has_most_subtypes():
    morph_l3_count = sum(
        len(ALL_L3_FOR_L2.get(l2, []))
        for l2 in ALL_L2_FOR_L1[Level1.MORPHOSYNTAX]
    )
    ortho_l3_count = sum(
        len(ALL_L3_FOR_L2.get(l2, []))
        for l2 in ALL_L2_FOR_L1[Level1.ORTHOGRAPHY]
    )
    assert morph_l3_count > ortho_l3_count
    assert len(ALL_L3_FOR_L2[Level2.CASE]) == 7


def test_error_annotation_creation():
    ann = ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.CASE, l3=Level3.DATIVE)
    assert ann.tag == "morphosyntax/case/dative"


def test_error_annotation_no_l3():
    ann = ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPELLING)
    assert ann.tag == "orthography/spelling"


def test_parse_annotation_roundtrip():
    ann = ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.PLURAL, l3=Level3.ALLOMORPH)
    parsed = parse_annotation(ann.tag)
    assert parsed == ann


def test_all_l2_categories_count():
    total = sum(len(v) for v in ALL_L2_FOR_L1.values())
    assert total >= 15
