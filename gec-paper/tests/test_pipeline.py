from gecpaper.pipeline import DualPipeline


def _mock_tagger(words):
    tags = []
    for w in words:
        if w == "мектепка":
            tags.append(("$REPLACE_мектепке", 0.95))
        else:
            tags.append(("$KEEP", 0.99))
    return tags


def _mock_seq2seq(text):
    return text.replace("мектепка", "мектепке")


def test_tagger_only():
    pipe = DualPipeline(tagger_fn=_mock_tagger, mode="tagger_only")
    result = pipe.correct("Ол мектепка барды")
    assert result == "Ол мектепке барды"


def test_seq2seq_only():
    pipe = DualPipeline(seq2seq_fn=_mock_seq2seq, mode="seq2seq_only")
    result = pipe.correct("Ол мектепка барды")
    assert result == "Ол мектепке барды"


def test_cascade():
    pipe = DualPipeline(
        tagger_fn=_mock_tagger,
        seq2seq_fn=_mock_seq2seq,
        mode="cascade",
    )
    result = pipe.correct("Ол мектепка барды")
    assert result == "Ол мектепке барды"


def test_cascade_no_tagger_change():
    def noop_tagger(words):
        return [("$KEEP", 0.99)] * len(words)

    def fix_seq2seq(text):
        return text.replace("бар ды", "барды")

    pipe = DualPipeline(
        tagger_fn=noop_tagger,
        seq2seq_fn=fix_seq2seq,
        mode="cascade",
    )
    result = pipe.correct("Ол мектепке бар ды")
    assert result == "Ол мектепке барды"


def test_morph_fn_strips_pipes():
    def morph(text):
        return "Ол мектеп|ке бар|ды"

    def seq2seq(text):
        return text

    pipe = DualPipeline(seq2seq_fn=seq2seq, morph_fn=morph, mode="seq2seq_only")
    result = pipe.correct("Ол мектепке барды")
    assert "|" not in result


def test_low_confidence_keeps_original():
    def low_conf_tagger(words):
        return [("$REPLACE_x", 0.5)] * len(words)

    pipe = DualPipeline(tagger_fn=low_conf_tagger, mode="tagger_only", tagger_threshold=0.9)
    result = pipe.correct("Ол барды")
    assert result == "Ол барды"
