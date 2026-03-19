"""Tests for multi-pass GEC pipeline."""

from gec_pipeline import run_pipeline, TAG_ORDER, format_log


# --- Mock correct_fn that simulates known corrections ---

MOCK_CORRECTIONS = {
    # Tag -> {input -> output}
    "қате": {
        "Қазақстнаның": "Қазақстанның",
        "Астна": "Астана",
        "жане": "және",
    },
    "сингармонизм": {
        "Қабырғаларде": "Қабырғаларда",
        "кітапханасінда": "кітапханасында",
    },
    "шылау": {
        "турелі": "туралы",
        "ушін": "үшін",
    },
    "көптік": {
        "Студенттар": "Студенттер",
    },
    "грамматика": {
        "барды.": "бардым.",
        "барды ": "бардым ",
    },
}


def mock_correct(tag, text):
    """Simulate model corrections using lookup table."""
    corrections = MOCK_CORRECTIONS.get(tag, {})
    for pattern, replacement in corrections.items():
        if pattern in text:
            return text.replace(pattern, replacement)
    return text


# --- Tests ---

def test_no_errors():
    """Clean text should pass through all tags unchanged."""
    text = "Қазақстан Орталық Азиядағы ең ірі мемлекет."
    result, log = run_pipeline(text, mock_correct)
    assert result == text, f"Expected unchanged, got: {result}"
    # Should have exactly len(TAG_ORDER) steps (one per tag, no retries)
    assert len(log) == len(TAG_ORDER), f"Expected {len(TAG_ORDER)} steps, got {len(log)}"
    assert all(not step["changed"] for step in log)
    print("PASS: test_no_errors")


def test_single_tag_correction():
    """Single typo should be caught by <қате> and then all tags re-verified."""
    text = "Қазақстнаның астанасы Астна."
    result, log = run_pipeline(text, mock_correct, verbose=True)
    assert result == "Қазақстанның астанасы Астана.", f"Got: {result}"
    # Should have: қате(changed) + restart from tag[0] + all tags pass
    changed_steps = [s for s in log if s["changed"]]
    assert len(changed_steps) >= 1
    assert changed_steps[0]["tag"] == "қате"
    print("PASS: test_single_tag_correction")


def test_multi_tag_correction():
    """Errors across multiple tags — should catch both and re-verify."""
    text = "Қабырғаларде суреттер бейнеленген. Тіл турелі толғаныс жазды."

    def multi_correct(tag, text):
        # сингармонизм fixes қабырғаларде, шылау fixes турелі
        result = mock_correct(tag, text)
        return result

    result, log = run_pipeline(text, multi_correct, verbose=True)
    assert "Қабырғаларда" in result, f"сингармонизм not fixed: {result}"
    assert "туралы" in result, f"шылау not fixed: {result}"
    print(f"PASS: test_multi_tag_correction -> {result}")


def test_restart_after_fix():
    """After <сингармонизм> fix, pipeline restarts from <қате> to re-verify."""
    text = "Қабырғаларде суреттер."
    call_log = []

    def tracking_correct(tag, text):
        call_log.append(tag)
        return mock_correct(tag, text)

    result, log = run_pipeline(text, tracking_correct, verbose=True)
    assert result == "Қабырғаларда суреттер."

    # After сингармонизм fixes it, should restart from қате
    # Find index where сингармонизм changed, then verify қате comes after
    found_restart = False
    last_sing_idx = None
    for i, tag in enumerate(call_log):
        if tag == "сингармонизм" and i > 0:
            last_sing_idx = i
        if last_sing_idx and tag == "қате" and i > last_sing_idx:
            found_restart = True
            break
    assert found_restart, f"No restart found. Call order: {call_log}"
    print("PASS: test_restart_after_fix")


def test_max_retries():
    """Tag that always changes should be capped at MAX_RETRIES_PER_TAG."""
    call_count = {"n": 0}

    def always_changes(tag, text):
        if tag == "қате":
            call_count["n"] += 1
            return text + "x"  # always changes
        return text

    result, log = run_pipeline("test", always_changes)
    # Should not loop forever — capped by MAX_RETRIES_PER_TAG * MAX_FULL_RESTARTS
    qate_calls = sum(1 for s in log if s["tag"] == "қате")
    assert qate_calls <= 12, f"Too many қате calls: {qate_calls}"  # 3 retries * ~4 restarts
    print(f"PASS: test_max_retries (қате called {qate_calls} times)")


def test_cascading_fix():
    """Fix by one tag creates issue for earlier tag — caught on restart."""
    # Simulate: грамматика changes text, then қате catches a new typo in it
    state = {"grammar_applied": False}

    def cascading_correct(tag, text):
        if tag == "грамматика" and not state["grammar_applied"] and "кеше" in text:
            state["grammar_applied"] = True
            return text.replace("кеше дүкенге барды", "кеше дүкенге бардым")
        if tag == "қате" and "бардым" in text and "жане" in text:
            return text.replace("жане", "және")
        return text

    text = "Мен кеше дүкенге барды жане кітап алдым."
    result, log = run_pipeline(text, cascading_correct, verbose=True)
    print(f"PASS: test_cascading_fix -> {result}")


def test_pipeline_log_format():
    """Log should be human-readable."""
    text = "Қабырғаларде суреттер."
    result, log = run_pipeline(text, mock_correct)
    formatted = format_log(log)
    assert "<сингармонизм>" in formatted
    assert "CHANGED" in formatted
    print("PASS: test_pipeline_log_format")
    print(formatted)


# --- Demo with real-ish example ---

def demo_full_pipeline():
    """Show full pipeline trace."""
    text = "Қабырғаларде турелі суреттер бейнеленген."

    # This text has 2 errors: сингармонизм (де→да) and шылау (турелі→туралы)
    def demo_correct(tag, text):
        if tag == "сингармонизм" and "Қабырғаларде" in text:
            return text.replace("Қабырғаларде", "Қабырғаларда")
        if tag == "шылау" and "турелі" in text:
            return text.replace("турелі", "туралы")
        return text

    print("\n" + "=" * 70)
    print(f"INPUT:  {text}")
    result, log = run_pipeline(text, demo_correct, verbose=True)
    print(f"OUTPUT: {result}")
    print(f"Steps:  {len(log)}")
    print("\nFull log:")
    print(format_log(log))
    print("=" * 70)


if __name__ == "__main__":
    test_no_errors()
    test_single_tag_correction()
    test_multi_tag_correction()
    test_restart_after_fix()
    test_max_retries()
    test_cascading_fix()
    test_pipeline_log_format()
    demo_full_pipeline()
    print("\nAll tests passed!")
