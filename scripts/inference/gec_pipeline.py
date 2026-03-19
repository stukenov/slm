"""
Multi-pass GEC pipeline with iterative tag-based correction.

Each sentence passes through tags in order (easy → hard).
If a tag finds an error, it retries (max 3), then restarts from tag[0]
to re-verify all previous corrections still hold.
"""

TAG_ORDER = [
    "қате",           # typos (easiest)
    "сингармонизм",   # vowel harmony
    "септік",         # case suffixes
    "тәуелдік",       # possessive
    "жіктік",         # personal endings
    "шылау",          # postpositions
    "көптік",         # plural
    "болымсыз",       # negation
    "грамматика",     # general grammar (catch-all, last)
]

MAX_RETRIES_PER_TAG = 3
MAX_FULL_RESTARTS = 3  # prevent infinite loops


def run_pipeline(text, correct_fn, verbose=False):
    """
    Run multi-pass GEC pipeline on a single sentence.

    Args:
        text: input sentence
        correct_fn: callable(tag, text) -> corrected_text
        verbose: if True, yield log entries

    Returns:
        (corrected_text, log) where log is list of step dicts
    """
    log = []
    current = text.strip()
    full_restarts = 0

    tag_idx = 0
    while tag_idx < len(TAG_ORDER):
        tag = TAG_ORDER[tag_idx]

        # Retry this tag up to MAX_RETRIES_PER_TAG times
        tag_changed = False
        for retry in range(MAX_RETRIES_PER_TAG):
            result = correct_fn(tag, current)
            result = result.strip()

            step = {
                "tag": tag,
                "retry": retry,
                "input": current,
                "output": result,
                "changed": result != current,
            }
            log.append(step)

            if result == current:
                # No change — this tag is done
                break

            # Found a correction
            current = result
            tag_changed = True

            if verbose:
                print(f"  <{tag}> retry={retry}: changed")

        if tag_changed:
            # A correction was made — restart from tag[0] to re-verify
            full_restarts += 1
            if full_restarts > MAX_FULL_RESTARTS:
                # Safety: prevent infinite loops
                if verbose:
                    print(f"  MAX_FULL_RESTARTS reached ({MAX_FULL_RESTARTS}), stopping")
                break
            if verbose:
                print(f"  Restarting from tag[0] (restart #{full_restarts})")
            tag_idx = 0
            continue

        # No change for this tag — move to next
        tag_idx += 1

    return current, log


def format_log(log):
    """Pretty-print pipeline log."""
    lines = []
    for step in log:
        tag = step["tag"]
        status = "CHANGED" if step["changed"] else "ok"
        if step["changed"]:
            lines.append(f"  <{tag}> [{step['retry']}] {status}: {step['input']!r} → {step['output']!r}")
        else:
            lines.append(f"  <{tag}> [{step['retry']}] {status}")
    return "\n".join(lines)
