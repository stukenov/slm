from __future__ import annotations

import logging
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)


def is_available() -> bool:
    return shutil.which("apertium") is not None


def segment_word(word: str) -> str:
    if not is_available():
        return word
    try:
        result = subprocess.run(
            ["apertium", "-d", ".", "kaz-morph"],
            input=word,
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.strip()
        if not output or output.startswith("*"):
            return word
        morphemes = re.findall(r"[^<>+/]+", output)
        morphemes = [m.strip() for m in morphemes if m.strip() and not m.startswith("@")]
        if morphemes:
            return "|".join(morphemes)
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning("apertium failed for '%s': %s", word, e)
    return word


def segment_text(text: str, fallback: bool = True) -> str:
    if not is_available():
        if fallback:
            return text
        raise RuntimeError("apertium not found in PATH")

    words = text.split()
    segmented = []
    for w in words:
        segmented.append(segment_word(w))
    return " ".join(segmented)
