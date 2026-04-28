"""Split documents into sentences with paragraph preservation and pre-filtering."""

import re
from filters import is_noisy_sentence

SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ\d"])')


def split_document(text: str) -> dict:
    """Split document into sentences preserving paragraph structure.

    Returns dict:
        sentences: list of {text, para_idx, sent_idx, skipped, skip_reason, is_paragraph_break}
        paragraph_count: int
    """
    paragraphs = text.split('\n')
    sentences = []
    global_sent_idx = 0

    for para_idx, para in enumerate(paragraphs):
        para_stripped = para.strip()
        if not para_stripped:
            sentences.append({
                "text": "",
                "para_idx": para_idx,
                "sent_idx": global_sent_idx,
                "skipped": True,
                "skip_reason": "empty_paragraph",
                "is_paragraph_break": True,
            })
            global_sent_idx += 1
            continue

        parts = SENT_RE.split(para_stripped)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            skipped = is_noisy_sentence(part)
            sentences.append({
                "text": part,
                "para_idx": para_idx,
                "sent_idx": global_sent_idx,
                "skipped": skipped,
                "skip_reason": "noisy" if skipped else "",
                "is_paragraph_break": False,
            })
            global_sent_idx += 1

    return {
        "sentences": sentences,
        "paragraph_count": len(paragraphs),
    }


def reassemble_document(doc: dict, translations: dict[int, str]) -> str:
    """Reassemble translated sentences into document preserving paragraph breaks.

    Args:
        doc: output of split_document
        translations: {sent_idx: translated_text} for non-skipped sentences

    Returns: reassembled translated text
    """
    paragraphs: dict[int, list[str]] = {}

    for sent in doc["sentences"]:
        para_idx = sent["para_idx"]
        paragraphs.setdefault(para_idx, [])

        if sent.get("is_paragraph_break"):
            continue

        sent_idx = sent["sent_idx"]
        if sent_idx in translations:
            paragraphs[para_idx].append(translations[sent_idx])

    result = []
    for para_idx in sorted(paragraphs.keys()):
        sents = paragraphs[para_idx]
        if sents:
            result.append(" ".join(sents))
        else:
            result.append("")

    return "\n".join(result).strip()
