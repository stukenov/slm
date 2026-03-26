"""Post-translation quality checks, filtering, and document reassembly."""

from filters import is_translation_bad
from sentence_splitter import reassemble_document
from translator import TranslationResult


def process_document(
    doc: dict,
    translations: dict[int, TranslationResult],
    original_text: str,
) -> dict:
    """Apply post-translation filters and reassemble document.

    Args:
        doc: output of split_document()
        translations: {sent_idx: TranslationResult} for non-skipped sentences
        original_text: original English text

    Returns dict with:
        text_kk, confidence_mean, confidence_min,
        sentences_total, sentences_translated, sentences_skipped
    """
    sentences = doc["sentences"]
    real_sentences = [s for s in sentences if not s.get("is_paragraph_break")]
    total = len(real_sentences)

    accepted_translations: dict[int, str] = {}
    confidences: list[float] = []
    skipped = 0

    for sent in real_sentences:
        sent_idx = sent["sent_idx"]

        if sent["skipped"]:
            skipped += 1
            continue

        if sent_idx not in translations:
            skipped += 1
            continue

        tr = translations[sent_idx]

        if is_translation_bad(sent["text"], tr.text):
            skipped += 1
            continue

        accepted_translations[sent_idx] = tr.text
        confidences.append(tr.confidence)

    text_kk = reassemble_document(doc, accepted_translations)
    text_kk = text_kk.strip()

    if confidences:
        confidence_mean = sum(confidences) / len(confidences)
        confidence_min = min(confidences)
    else:
        confidence_mean = 0.0
        confidence_min = 0.0

    return {
        "text_kk": text_kk,
        "confidence_mean": round(confidence_mean, 6),
        "confidence_min": round(confidence_min, 6),
        "sentences_total": total,
        "sentences_translated": len(accepted_translations),
        "sentences_skipped": skipped,
    }
