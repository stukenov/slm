from __future__ import annotations

from collections import Counter


def char_error_rate(prediction: str, reference: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    m, n = len(prediction), len(reference)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if prediction[i - 1] == reference[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / n


def word_level_edits(source: str, text: str) -> set[tuple]:
    src_words = source.split()
    txt_words = text.split()
    edits = set()
    m, n = len(src_words), len(txt_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_words[i - 1] == txt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    i, j = m, n
    while i > 0 and j > 0:
        if src_words[i - 1] == txt_words[j - 1]:
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            edits.add(("del", i - 1, src_words[i - 1]))
            i -= 1
        else:
            edits.add(("ins", j - 1, txt_words[j - 1]))
            j -= 1
    while i > 0:
        edits.add(("del", i - 1, src_words[i - 1]))
        i -= 1
    while j > 0:
        edits.add(("ins", j - 1, txt_words[j - 1]))
        j -= 1
    return edits


def compute_word_f05(
    source: str, prediction: str, reference: str, beta: float = 0.5,
) -> dict:
    gold_edits = word_level_edits(source, reference)
    pred_edits = word_level_edits(source, prediction)
    tp = len(gold_edits & pred_edits)
    fp = len(pred_edits - gold_edits)
    fn = len(gold_edits - pred_edits)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = beta**2 * precision + recall
    f05 = (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0
    return {"precision": precision, "recall": recall, "f05": f05, "tp": tp, "fp": fp, "fn": fn}


def _ngrams(words: list[str], n: int) -> Counter:
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def compute_gleu(source: str, prediction: str, reference: str, max_n: int = 4) -> float:
    src_words = source.split()
    pred_words = prediction.split()
    ref_words = reference.split()
    all_src_ngrams: Counter = Counter()
    all_ref_ngrams: Counter = Counter()
    all_pred_ngrams: Counter = Counter()
    for n in range(1, max_n + 1):
        all_src_ngrams += _ngrams(src_words, n)
        all_ref_ngrams += _ngrams(ref_words, n)
        all_pred_ngrams += _ngrams(pred_words, n)
    ref_diff = all_ref_ngrams - all_src_ngrams
    key_ngrams = ref_diff + all_ref_ngrams
    num = sum((all_pred_ngrams & key_ngrams).values())
    denom = max(sum(all_pred_ngrams.values()), 1)
    return num / denom


def multi_ref_word_f05(source: str, prediction: str, references: list[str]) -> float:
    return max(compute_word_f05(source, prediction, ref)["f05"] for ref in references)


def multi_ref_cer(prediction: str, references: list[str]) -> float:
    return min(char_error_rate(prediction, ref) for ref in references)


def multi_ref_exact_match(prediction: str, references: list[str]) -> bool:
    pred = prediction.strip()
    return any(pred == ref.strip() for ref in references)


def multi_ref_gleu(source: str, prediction: str, references: list[str]) -> float:
    return sum(compute_gleu(source, prediction, ref) for ref in references) / len(references)
