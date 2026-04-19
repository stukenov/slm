from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

TAG_KEEP = "$KEEP"
TAG_DELETE = "$DELETE"


def extract_edit_tags(src_words: list[str], tgt_words: list[str]) -> list[str]:
    m, n = len(src_words), len(tgt_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_words[i - 1] == tgt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    alignment: list[tuple[int, int]] = []
    i, j = m, n
    while i > 0 and j > 0:
        if src_words[i - 1] == tgt_words[j - 1]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    alignment.reverse()

    aligned_src = {a[0] for a in alignment}
    aligned_tgt = {a[1] for a in alignment}

    tags: list[str] = [TAG_DELETE] * m

    src_to_tgt: dict[int, int] = {a[0]: a[1] for a in alignment}

    for si, ti in alignment:
        tags[si] = TAG_KEEP

    unaligned_tgt = sorted(set(range(n)) - aligned_tgt)

    for si in range(m):
        if si in aligned_src:
            continue
        best_ti = None
        for ti in unaligned_tgt:
            if best_ti is None:
                best_ti = ti
                break
        if best_ti is not None:
            tags[si] = f"$REPLACE_{tgt_words[best_ti]}"
            unaligned_tgt.remove(best_ti)

    for ti in reversed(unaligned_tgt):
        best_si = None
        for si in range(m):
            ti_aligned = src_to_tgt.get(si)
            if ti_aligned is not None and ti_aligned < ti:
                best_si = si
        if best_si is None:
            best_si = 0
        current_tag = tags[best_si]
        if current_tag == TAG_KEEP:
            tags[best_si] = f"$APPEND_{tgt_words[ti]}"
        elif current_tag.startswith("$APPEND_"):
            tags[best_si] = f"{current_tag}_{tgt_words[ti]}"

    return tags


def build_tag_vocab(
    examples: list[tuple[list[str], list[str]]],
    top_k: int = 2000,
) -> list[str]:
    counter: Counter = Counter()
    for src_words, tgt_words in examples:
        tags = extract_edit_tags(src_words, tgt_words)
        counter.update(tags)

    base_tags = [TAG_KEEP, TAG_DELETE]
    vocab = list(base_tags)
    for tag, _ in counter.most_common():
        if tag not in vocab:
            vocab.append(tag)
        if len(vocab) >= top_k:
            break
    return vocab


def apply_tags(src_words: list[str], tags: list[str]) -> list[str]:
    result = []
    for word, tag in zip(src_words, tags):
        if tag == TAG_KEEP:
            result.append(word)
        elif tag == TAG_DELETE:
            continue
        elif tag.startswith("$REPLACE_"):
            replacement = tag[len("$REPLACE_"):]
            result.append(replacement)
        elif tag.startswith("$APPEND_"):
            result.append(word)
            appended = tag[len("$APPEND_"):].split("_")
            result.extend(appended)
    return result


def tags_to_jsonl(
    examples: list[tuple[str, str]],
    output_path: Path,
    tag_vocab: list[str] | None = None,
) -> list[str]:
    tag2id = {t: i for i, t in enumerate(tag_vocab)} if tag_vocab else {}
    rows = []
    with open(output_path, "w") as f:
        for src, tgt in examples:
            src_words = src.split()
            tgt_words = tgt.split()
            tags = extract_edit_tags(src_words, tgt_words)
            if tag_vocab:
                tags = [t if t in tag2id else TAG_KEEP for t in tags]
            row = {"input": src, "target": tgt, "tags": tags}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    return tag_vocab or list(set(t for r in rows for t in r["tags"]))
