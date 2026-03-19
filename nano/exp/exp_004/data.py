"""
exp_003/data.py — Датасет: арифметика на словах (KK↔EN)

Генерирует тройки: (KK вопрос, EN вопрос, EN ответ, KK ответ)
Пример: "екі қосу үш нешеге тең?" → "what is two plus three?" → "two plus three is five" → "екі қосу үш — бес"
"""

import random

# Числа 0-10
NUM_KK = ["нөл", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз", "он"]
NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

OPS = {
    "қосу":  ("plus",  lambda a, b: a + b),
    "алу":   ("minus", lambda a, b: a - b),
}

def make_dataset(max_num=10):
    """Генерирует все валидные примеры арифметики."""
    samples = []
    for op_kk, (op_en, fn) in OPS.items():
        for a in range(max_num + 1):
            for b in range(max_num + 1):
                result = fn(a, b)
                if result < 0 or result > 10:
                    continue
                kk_q = f"{NUM_KK[a]} {op_kk} {NUM_KK[b]} нешеге тең ?"
                en_q = f"what is {NUM_EN[a]} {op_en} {NUM_EN[b]} ?"
                en_a = f"{NUM_EN[a]} {op_en} {NUM_EN[b]} is {NUM_EN[result]} ."
                kk_a = f"{NUM_KK[a]} {op_kk} {NUM_KK[b]} — {NUM_KK[result]} ."
                samples.append((kk_q, en_q, en_a, kk_a))
    return samples


def build_vocabs(samples):
    """Строит словари слов для каждого языка."""
    PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
    specials = [PAD, BOS, EOS]

    kk_words = set()
    en_words = set()
    for kk_q, en_q, en_a, kk_a in samples:
        kk_words.update(kk_q.split())
        kk_words.update(kk_a.split())
        en_words.update(en_q.split())
        en_words.update(en_a.split())

    kk_vocab = specials + sorted(kk_words)
    en_vocab = specials + sorted(en_words)

    kk_w2i = {w: i for i, w in enumerate(kk_vocab)}
    kk_i2w = {i: w for w, i in kk_w2i.items()}
    en_w2i = {w: i for i, w in enumerate(en_vocab)}
    en_i2w = {i: w for w, i in en_w2i.items()}

    return (kk_vocab, kk_w2i, kk_i2w), (en_vocab, en_w2i, en_i2w)


def encode_seq(text, w2i, add_bos=True, add_eos=True):
    ids = []
    if add_bos:
        ids.append(w2i["<bos>"])
    ids.extend(w2i[w] for w in text.split())
    if add_eos:
        ids.append(w2i["<eos>"])
    return ids


def decode_seq(ids, i2w):
    words = []
    for i in ids:
        w = i2w.get(i, "?")
        if w == "<eos>":
            break
        if w not in ("<pad>", "<bos>"):
            words.append(w)
    return " ".join(words)


def pad_batch(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]


if __name__ == "__main__":
    samples = make_dataset()
    (kk_vocab, kk_w2i, _), (en_vocab, en_w2i, _) = build_vocabs(samples)
    print(f"Примеров: {len(samples)}")
    print(f"KK словарь: {len(kk_vocab)} слов: {kk_vocab}")
    print(f"EN словарь: {len(en_vocab)} слов: {en_vocab}")
    print()
    for s in random.sample(samples, 3):
        print(f"  KK вопрос: {s[0]}")
        print(f"  EN вопрос: {s[1]}")
        print(f"  EN ответ:  {s[2]}")
        print(f"  KK ответ:  {s[3]}")
        print()
