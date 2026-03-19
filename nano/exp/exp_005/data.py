"""
exp_005/data.py — Двуязычный датасет: KK+RU арифметика

Генерирует четвёрки: (lang, src_question, en_question, en_answer, src_answer)
"""

NUM_KK = ["нөл", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз", "он"]
NUM_RU = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять"]
NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

OPS = {
    "kk": {"қосу": ("plus", lambda a, b: a + b), "алу": ("minus", lambda a, b: a - b)},
    "ru": {"плюс": ("plus", lambda a, b: a + b), "минус": ("minus", lambda a, b: a - b)},
}

def make_dataset(max_num=10):
    samples = []
    for lang, ops in OPS.items():
        nums_src = NUM_KK if lang == "kk" else NUM_RU
        for op_src, (op_en, fn) in ops.items():
            for a in range(max_num + 1):
                for b in range(max_num + 1):
                    result = fn(a, b)
                    if result < 0 or result > 10:
                        continue
                    if lang == "kk":
                        src_q = f"{nums_src[a]} {op_src} {nums_src[b]} нешеге тең ?"
                        src_a = f"{nums_src[a]} {op_src} {nums_src[b]} — {nums_src[result]} ."
                    else:
                        src_q = f"сколько будет {nums_src[a]} {op_src} {nums_src[b]} ?"
                        src_a = f"{nums_src[a]} {op_src} {nums_src[b]} равно {nums_src[result]} ."

                    en_q = f"what is {NUM_EN[a]} {op_en} {NUM_EN[b]} ?"
                    en_a = f"{NUM_EN[a]} {op_en} {NUM_EN[b]} is {NUM_EN[result]} ."
                    samples.append((lang, src_q, en_q, en_a, src_a))
    return samples


def pad_batch(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]


if __name__ == "__main__":
    samples = make_dataset()
    kk = [s for s in samples if s[0] == "kk"]
    ru = [s for s in samples if s[0] == "ru"]
    print(f"Всего: {len(samples)} (KK: {len(kk)}, RU: {len(ru)})")
    print()
    for s in samples[:2] + [s for s in samples if s[0] == "ru"][:2]:
        print(f"  [{s[0]}] {s[1]}")
        print(f"       EN q: {s[2]}")
        print(f"       EN a: {s[3]}")
        print(f"       ответ: {s[4]}")
        print()
