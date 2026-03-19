"""
exp_006/data.py — Трёхъязычный датасет (KK+RU+EN) + планы действий для Think

Каждый пример: (lang, src_q, en_q, en_a, src_a, plan)
plan — последовательность action-токенов для Think.
"""

NUM_KK = ["нөл", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз", "он"]
NUM_RU = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять"]
NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

OPS = {
    "kk": {"қосу": ("plus", lambda a, b: a + b), "алу": ("minus", lambda a, b: a - b)},
    "ru": {"плюс": ("plus", lambda a, b: a + b), "минус": ("minus", lambda a, b: a - b)},
    "en": {"plus": ("plus", lambda a, b: a + b), "minus": ("minus", lambda a, b: a - b)},
}

# Планы действий
PLAN_KK = "<lang_kk> <translate> <math> <translate_back>"
PLAN_RU = "<lang_ru> <translate> <math> <translate_back>"
PLAN_EN = "<lang_en> <math>"


def make_dataset(max_num=10):
    samples = []
    for lang, ops in OPS.items():
        nums_src = {"kk": NUM_KK, "ru": NUM_RU, "en": NUM_EN}[lang]
        plan = {"kk": PLAN_KK, "ru": PLAN_RU, "en": PLAN_EN}[lang]
        for op_src, (op_en, fn) in ops.items():
            for a in range(max_num + 1):
                for b in range(max_num + 1):
                    result = fn(a, b)
                    if result < 0 or result > 10:
                        continue
                    if lang == "kk":
                        src_q = f"{nums_src[a]} {op_src} {nums_src[b]} нешеге тең ?"
                        src_a = f"{nums_src[a]} {op_src} {nums_src[b]} — {nums_src[result]} ."
                    elif lang == "ru":
                        src_q = f"сколько будет {nums_src[a]} {op_src} {nums_src[b]} ?"
                        src_a = f"{nums_src[a]} {op_src} {nums_src[b]} равно {nums_src[result]} ."
                    else:
                        src_q = f"what is {nums_src[a]} {op_src} {nums_src[b]} ?"
                        src_a = f"{nums_src[a]} {op_src} {nums_src[b]} is {nums_src[result]} ."

                    en_q = f"what is {NUM_EN[a]} {op_en} {NUM_EN[b]} ?"
                    en_a = f"{NUM_EN[a]} {op_en} {NUM_EN[b]} is {NUM_EN[result]} ."
                    samples.append((lang, src_q, en_q, en_a, src_a, plan))
    return samples


def pad_batch(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]


if __name__ == "__main__":
    samples = make_dataset()
    for lang in ["kk", "ru", "en"]:
        ex = next(s for s in samples if s[0] == lang)
        print(f"[{lang}] {ex[1]}")
        print(f"     plan: {ex[5]}")
        print(f"     en_q: {ex[2]} → en_a: {ex[3]} → src_a: {ex[4]}\n")
    print(f"Всего: {len(samples)}")
