"""
exp_009/data.py — Датасет: KK+RU+EN арифметика + Code + Error
Тот же что exp_008, но M1/M3 будут реальные HPLT модели.
"""

NUM_KK = ["нөл", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз", "он"]
NUM_RU = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять"]
NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def make_math_samples(max_num=10):
    samples = []
    ops = {
        "kk": {"қосу": ("plus", lambda a, b: a+b), "алу": ("minus", lambda a, b: a-b)},
        "ru": {"плюс": ("plus", lambda a, b: a+b), "минус": ("minus", lambda a, b: a-b)},
        "en": {"plus": ("plus", lambda a, b: a+b), "minus": ("minus", lambda a, b: a-b)},
    }
    for lang, lang_ops in ops.items():
        nums = {"kk": NUM_KK, "ru": NUM_RU, "en": NUM_EN}[lang]
        for op_word, (op_en, fn) in lang_ops.items():
            for a in range(max_num + 1):
                for b in range(max_num + 1):
                    r = fn(a, b)
                    if r < 0 or r > 10:
                        continue
                    if lang == "kk":
                        src = f"{nums[a]} {op_word} {nums[b]} нешеге тең ?"
                        answer = f"{nums[a]} {op_word} {nums[b]} — {nums[r]} ."
                        plan = "<lang_kk> <translate> <math> <translate_back>"
                    elif lang == "ru":
                        src = f"сколько будет {nums[a]} {op_word} {nums[b]} ?"
                        answer = f"{nums[a]} {op_word} {nums[b]} равно {nums[r]} ."
                        plan = "<lang_ru> <translate> <math> <translate_back>"
                    else:
                        src = f"what is {NUM_EN[a]} {op_word} {NUM_EN[b]} ?"
                        answer = f"{NUM_EN[a]} {op_word} {NUM_EN[b]} is {NUM_EN[r]} ."
                        plan = "<lang_en> <math>"

                    en_q = f"what is {NUM_EN[a]} {op_en} {NUM_EN[b]} ?"
                    en_a = f"{NUM_EN[a]} {op_en} {NUM_EN[b]} is {NUM_EN[r]} ."
                    samples.append({
                        "domain": "math", "lang": lang,
                        "src": src, "plan": plan,
                        "en_q": en_q, "en_a": en_a, "answer": answer,
                    })
    return samples


def make_code_samples(max_num=10):
    samples = []
    ops = {"+": lambda a, b: a+b, "-": lambda a, b: a-b}
    for op_sym, fn in ops.items():
        for a in range(max_num + 1):
            for b in range(max_num + 1):
                r = fn(a, b)
                if r < 0 or r > 20:
                    continue
                for lang, fmt in [("py", "print ( {a} {op} {b} )"), ("js", "console.log ( {a} {op} {b} )")]:
                    src = fmt.format(a=a, op=op_sym, b=b)
                    samples.append({
                        "domain": "code", "lang": lang,
                        "src": src, "plan": f"<lang_{lang}> <code>",
                        "code_input": src, "code_output": str(r), "answer": str(r),
                    })
    return samples


def make_error_samples():
    errors = {
        "kk": (["сәлем қалайсың ?", "бүгін ауа райы қандай ?", "менің атым кім ?",
                 "қазақстан қайда ?", "сағат неше ?", "сен кімсің ?",
                 "маған көмектес", "жақсы күн", "рахмет саған", "кітап оқы"],
                "қате : сұраныс түсініксіз"),
        "ru": (["привет как дела ?", "какая сегодня погода ?", "как тебя зовут ?",
                 "где находится казахстан ?", "который час ?", "ты кто ?",
                 "помоги мне", "хороший день", "спасибо тебе", "читай книгу"],
                "ошибка : запрос непонятен"),
        "en": (["hello how are you ?", "what is the weather today ?", "what is your name ?",
                 "where is kazakhstan ?", "what time is it ?", "who are you ?",
                 "help me please", "nice day today", "thank you very much", "read a book"],
                "error : unknown request"),
    }
    samples = []
    for lang, (phrases, msg) in errors.items():
        for p in phrases:
            samples.append({
                "domain": "error", "lang": lang, "src": p, "plan": "<error>",
                "err_input": p, "err_output": msg, "answer": msg,
            })
    return samples


def make_dataset():
    return make_math_samples() + make_code_samples() + make_error_samples()

def pad_batch(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]
