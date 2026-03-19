"""
exp_008/data.py — Multi-tool датасет: Math + Code + Error, 5 языков (KK/RU/EN/PY/JS)

Каждый пример: (domain, src_input, plan, tool_input, tool_output, final_answer)
"""

NUM_KK = ["нөл", "бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз", "он"]
NUM_RU = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять"]
NUM_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def make_math_samples(max_num=10):
    """Арифметика на 3 языках."""
    samples = []
    ops = {
        "kk": {"қосу": ("+", lambda a, b: a+b), "алу": ("-", lambda a, b: a-b)},
        "ru": {"плюс": ("+", lambda a, b: a+b), "минус": ("-", lambda a, b: a-b)},
        "en": {"plus": ("+", lambda a, b: a+b), "minus": ("-", lambda a, b: a-b)},
    }
    for lang, lang_ops in ops.items():
        nums = {"kk": NUM_KK, "ru": NUM_RU, "en": NUM_EN}[lang]
        for op_word, (op_sym, fn) in lang_ops.items():
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

                    en_q = f"what is {NUM_EN[a]} {op_word if lang == 'en' else ('+' if fn(1,1)==2 else '-')} {NUM_EN[b]} ?"
                    # fix: use EN op words
                    en_op = "plus" if fn(1,1) == 2 else "minus"
                    en_q = f"what is {NUM_EN[a]} {en_op} {NUM_EN[b]} ?"
                    en_a = f"{NUM_EN[a]} {en_op} {NUM_EN[b]} is {NUM_EN[r]} ."

                    samples.append({
                        "domain": "math", "lang": lang,
                        "src": src, "plan": plan,
                        "en_q": en_q, "en_a": en_a,
                        "answer": answer,
                    })
    return samples


def make_code_samples(max_num=10):
    """Код на Python/JavaScript — арифметика с цифрами."""
    samples = []
    ops = {"+": lambda a, b: a+b, "-": lambda a, b: a-b}
    for op_sym, fn in ops.items():
        for a in range(max_num + 1):
            for b in range(max_num + 1):
                r = fn(a, b)
                if r < 0 or r > 20:
                    continue
                # Python
                samples.append({
                    "domain": "code", "lang": "py",
                    "src": f"print ( {a} {op_sym} {b} )",
                    "plan": "<lang_py> <code>",
                    "code_input": f"print ( {a} {op_sym} {b} )",
                    "code_output": str(r),
                    "answer": str(r),
                })
                # JavaScript
                samples.append({
                    "domain": "code", "lang": "js",
                    "src": f"console.log ( {a} {op_sym} {b} )",
                    "plan": "<lang_js> <code>",
                    "code_input": f"console.log ( {a} {op_sym} {b} )",
                    "code_output": str(r),
                    "answer": str(r),
                })
    return samples


def make_error_samples():
    """Запросы которые не попадают ни в math ни в code."""
    errors_kk = [
        "сәлем қалайсың ?",
        "бүгін ауа райы қандай ?",
        "менің атым кім ?",
        "қазақстан қайда ?",
        "сағат неше ?",
        "сен кімсің ?",
        "маған көмектес",
        "жақсы күн",
        "рахмет саған",
        "кітап оқы",
    ]
    errors_ru = [
        "привет как дела ?",
        "какая сегодня погода ?",
        "как тебя зовут ?",
        "где находится казахстан ?",
        "который час ?",
        "ты кто ?",
        "помоги мне",
        "хороший день",
        "спасибо тебе",
        "читай книгу",
    ]
    errors_en = [
        "hello how are you ?",
        "what is the weather today ?",
        "what is your name ?",
        "where is kazakhstan ?",
        "what time is it ?",
        "who are you ?",
        "help me please",
        "nice day today",
        "thank you very much",
        "read a book",
    ]
    samples = []
    for lang, phrases in [("kk", errors_kk), ("ru", errors_ru), ("en", errors_en)]:
        for phrase in phrases:
            # Ответ ошибки на языке запроса
            if lang == "kk":
                err_msg = "қате : сұраныс түсініксіз"
            elif lang == "ru":
                err_msg = "ошибка : запрос непонятен"
            else:
                err_msg = "error : unknown request"
            samples.append({
                "domain": "error", "lang": lang,
                "src": phrase, "plan": "<error>",
                "err_input": phrase,
                "err_output": err_msg,
                "answer": err_msg,
            })
    return samples


def make_dataset():
    return make_math_samples() + make_code_samples() + make_error_samples()


def pad_batch(seqs, pad_id=0):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_id] * (max_len - len(s)) for s in seqs]


if __name__ == "__main__":
    samples = make_dataset()
    math_n = sum(1 for s in samples if s["domain"] == "math")
    code_n = sum(1 for s in samples if s["domain"] == "code")
    err_n = sum(1 for s in samples if s["domain"] == "error")
    print(f"Всего: {len(samples)} (math:{math_n} code:{code_n} error:{err_n})\n")

    for d in ["math", "code", "error"]:
        ex = next(s for s in samples if s["domain"] == d)
        print(f"[{d}] {ex['src']}")
        print(f"  plan: {ex['plan']} → answer: {ex['answer']}\n")
