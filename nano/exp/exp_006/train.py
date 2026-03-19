"""
exp_006/train.py — Think-оркестратор (decoder-only) генерирует план действий

Пайплайн:
  Input → [Think] → генерирует план (action-токены)
  Оркестратор парсит план и выполняет:
    <lang_en> <math>                         → Math напрямую
    <lang_kk> <translate> <math> <translate_back> → M1 → Math → M3
    <lang_ru> <translate> <math> <translate_back> → M1 → Math → M3

4 модели (все decoder-only):
  Think:  <think> [input] <plan> [action tokens] <eos>
  M1:     <kk>/<ru> [src] <en> [en translation] <eos>
  Math:   <q> [en question] <a> [en answer] <eos>
  M3:     <en> [en answer] <kk>/<ru> [src answer] <eos>
"""

import json
import random
import time
from pathlib import Path
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import Adam

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, pad_batch
from model import DecoderOnlyModel

# ============================================================
DIM, N_HEADS, N_LAYERS = 64, 4, 2
BATCH, LR, STEPS = 32, 3e-3, 300
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# Логирование
# ============================================================
class Logger:
    def __init__(self, name):
        self.name, self.entries, self.t0 = name, [], time.time()
        (LOG_DIR / f"{name}.jsonl").write_text("")

    def log(self, step, loss):
        e = {"step": step, "loss": round(loss, 5), "t": round(time.time() - self.t0, 2)}
        self.entries.append(e)
        with open(LOG_DIR / f"{self.name}.jsonl", "a") as f:
            f.write(json.dumps(e) + "\n")

    def summary(self):
        return {"model": self.name, "steps": len(self.entries),
                "loss_start": self.entries[0]["loss"], "loss_end": self.entries[-1]["loss"],
                "time_s": round(time.time() - self.t0, 1)}

# ============================================================
# Данные + единый словарь
# ============================================================
samples = make_dataset()
random.shuffle(samples)

SPECIALS = [
    "<pad>", "<eos>",
    # Think
    "<think>", "<plan>",
    # Action tokens
    "<lang_kk>", "<lang_ru>", "<lang_en>", "<translate>", "<math>", "<translate_back>",
    # M1/M3
    "<kk>", "<ru>", "<en>",
    # Math
    "<q>", "<a>",
]

all_words = set()
for s in samples:
    for text in s[1:5]:
        all_words.update(text.split())

vocab = SPECIALS + sorted(all_words)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
V = len(vocab)
PAD, EOS = w2i["<pad>"], w2i["<eos>"]

print(f"Примеров: {len(samples)} (KK:{sum(1 for s in samples if s[0]=='kk')}"
      f" RU:{sum(1 for s in samples if s[0]=='ru')} EN:{sum(1 for s in samples if s[0]=='en')})")
print(f"Словарь: {V} слов\n")

# ============================================================
# Подготовка данных
# ============================================================
def ids(text):
    return [w2i[w] for w in text.split()]

# Think: <think> [input] <plan> [actions] <eos>
think_pairs = []
for s in samples:
    pre = [w2i["<think>"]] + ids(s[1])
    suf = [w2i["<plan>"]] + ids(s[5]) + [EOS]
    think_pairs.append((pre, suf))

# M1: <kk>/<ru> [src_q] <en> [en_q] <eos>  (только не-EN)
m1_pairs = []
for s in samples:
    if s[0] == "en":
        continue
    pre = [w2i[f"<{s[0]}>"]] + ids(s[1])
    suf = [w2i["<en>"]] + ids(s[2]) + [EOS]
    m1_pairs.append((pre, suf))

# Math: <q> [en_q] <a> [en_a] <eos>  (все)
math_pairs = []
for s in samples:
    pre = [w2i["<q>"]] + ids(s[2])
    suf = [w2i["<a>"]] + ids(s[3]) + [EOS]
    math_pairs.append((pre, suf))

# M3: <en> [en_a] <kk>/<ru> [src_a] <eos>  (только не-EN)
m3_pairs = []
for s in samples:
    if s[0] == "en":
        continue
    pre = [w2i["<en>"]] + ids(s[3])
    suf = [w2i[f"<{s[0]}>"]] + ids(s[4]) + [EOS]
    m3_pairs.append((pre, suf))

print(f"Think: {len(think_pairs)} | M1: {len(m1_pairs)} | Math: {len(math_pairs)} | M3: {len(m3_pairs)}\n")

# ============================================================
# Батчи (masked loss — только target часть)
# ============================================================
def make_batch(pairs, batch_size):
    idxs = random.sample(range(len(pairs)), min(batch_size, len(pairs)))
    seqs = [pairs[i][0] + pairs[i][1] for i in idxs]
    masks = [[0]*len(pairs[i][0]) + [1]*len(pairs[i][1]) for i in idxs]
    padded = pad_batch(seqs, PAD)
    padded_m = pad_batch(masks, 0)
    inp = [s[:-1] for s in padded]
    tgt = [s[1:] for s in padded]
    msk = [m[1:] for m in padded_m]
    return (Tensor(inp, dtype=dtypes.int32), Tensor(tgt, dtype=dtypes.int32),
            Tensor(msk, dtype=dtypes.float32))

# ============================================================
# Loss
# ============================================================
def masked_ce(logits, targets, mask):
    B, T, Voc = logits.shape
    lp = logits.reshape(-1, Voc).log_softmax(axis=-1)
    tf = targets.reshape(-1)
    mf = mask.reshape(-1)
    return ((-lp[Tensor.arange(lp.shape[0]), tf]) * mf).sum() / mf.sum()

# ============================================================
# Обучение
# ============================================================
def train(name, model, pairs, steps):
    logger = Logger(name)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"{'='*55}\n{name} ({n_p:,} params)\n{'='*55}")
    opt = Adam(model.parameters(), lr=LR)
    for step in range(steps):
        inp, tgt, msk = make_batch(pairs, BATCH)
        loss = masked_ce(model(inp), tgt, msk)
        opt.zero_grad(); loss.backward(); opt.step()
        lv = loss.item(); logger.log(step, lv)
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={lv:.4f}  [{logger.entries[-1]['t']:.1f}s]")
    s = logger.summary()
    print(f"  → {s['loss_start']:.3f} → {s['loss_end']:.4f} за {s['time_s']}s\n")
    return logger

# ============================================================
# Greedy generation
# ============================================================
def generate(model, prefix_ids, max_new=20):
    gen_ids = list(prefix_ids)
    for _ in range(max_new):
        logits = model(Tensor([gen_ids], dtype=dtypes.int32))
        next_id = int(logits[0, -1].argmax().item())
        gen_ids.append(next_id)
        if next_id == EOS:
            break
    return gen_ids

def extract_after_tag(gen_ids, tag):
    """Извлекает слова после последнего вхождения tag до <eos>."""
    tag_id = w2i[tag]
    if tag_id not in gen_ids:
        return ""
    last = len(gen_ids) - 1 - gen_ids[::-1].index(tag_id)
    words = []
    for i in gen_ids[last+1:]:
        w = i2w.get(i, "?")
        if w in ("<eos>", "<pad>"):
            break
        words.append(w)
    return " ".join(words)

def parse_plan(gen_ids):
    """Парсит action-токены из сгенерированного плана."""
    plan_id = w2i["<plan>"]
    if plan_id not in gen_ids:
        return []
    idx = gen_ids.index(plan_id)
    actions = []
    for i in gen_ids[idx+1:]:
        w = i2w.get(i, "?")
        if w == "<eos>":
            break
        actions.append(w)
    return actions

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    Tensor.training = True

    think = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    m1 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    math_model = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    m3 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)

    log_think = train("Think", think, think_pairs, STEPS)
    log_m1 = train("M1_multi2en", m1, m1_pairs, STEPS)
    log_math = train("Math", math_model, math_pairs, STEPS)
    log_m3 = train("M3_en2multi", m3, m3_pairs, STEPS)

    # --- Инференс ---
    Tensor.training = False

    print(f"{'='*55}")
    print("ИНФЕРЕНС: Think → [plan] → execute")
    print(f"{'='*55}\n")

    test_questions = [
        # KK (ожидаем: translate → math → translate_back)
        ("бір қосу екі нешеге тең ?", "kk"),
        ("он алу бес нешеге тең ?", "kk"),
        ("алты қосу үш нешеге тең ?", "kk"),
        # RU (ожидаем: translate → math → translate_back)
        ("сколько будет два плюс три ?", "ru"),
        ("сколько будет десять минус семь ?", "ru"),
        ("сколько будет четыре плюс пять ?", "ru"),
        # EN (ожидаем: math напрямую)
        ("what is one plus two ?", "en"),
        ("what is ten minus five ?", "en"),
        ("what is six plus three ?", "en"),
        ("what is eight minus two ?", "en"),
    ]

    results = []
    correct = 0
    plan_correct = 0

    for q, real_lang in test_questions:
        # 1. Think генерирует план
        think_prefix = [w2i["<think>"]] + ids(q) + [w2i["<plan>"]]
        think_out = generate(think, think_prefix)
        actions = parse_plan(think_out)
        plan_str = " ".join(actions)

        # Парсим язык из плана
        detected_lang = None
        for a in actions:
            if a.startswith("<lang_"):
                detected_lang = a[6:-1]  # <lang_kk> → kk
                break

        needs_translate = "<translate>" in actions
        needs_translate_back = "<translate_back>" in actions

        # Проверяем план
        if real_lang == "en":
            expected_plan = "<lang_en> <math>"
        else:
            expected_plan = f"<lang_{real_lang}> <translate> <math> <translate_back>"
        p_ok = plan_str.strip() == expected_plan.strip()
        if p_ok:
            plan_correct += 1

        trace = {
            "input": q, "real_lang": real_lang,
            "plan": plan_str, "plan_ok": p_ok,
            "detected_lang": detected_lang,
        }

        # 2. Выполняем план
        if needs_translate and detected_lang:
            # M1: перевод на EN
            m1_prefix = [w2i[f"<{detected_lang}>"]] + ids(q) + [w2i["<en>"]]
            m1_out = generate(m1, m1_prefix)
            en_q = extract_after_tag(m1_out, "<en>")
            trace["m1_out"] = en_q
        else:
            en_q = q

        # Math
        math_prefix = [w2i["<q>"]] + [w2i.get(w, PAD) for w in en_q.split()] + [w2i["<a>"]]
        math_out = generate(math_model, math_prefix)
        en_a = extract_after_tag(math_out, "<a>")
        trace["math_out"] = en_a

        if needs_translate_back and detected_lang:
            # M3: перевод обратно
            m3_prefix = [w2i["<en>"]] + [w2i.get(w, PAD) for w in en_a.split()] + [w2i[f"<{detected_lang}>"]]
            m3_out = generate(m3, m3_prefix)
            final = extract_after_tag(m3_out, f"<{detected_lang}>")
            trace["m3_out"] = final
        else:
            final = en_a

        trace["final"] = final

        # Проверяем ответ
        expected = None
        for s in samples:
            if s[1] == q:
                expected = s[4]
                break
        is_ok = expected and final.strip() == expected.strip()
        if is_ok:
            correct += 1
        trace["expected"] = expected
        trace["correct"] = is_ok
        results.append(trace)

        # Вывод
        a_mark = "OK" if is_ok else "FAIL"
        p_mark = "P:ok" if p_ok else "P:FAIL"
        print(f"  [{a_mark}] [{p_mark}] {q}")
        print(f"       план: {plan_str}")
        if needs_translate:
            print(f"       M1→ {en_q}")
        print(f"       Math→ {en_a}")
        if needs_translate_back:
            print(f"       M3→ {final}")
        else:
            print(f"       → {final}")
        if not is_ok:
            print(f"       exp: {expected}")
        print()

    total_time = time.time() - t0
    acc = correct / len(test_questions)
    p_acc = plan_correct / len(test_questions)

    print(f"Plan точность:   {plan_correct}/{len(test_questions)} ({100*p_acc:.0f}%)")
    print(f"Общая точность:  {correct}/{len(test_questions)} ({100*acc:.0f}%)")
    print(f"Время: {total_time:.1f}s")

    # Отчёт
    report = {
        "experiment": "exp_006",
        "description": "Think orchestrator (decoder-only, generates action plans) + Math + multilingual translators",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparams": {"dim": DIM, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                        "batch": BATCH, "lr": LR, "steps": STEPS},
        "data": {"n_samples": len(samples), "vocab_size": V},
        "training": {"Think": log_think.summary(), "M1": log_m1.summary(),
                     "Math": log_math.summary(), "M3": log_m3.summary()},
        "inference": {"plan_accuracy": p_acc, "accuracy": acc,
                      "correct": correct, "total": len(test_questions), "results": results},
        "total_time_s": round(total_time, 1),
    }
    with open(LOG_DIR / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {LOG_DIR / 'report.json'}")
