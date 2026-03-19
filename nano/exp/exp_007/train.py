"""
exp_007/train.py — Все 4 модели encoder-decoder (вариант A)

Think: Encoder=[input] → Decoder=[plan actions]
M1:    Encoder=<kk>/<ru> [src_q] → Decoder=[en_q]
Math:  Encoder=[en_q] → Decoder=[en_a]
M3:    Encoder=[en_a] → Decoder=<kk>/<ru> [src_a]
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
from model import EncoderDecoderModel

# ============================================================
DIM, N_HEADS, N_LAYERS = 64, 4, 2
BATCH, LR, STEPS = 32, 3e-3, 300
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

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
# Данные + словарь
# ============================================================
samples = make_dataset()
random.shuffle(samples)

SPECIALS = [
    "<pad>", "<bos>", "<eos>",
    "<lang_kk>", "<lang_ru>", "<lang_en>", "<translate>", "<math>", "<translate_back>",
    "<kk>", "<ru>",
]
all_words = set()
for s in samples:
    for text in s[1:5]:
        all_words.update(text.split())
vocab = SPECIALS + sorted(all_words)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
V = len(vocab)
PAD, BOS, EOS = w2i["<pad>"], w2i["<bos>"], w2i["<eos>"]

print(f"Примеров: {len(samples)} | Словарь: {V}\n")

# ============================================================
# Подготовка пар (src_ids, tgt_ids) для encoder-decoder
# ============================================================
def ids(text): return [w2i[w] for w in text.split()]

# Think: src=input text, tgt=plan
think_data = []
for s in samples:
    src = ids(s[1])
    tgt = ids(s[5])  # plan tokens
    think_data.append((src, tgt))

# M1: src=<lang> + src_q, tgt=en_q (только не-EN)
m1_data = []
for s in samples:
    if s[0] == "en": continue
    src = [w2i[f"<{s[0]}>"]] + ids(s[1])
    tgt = ids(s[2])
    m1_data.append((src, tgt))

# Math: src=en_q, tgt=en_a
math_data = []
for s in samples:
    src = ids(s[2])
    tgt = ids(s[3])
    math_data.append((src, tgt))

# M3: src=<lang> + en_a, tgt=src_a (только не-EN)
# Тег языка в encoder — decoder знает на какой язык переводить
m3_data = []
for s in samples:
    if s[0] == "en": continue
    src = [w2i[f"<{s[0]}>"]] + ids(s[3])
    tgt = ids(s[4])
    m3_data.append((src, tgt))

print(f"Think: {len(think_data)} | M1: {len(m1_data)} | Math: {len(math_data)} | M3: {len(m3_data)}\n")

# ============================================================
# Батчи: src → encoder, <bos>+tgt → decoder input, tgt+<eos> → decoder target
# ============================================================
def make_batch(data, batch_size):
    idxs = random.sample(range(len(data)), min(batch_size, len(data)))
    src_seqs = [data[i][0] for i in idxs]
    dec_in = [[BOS] + data[i][1] for i in idxs]
    dec_tgt = [data[i][1] + [EOS] for i in idxs]
    return (Tensor(pad_batch(src_seqs, PAD), dtype=dtypes.int32),
            Tensor(pad_batch(dec_in, PAD), dtype=dtypes.int32),
            Tensor(pad_batch(dec_tgt, PAD), dtype=dtypes.int32))

# ============================================================
def ce_loss(logits, targets):
    B, T, Voc = logits.shape
    lp = logits.reshape(-1, Voc).log_softmax(axis=-1)
    return -lp[Tensor.arange(lp.shape[0]), targets.reshape(-1)].mean()

# ============================================================
def train(name, model, data, steps):
    logger = Logger(name)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"{'='*55}\n{name} ({n_p:,} params)\n{'='*55}")
    opt = Adam(model.parameters(), lr=LR)
    for step in range(steps):
        src, dec_in, dec_tgt = make_batch(data, BATCH)
        loss = ce_loss(model(src, dec_in), dec_tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        lv = loss.item(); logger.log(step, lv)
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={lv:.4f}  [{logger.entries[-1]['t']:.1f}s]")
    s = logger.summary()
    print(f"  → {s['loss_start']:.3f} → {s['loss_end']:.4f} за {s['time_s']}s\n")
    return logger

# ============================================================
# Greedy decode для encoder-decoder
# ============================================================
def greedy_decode(model, src_ids, max_len=20):
    src_t = Tensor([src_ids], dtype=dtypes.int32)
    enc = model.encode(src_t)
    dec_ids = [BOS]
    for _ in range(max_len):
        logits = model.decode(Tensor([dec_ids], dtype=dtypes.int32), enc)
        next_id = int(logits[0, -1].argmax().item())
        dec_ids.append(next_id)
        if next_id == EOS:
            break
    return [i for i in dec_ids[1:] if i != EOS]  # skip BOS and EOS

FILTER_TOKENS = {"<pad>", "<bos>", "<eos>", "<kk>", "<ru>", "<en>",
                  "<lang_kk>", "<lang_ru>", "<lang_en>", "<translate>", "<math>", "<translate_back>"}

def ids_to_text(id_list):
    return " ".join(i2w.get(i, "?") for i in id_list if i2w.get(i, "?") not in FILTER_TOKENS)

def parse_plan(id_list):
    return [i2w.get(i, "?") for i in id_list]

# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    Tensor.training = True

    think = EncoderDecoderModel(V, V, DIM, N_HEADS, N_LAYERS)
    m1 = EncoderDecoderModel(V, V, DIM, N_HEADS, N_LAYERS)
    math_model = EncoderDecoderModel(V, V, DIM, N_HEADS, N_LAYERS)
    m3 = EncoderDecoderModel(V, V, DIM, N_HEADS, N_LAYERS)

    log_t = train("Think", think, think_data, STEPS)
    log_m1 = train("M1_multi2en", m1, m1_data, STEPS)
    log_math = train("Math", math_model, math_data, STEPS)
    log_m3 = train("M3_en2multi", m3, m3_data, STEPS)

    # --- Инференс ---
    Tensor.training = False

    print(f"{'='*55}")
    print("ИНФЕРЕНС: Think(enc-dec) → [plan] → execute")
    print(f"{'='*55}\n")

    test_questions = [
        ("бір қосу екі нешеге тең ?", "kk"),
        ("он алу бес нешеге тең ?", "kk"),
        ("алты қосу үш нешеге тең ?", "kk"),
        ("сколько будет два плюс три ?", "ru"),
        ("сколько будет десять минус семь ?", "ru"),
        ("сколько будет четыре плюс пять ?", "ru"),
        ("what is one plus two ?", "en"),
        ("what is ten minus five ?", "en"),
        ("what is six plus three ?", "en"),
        ("what is eight minus two ?", "en"),
    ]

    results = []
    correct = 0
    plan_correct = 0

    for q, real_lang in test_questions:
        # Think
        plan_ids = greedy_decode(think, ids(q))
        actions = parse_plan(plan_ids)
        plan_str = " ".join(actions)

        detected_lang = None
        for a in actions:
            if a.startswith("<lang_"):
                detected_lang = a[6:-1]
                break

        needs_translate = "<translate>" in actions
        needs_back = "<translate_back>" in actions

        if real_lang == "en":
            expected_plan = "<lang_en> <math>"
        else:
            expected_plan = f"<lang_{real_lang}> <translate> <math> <translate_back>"
        p_ok = plan_str.strip() == expected_plan.strip()
        if p_ok: plan_correct += 1

        trace = {"input": q, "real_lang": real_lang, "plan": plan_str, "plan_ok": p_ok}

        # M1
        if needs_translate and detected_lang:
            m1_src = [w2i[f"<{detected_lang}>"]] + ids(q)
            en_q_ids = greedy_decode(m1, m1_src)
            en_q = ids_to_text(en_q_ids)
            trace["m1_out"] = en_q
        else:
            en_q = q
            en_q_ids = ids(q)

        # Math
        math_out_ids = greedy_decode(math_model, [w2i.get(w, PAD) for w in en_q.split()])
        en_a = ids_to_text(math_out_ids)
        trace["math_out"] = en_a

        # M3
        if needs_back and detected_lang:
            m3_src = [w2i[f"<{detected_lang}>"]] + [w2i.get(w, PAD) for w in en_a.split()]
            m3_out_ids = greedy_decode(m3, m3_src)
            final = ids_to_text(m3_out_ids)
            trace["m3_out"] = final
        else:
            final = en_a

        trace["final"] = final

        expected = None
        for s in samples:
            if s[1] == q:
                expected = s[4]
                break
        is_ok = expected and final.strip() == expected.strip()
        if is_ok: correct += 1
        trace["expected"] = expected
        trace["correct"] = is_ok
        results.append(trace)

        a_mark = "OK" if is_ok else "FAIL"
        p_mark = "P:ok" if p_ok else "P:FAIL"
        route = "DIRECT" if not needs_translate else f"{detected_lang}→en→math→en→{detected_lang}"
        print(f"  [{a_mark}] [{p_mark}] [{route}]")
        print(f"       вход: {q}")
        if needs_translate: print(f"       M1→ {en_q}")
        print(f"       Math→ {en_a}")
        if needs_back: print(f"       M3→ {final}")
        else: print(f"       → {final}")
        if not is_ok: print(f"       exp: {expected}")
        print()

    total_time = time.time() - t0
    acc = correct / len(test_questions)
    p_acc = plan_correct / len(test_questions)

    print(f"Plan точность:   {plan_correct}/{len(test_questions)} ({100*p_acc:.0f}%)")
    print(f"Общая точность:  {correct}/{len(test_questions)} ({100*acc:.0f}%)")
    print(f"Время: {total_time:.1f}s")

    report = {
        "experiment": "exp_007",
        "description": "All encoder-decoder: Think orchestrator + Math + multilingual translators",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparams": {"dim": DIM, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                        "batch": BATCH, "lr": LR, "steps": STEPS},
        "data": {"n_samples": len(samples), "vocab_size": V},
        "training": {"Think": log_t.summary(), "M1": log_m1.summary(),
                     "Math": log_math.summary(), "M3": log_m3.summary()},
        "inference": {"plan_accuracy": p_acc, "accuracy": acc,
                      "correct": correct, "total": len(test_questions), "results": results},
        "total_time_s": round(total_time, 1),
    }
    with open(LOG_DIR / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {LOG_DIR / 'report.json'}")
