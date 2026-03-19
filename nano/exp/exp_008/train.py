"""
exp_008/train.py — Multi-tool Think: Math + Code + Error

6 моделей (все decoder-only):
  Think:  <think> [input] <plan> [actions] <eos>
  M1:     <kk>/<ru> [src] <en> [en_q] <eos>        (Multi→EN)
  Math:   <q> [en_q] <a> [en_a] <eos>              (арифметика)
  Code:   <py>/<js> [code] <result> [output] <eos>  (исполнитель)
  M3:     <en> [en_a] <kk>/<ru> [src_a] <eos>      (EN→Multi)
  Err:    <err> [input] <msg> [error msg] <eos>     (ошибки)

Think планы:
  <lang_kk> <translate> <math> <translate_back>
  <lang_ru> <translate> <math> <translate_back>
  <lang_en> <math>
  <lang_py> <code>
  <lang_js> <code>
  <error>
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
    "<pad>", "<eos>",
    "<think>", "<plan>",
    "<lang_kk>", "<lang_ru>", "<lang_en>", "<lang_py>", "<lang_js>",
    "<translate>", "<math>", "<translate_back>", "<code>", "<error>",
    "<kk>", "<ru>", "<en>", "<py>", "<js>",
    "<q>", "<a>", "<result>", "<err>", "<msg>",
]

all_words = set()
for s in samples:
    all_words.update(s["src"].split())
    all_words.update(s["answer"].split())
    if s["domain"] == "math":
        all_words.update(s["en_q"].split())
        all_words.update(s["en_a"].split())
    if s["domain"] == "code":
        all_words.update(s["code_output"].split())
    if s["domain"] == "error":
        all_words.update(s["err_output"].split())

vocab = SPECIALS + sorted(all_words)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
V = len(vocab)
PAD, EOS = w2i["<pad>"], w2i["<eos>"]

math_n = sum(1 for s in samples if s["domain"] == "math")
code_n = sum(1 for s in samples if s["domain"] == "code")
err_n = sum(1 for s in samples if s["domain"] == "error")
print(f"Примеров: {len(samples)} (math:{math_n} code:{code_n} error:{err_n})")
print(f"Словарь: {V}\n")

# ============================================================
# Подготовка пар
# ============================================================
def ids(text): return [w2i[w] for w in text.split()]

# Think: <think> [src] <plan> [actions] <eos>
think_pairs = []
for s in samples:
    pre = [w2i["<think>"]] + ids(s["src"])
    suf = [w2i["<plan>"]] + ids(s["plan"]) + [EOS]
    think_pairs.append((pre, suf))

# M1: <kk>/<ru> [src_q] <en> [en_q] <eos>  (math, не-EN)
m1_pairs = []
for s in samples:
    if s["domain"] != "math" or s["lang"] == "en":
        continue
    pre = [w2i[f"<{s['lang']}>"]] + ids(s["src"])
    suf = [w2i["<en>"]] + ids(s["en_q"]) + [EOS]
    m1_pairs.append((pre, suf))

# Math: <q> [en_q] <a> [en_a] <eos>
math_pairs = []
for s in samples:
    if s["domain"] != "math":
        continue
    pre = [w2i["<q>"]] + ids(s["en_q"])
    suf = [w2i["<a>"]] + ids(s["en_a"]) + [EOS]
    math_pairs.append((pre, suf))

# Code: <py>/<js> [code] <result> [output] <eos>
code_pairs = []
for s in samples:
    if s["domain"] != "code":
        continue
    pre = [w2i[f"<{s['lang']}>"]] + ids(s["code_input"])
    suf = [w2i["<result>"]] + ids(s["code_output"]) + [EOS]
    code_pairs.append((pre, suf))

# M3: <en> [en_a] <kk>/<ru> [answer] <eos>  (math, не-EN)
m3_pairs = []
for s in samples:
    if s["domain"] != "math" or s["lang"] == "en":
        continue
    pre = [w2i["<en>"]] + ids(s["en_a"])
    suf = [w2i[f"<{s['lang']}>"]] + ids(s["answer"]) + [EOS]
    m3_pairs.append((pre, suf))

# Err: <err> [src] <msg> [error msg] <eos>
err_pairs = []
for s in samples:
    if s["domain"] != "error":
        continue
    pre = [w2i["<err>"]] + ids(s["src"])
    suf = [w2i["<msg>"]] + ids(s["err_output"]) + [EOS]
    err_pairs.append((pre, suf))

print(f"Think:{len(think_pairs)} M1:{len(m1_pairs)} Math:{len(math_pairs)} "
      f"Code:{len(code_pairs)} M3:{len(m3_pairs)} Err:{len(err_pairs)}\n")

# ============================================================
# Батчи + Loss
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
    print(f"{'='*55}\n{name} ({n_p:,} params, {len(pairs)} examples)\n{'='*55}")
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
# Generation
# ============================================================
FILTER = {"<pad>", "<eos>", "<kk>", "<ru>", "<en>", "<py>", "<js>",
          "<lang_kk>", "<lang_ru>", "<lang_en>", "<lang_py>", "<lang_js>",
          "<translate>", "<math>", "<translate_back>", "<code>", "<error>",
          "<think>", "<plan>", "<q>", "<a>", "<result>", "<err>", "<msg>"}

def generate(model, prefix_ids, max_new=25):
    gen = list(prefix_ids)
    for _ in range(max_new):
        logits = model(Tensor([gen], dtype=dtypes.int32))
        nid = int(logits[0, -1].argmax().item())
        gen.append(nid)
        if nid == EOS: break
    return gen

def extract_after(gen_ids, tag):
    tid = w2i[tag]
    if tid not in gen_ids: return ""
    last = len(gen_ids) - 1 - gen_ids[::-1].index(tid)
    words = []
    for i in gen_ids[last+1:]:
        w = i2w.get(i, "?")
        if w in ("<eos>", "<pad>"): break
        words.append(w)
    return " ".join(words)

def clean(text):
    return " ".join(w for w in text.split() if w not in FILTER)

def parse_plan(gen_ids):
    pid = w2i["<plan>"]
    if pid not in gen_ids: return []
    idx = gen_ids.index(pid)
    actions = []
    for i in gen_ids[idx+1:]:
        w = i2w.get(i, "?")
        if w == "<eos>": break
        actions.append(w)
    return actions

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    Tensor.training = True

    think = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=48)
    m1 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    math_model = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    code_model = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    m3 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    err_model = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)

    log_think = train("Think", think, think_pairs, STEPS)
    log_m1 = train("M1", m1, m1_pairs, STEPS)
    log_math = train("Math", math_model, math_pairs, STEPS)
    log_code = train("Code", code_model, code_pairs, STEPS)
    log_m3 = train("M3", m3, m3_pairs, STEPS)
    log_err = train("Err", err_model, err_pairs, STEPS)

    # --- Инференс ---
    Tensor.training = False

    print(f"{'='*55}")
    print("ИНФЕРЕНС: Think → [plan] → route → execute")
    print(f"{'='*55}\n")

    test_cases = [
        # Math KK
        ("бір қосу екі нешеге тең ?", "math", "kk"),
        ("он алу бес нешеге тең ?", "math", "kk"),
        # Math RU
        ("сколько будет два плюс три ?", "math", "ru"),
        ("сколько будет десять минус семь ?", "math", "ru"),
        # Math EN
        ("what is one plus two ?", "math", "en"),
        ("what is eight minus two ?", "math", "en"),
        # Code Python
        ("print ( 3 + 4 )", "code", "py"),
        ("print ( 10 - 6 )", "code", "py"),
        # Code JS
        ("console.log ( 5 + 5 )", "code", "js"),
        ("console.log ( 9 - 3 )", "code", "js"),
        # Errors
        ("сәлем қалайсың ?", "error", "kk"),
        ("привет как дела ?", "error", "ru"),
        ("hello how are you ?", "error", "en"),
        ("менің атым кім ?", "error", "kk"),
    ]

    results = []
    correct = 0
    plan_correct = 0

    for q, expected_domain, real_lang in test_cases:
        # Think
        think_out = generate(think, [w2i["<think>"]] + ids(q) + [w2i["<plan>"]])
        actions = parse_plan(think_out)
        plan_str = " ".join(actions)

        # Determine route from plan
        detected_lang = None
        for a in actions:
            if a.startswith("<lang_"):
                detected_lang = a[6:-1]
                break

        has_translate = "<translate>" in actions
        has_math = "<math>" in actions
        has_code = "<code>" in actions
        has_error = "<error>" in actions
        has_back = "<translate_back>" in actions

        # Check plan correctness
        expected_plans = {
            ("math", "kk"): "<lang_kk> <translate> <math> <translate_back>",
            ("math", "ru"): "<lang_ru> <translate> <math> <translate_back>",
            ("math", "en"): "<lang_en> <math>",
            ("code", "py"): "<lang_py> <code>",
            ("code", "js"): "<lang_js> <code>",
            ("error", "kk"): "<error>",
            ("error", "ru"): "<error>",
            ("error", "en"): "<error>",
        }
        exp_plan = expected_plans.get((expected_domain, real_lang), "")
        p_ok = plan_str.strip() == exp_plan.strip()
        if p_ok: plan_correct += 1

        trace = {"input": q, "domain": expected_domain, "lang": real_lang,
                 "plan": plan_str, "plan_ok": p_ok}
        final = ""

        # Execute plan
        if has_error:
            # Error model
            err_out = generate(err_model, [w2i["<err>"]] + ids(q) + [w2i["<msg>"]])
            final = extract_after(err_out, "<msg>")
            trace["err_out"] = final

        elif has_code:
            # Code model
            lang_tag = f"<{detected_lang}>" if detected_lang else "<py>"
            code_out = generate(code_model,
                [w2i[lang_tag]] + ids(q) + [w2i["<result>"]])
            final = extract_after(code_out, "<result>")
            trace["code_out"] = final

        elif has_math:
            if has_translate:
                # M1: translate to EN
                m1_out = generate(m1,
                    [w2i[f"<{detected_lang}>"]] + ids(q) + [w2i["<en>"]])
                en_q = extract_after(m1_out, "<en>")
                trace["m1_out"] = en_q
            else:
                en_q = q

            # Math
            math_out = generate(math_model,
                [w2i["<q>"]] + [w2i.get(w, PAD) for w in en_q.split()] + [w2i["<a>"]])
            en_a = extract_after(math_out, "<a>")
            trace["math_out"] = en_a

            if has_back:
                # M3: translate back
                m3_out = generate(m3,
                    [w2i["<en>"]] + [w2i.get(w, PAD) for w in en_a.split()]
                    + [w2i[f"<{detected_lang}>"]])
                final = clean(extract_after(m3_out, f"<{detected_lang}>"))
                trace["m3_out"] = final
            else:
                final = en_a
        else:
            final = "???"

        trace["final"] = final

        # Check answer
        expected_answer = None
        for s in samples:
            if s["src"] == q:
                expected_answer = s["answer"]
                break
        is_ok = expected_answer and final.strip() == expected_answer.strip()
        if is_ok: correct += 1
        trace["expected"] = expected_answer
        trace["correct"] = is_ok
        results.append(trace)

        # Output
        a_mark = "OK" if is_ok else "FAIL"
        p_mark = "P:ok" if p_ok else "P:FAIL"
        domain_mark = expected_domain.upper()
        print(f"  [{a_mark}] [{p_mark}] [{domain_mark}]")
        print(f"       вход: {q}")
        print(f"       план: {plan_str}")
        if has_translate: print(f"       M1→ {trace.get('m1_out','')}")
        if has_math: print(f"       Math→ {trace.get('math_out','')}")
        if has_code: print(f"       Code→ {trace.get('code_out','')}")
        if has_back: print(f"       M3→ {final}")
        elif has_error: print(f"       Err→ {final}")
        else: print(f"       → {final}")
        if not is_ok: print(f"       exp: {expected_answer}")
        print()

    total_time = time.time() - t0
    n = len(test_cases)
    acc = correct / n
    p_acc = plan_correct / n

    print(f"Plan точность:   {plan_correct}/{n} ({100*p_acc:.0f}%)")
    print(f"Общая точность:  {correct}/{n} ({100*acc:.0f}%)")
    print(f"Время: {total_time:.1f}s")

    report = {
        "experiment": "exp_008",
        "description": "Multi-tool Think: Math + Code + Error, 6 decoder-only models",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparams": {"dim": DIM, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                        "batch": BATCH, "lr": LR, "steps": STEPS},
        "data": {"n_samples": len(samples), "vocab_size": V,
                 "math": math_n, "code": code_n, "error": err_n},
        "training": {
            "Think": log_think.summary(), "M1": log_m1.summary(),
            "Math": log_math.summary(), "Code": log_code.summary(),
            "M3": log_m3.summary(), "Err": log_err.summary(),
        },
        "inference": {"plan_accuracy": p_acc, "accuracy": acc,
                      "correct": correct, "total": n, "results": results},
        "total_time_s": round(total_time, 1),
    }
    with open(LOG_DIR / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {LOG_DIR / 'report.json'}")
