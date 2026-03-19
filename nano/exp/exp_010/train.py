"""
exp_010/train.py — SmolLM2-135M-Instruct + HPLT float32 + tinygrad pipeline

Math: SmolLM2-135M-Instruct (реальный LLM, transformers)
M1_kk/M3_kk: HPLT CTranslate2 float32 (без квантизации)
Think, M1_ru, Code, M3_ru, Err: наши decoder-only на tinygrad

Pipeline:
  Input → [Think] → plan
    <lang_kk> <translate> <math> <translate_back> → HPLT KK→EN → SmolLM2 → HPLT EN→KK
    <lang_ru> <translate> <math> <translate_back> → M1_ru → SmolLM2 → M3_ru
    <lang_en> <math> → SmolLM2 напрямую
    <lang_py>/<lang_js> <code> → Code
    <error> → Err
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
from adapter import TranslatorAdapter
from smollm_adapter import SmolLMAdapter

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
# Данные + словарь (для tinygrad моделей)
# ============================================================
samples = make_dataset()
random.shuffle(samples)

SPECIALS = [
    "<pad>", "<eos>",
    "<think>", "<plan>",
    "<lang_kk>", "<lang_ru>", "<lang_en>", "<lang_py>", "<lang_js>",
    "<translate>", "<math>", "<translate_back>", "<code>", "<error>",
    "<ru>", "<en>",
    "<q>", "<a>", "<result>", "<err>", "<msg>",
]

all_words = set()
for s in samples:
    for key in ["src", "answer"]:
        all_words.update(s[key].split())
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
print(f"Примеров: {len(samples)} (math:{math_n} code:{code_n} error:{err_n}) | Словарь: {V}\n")

# ============================================================
# Подготовка пар для tinygrad моделей (без Math — он теперь SmolLM2)
# ============================================================
def ids(text): return [w2i[w] for w in text.split()]

# Think
think_pairs = []
for s in samples:
    pre = [w2i["<think>"]] + ids(s["src"])
    suf = [w2i["<plan>"]] + ids(s["plan"]) + [EOS]
    think_pairs.append((pre, suf))

# M1_ru: только RU→EN
m1_ru_pairs = []
for s in samples:
    if s["domain"] != "math" or s["lang"] != "ru":
        continue
    pre = [w2i["<ru>"]] + ids(s["src"])
    suf = [w2i["<en>"]] + ids(s["en_q"]) + [EOS]
    m1_ru_pairs.append((pre, suf))

# Code
code_pairs = []
for s in samples:
    if s["domain"] != "code": continue
    tag = "<lang_py>" if s["lang"] == "py" else "<lang_js>"
    pre = [w2i[tag]] + ids(s["code_input"])
    suf = [w2i["<result>"]] + ids(s["code_output"]) + [EOS]
    code_pairs.append((pre, suf))

# M3_ru: только EN→RU
m3_ru_pairs = []
for s in samples:
    if s["domain"] != "math" or s["lang"] != "ru":
        continue
    pre = [w2i["<en>"]] + ids(s["en_a"])
    suf = [w2i["<ru>"]] + ids(s["answer"]) + [EOS]
    m3_ru_pairs.append((pre, suf))

# Err
err_pairs = []
for s in samples:
    if s["domain"] != "error": continue
    pre = [w2i["<err>"]] + ids(s["src"])
    suf = [w2i["<msg>"]] + ids(s["err_output"]) + [EOS]
    err_pairs.append((pre, suf))

print(f"Think:{len(think_pairs)} M1_ru:{len(m1_ru_pairs)} "
      f"Code:{len(code_pairs)} M3_ru:{len(m3_ru_pairs)} Err:{len(err_pairs)}")
print("M1_kk/M3_kk: HPLT CTranslate2 float32")
print("Math: SmolLM2-135M-Instruct\n")

# ============================================================
# Batch + Loss
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
# Training (tinygrad models only)
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
    print(f"  -> {s['loss_start']:.3f} -> {s['loss_end']:.4f} за {s['time_s']}s\n")
    return logger

# ============================================================
# Generation (tinygrad)
# ============================================================
FILTER = {"<pad>", "<eos>", "<ru>", "<en>",
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
    return [i2w.get(i, "?") for i in gen_ids[idx+1:] if i2w.get(i, "?") != "<eos>"]

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()

    # --- SmolLM2-135M-Instruct ---
    print("Загрузка SmolLM2-135M-Instruct...")
    smol_t = time.time()
    smollm = SmolLMAdapter(device="cpu")
    print(f"SmolLM2 загружен за {time.time()-smol_t:.1f}s\n")

    # --- HPLT адаптеры (float32) ---
    print("Загрузка HPLT CTranslate2 float32...")
    hplt_t = time.time()
    hplt_kk2en = TranslatorAdapter("kk_en")
    hplt_en2kk = TranslatorAdapter("en_kk")
    print(f"HPLT загружен за {time.time()-hplt_t:.1f}s\n")

    # --- tinygrad модели ---
    Tensor.training = True

    think = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=48)
    m1_ru = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    code_model = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    m3_ru = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    err_model = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)

    log_think = train("Think", think, think_pairs, STEPS)
    log_m1ru = train("M1_ru", m1_ru, m1_ru_pairs, STEPS)
    log_code = train("Code", code_model, code_pairs, STEPS)
    log_m3ru = train("M3_ru", m3_ru, m3_ru_pairs, STEPS)
    log_err = train("Err", err_model, err_pairs, STEPS)

    # --- Инференс ---
    Tensor.training = False

    print(f"{'='*55}")
    print("ИНФЕРЕНС: Think -> route -> execute")
    print("Math: SmolLM2-135M-Instruct (135M params)")
    print("M1/M3 KK<->EN: HPLT CTranslate2 float32")
    print("M1/M3 RU<->EN: tinygrad decoder-only")
    print(f"{'='*55}\n")

    test_cases = [
        # Math KK (HPLT + SmolLM2)
        ("бір қосу екі нешеге тең ?", "math", "kk"),
        ("он алу бес нешеге тең ?", "math", "kk"),
        ("алты қосу үш нешеге тең ?", "math", "kk"),
        # Math RU (tinygrad M1 + SmolLM2 + tinygrad M3)
        ("сколько будет два плюс три ?", "math", "ru"),
        ("сколько будет десять минус семь ?", "math", "ru"),
        # Math EN (SmolLM2 direct)
        ("what is one plus two ?", "math", "en"),
        ("what is eight minus two ?", "math", "en"),
        ("what is four plus five ?", "math", "en"),
        ("what is seven minus three ?", "math", "en"),
        # Code
        ("print ( 3 + 4 )", "code", "py"),
        ("console.log ( 5 + 5 )", "code", "js"),
        # Error
        ("сәлем қалайсың ?", "error", "kk"),
        ("привет как дела ?", "error", "ru"),
        ("hello how are you ?", "error", "en"),
    ]

    results = []
    correct = 0
    plan_correct = 0

    for q, expected_domain, real_lang in test_cases:
        # Think
        think_out = generate(think, [w2i["<think>"]] + ids(q) + [w2i["<plan>"]])
        actions = parse_plan(think_out)
        plan_str = " ".join(actions)

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

        expected_plans = {
            ("math", "kk"): "<lang_kk> <translate> <math> <translate_back>",
            ("math", "ru"): "<lang_ru> <translate> <math> <translate_back>",
            ("math", "en"): "<lang_en> <math>",
            ("code", "py"): "<lang_py> <code>",
            ("code", "js"): "<lang_js> <code>",
            ("error", "kk"): "<error>", ("error", "ru"): "<error>", ("error", "en"): "<error>",
        }
        p_ok = plan_str.strip() == expected_plans.get((expected_domain, real_lang), "").strip()
        if p_ok: plan_correct += 1

        trace = {"input": q, "domain": expected_domain, "lang": real_lang,
                 "plan": plan_str, "plan_ok": p_ok, "steps": []}
        final = ""

        if has_error:
            err_out = generate(err_model, [w2i["<err>"]] + ids(q) + [w2i["<msg>"]])
            final = extract_after(err_out, "<msg>")
            trace["steps"].append(("Err", final))

        elif has_code:
            tag = f"<lang_{detected_lang}>" if detected_lang else "<lang_py>"
            code_out = generate(code_model, [w2i[tag]] + ids(q) + [w2i["<result>"]])
            final = extract_after(code_out, "<result>")
            trace["steps"].append(("Code", final))

        elif has_math:
            # Step 1: Translate to EN if needed
            en_question = q  # default for EN direct
            if has_translate and detected_lang == "kk":
                en_question = hplt_kk2en.translate(q)
                trace["steps"].append(("HPLT_kk2en", en_question))
            elif has_translate and detected_lang == "ru":
                m1_out = generate(m1_ru, [w2i["<ru>"]] + ids(q) + [w2i["<en>"]])
                en_question = extract_after(m1_out, "<en>")
                trace["steps"].append(("M1_ru", en_question))

            # Step 2: SmolLM2 answers in English
            smol_prompt = f"Calculate and reply with ONLY the number, nothing else. {en_question}"
            smol_answer = smollm.answer(smol_prompt, max_new_tokens=10)
            trace["steps"].append(("SmolLM2", f"Q: {en_question} -> A: {smol_answer}"))

            # Step 3: Translate back if needed
            if has_back and detected_lang == "kk":
                final = hplt_en2kk.translate(smol_answer)
                trace["steps"].append(("HPLT_en2kk", final))
            elif has_back and detected_lang == "ru":
                # SmolLM2 ответ в свободной форме, M3_ru ожидает наш формат
                # Попробуем передать как есть
                m3_input_words = smol_answer.split()
                m3_prefix = [w2i["<en>"]] + [w2i.get(w, PAD) for w in m3_input_words] + [w2i["<ru>"]]
                m3_out = generate(m3_ru, m3_prefix)
                final = clean(extract_after(m3_out, "<ru>"))
                trace["steps"].append(("M3_ru", final))
            else:
                final = smol_answer

        trace["final"] = final
        trace["smol_raw"] = smol_answer if has_math and not has_error and not has_code else None

        # Для сравнения с GT
        expected_answer = None
        for s in samples:
            if s["src"] == q:
                expected_answer = s["answer"]
                break

        # SmolLM2 отвечает свободным текстом, проверяем содержит ли правильный ответ
        is_ok = False
        if expected_answer:
            if has_math and not has_error and not has_code:
                # Для math: проверяем что SmolLM2 дал правильное число
                # Ищем GT число из en_a
                gt_number = None
                for s in samples:
                    if s["src"] == q and s["domain"] == "math":
                        # en_a: "one plus two is three ."
                        en_a_words = s["en_a"].split()
                        # число перед точкой
                        gt_number = en_a_words[-2] if en_a_words[-1] == "." else en_a_words[-1]
                        break
                if gt_number and gt_number.lower() in smol_answer.lower():
                    is_ok = True
                elif final.strip() == expected_answer.strip():
                    is_ok = True
            else:
                is_ok = final.strip() == expected_answer.strip()

        if is_ok: correct += 1
        trace["expected"] = expected_answer
        trace["correct"] = is_ok
        results.append(trace)

        a_mark = "OK" if is_ok else "FAIL"
        p_mark = "P:ok" if p_ok else "P:FAIL"
        print(f"  [{a_mark}] [{p_mark}] {q}")
        print(f"       план: {plan_str}")
        for step_name, step_val in trace["steps"]:
            print(f"       {step_name}: {step_val}")
        if not is_ok:
            print(f"       expected: {expected_answer}")
        print()

    total_time = time.time() - t0
    n = len(test_cases)

    print(f"Plan точность:   {plan_correct}/{n} ({100*plan_correct/n:.0f}%)")
    print(f"Общая точность:  {correct}/{n} ({100*correct/n:.0f}%)")
    print(f"Время: {total_time:.1f}s")

    report = {
        "experiment": "exp_010",
        "description": "SmolLM2-135M-Instruct + HPLT float32 + tinygrad pipeline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparams": {"dim": DIM, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                        "batch": BATCH, "lr": LR, "steps": STEPS},
        "data": {"n_samples": len(samples), "vocab_size": V},
        "adapters": {
            "SmolLM2": "HuggingFaceTB/SmolLM2-135M-Instruct (135M params, float32)",
            "HPLT_kk2en": "HPLT/translate-kk-en-v2.0-hplt_opus (CTranslate2 float32)",
            "HPLT_en2kk": "HPLT/translate-en-kk-v2.0-hplt_opus (CTranslate2 float32)",
        },
        "training": {
            "Think": log_think.summary(), "M1_ru": log_m1ru.summary(),
            "Code": log_code.summary(),
            "M3_ru": log_m3ru.summary(), "Err": log_err.summary(),
        },
        "inference": {"plan_accuracy": plan_correct/n, "accuracy": correct/n,
                      "correct": correct, "total": n, "results": results},
        "total_time_s": round(total_time, 1),
    }
    with open(LOG_DIR / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {LOG_DIR / 'report.json'}")
