"""
exp_005/train.py — 4 модели: Router + M1(Multi→EN) + M2(Thinker) + M3(EN→Multi)

Router: classifier, определяет язык (kk/ru) по входному тексту
M1: <kk> ... <en> ... или <ru> ... <en> ...  (мультиязычный переводчик)
M2: <q> ... <a> ...  (thinker, решает арифметику на EN)
M3: <en> ... <kk> ... или <en> ... <ru> ...  (мультиязычный переводчик обратно)
"""

import json
import random
import time
import numpy as np
from pathlib import Path
from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, Embedding, LayerNorm
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, pad_batch
from model import DecoderOnlyModel

# ============================================================
# Гиперпараметры
# ============================================================
DIM      = 64
N_HEADS  = 4
N_LAYERS = 2
BATCH    = 32
LR       = 3e-3
STEPS    = 300

# ============================================================
# Логирование
# ============================================================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

class Logger:
    def __init__(self, name):
        self.name = name
        self.path = LOG_DIR / f"{name}.jsonl"
        self.path.write_text("")
        self.entries = []
        self.t0 = time.time()

    def log(self, step, loss):
        e = {"step": step, "loss": round(loss, 5), "t": round(time.time() - self.t0, 2)}
        self.entries.append(e)
        with open(self.path, "a") as f:
            f.write(json.dumps(e) + "\n")

    def summary(self):
        return {
            "model": self.name, "steps": len(self.entries),
            "loss_start": self.entries[0]["loss"], "loss_end": self.entries[-1]["loss"],
            "time_s": round(time.time() - self.t0, 1),
        }

# ============================================================
# Данные + словарь
# ============================================================
samples = make_dataset()
random.shuffle(samples)

SPECIALS = ["<pad>", "<bos>", "<eos>", "<kk>", "<ru>", "<en>", "<q>", "<a>"]
all_words = set()
for s in samples:
    for text in s[1:]:  # skip lang tag
        all_words.update(text.split())

vocab = SPECIALS + sorted(all_words)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
V = len(vocab)
PAD, EOS = w2i["<pad>"], w2i["<eos>"]

print(f"Примеров: {len(samples)} (KK: {sum(1 for s in samples if s[0]=='kk')}, RU: {sum(1 for s in samples if s[0]=='ru')})")
print(f"Словарь: {V} слов\n")

# ============================================================
# Router: простой классификатор (Embedding → mean pool → Linear(2))
# ============================================================
class Router:
    def __init__(self, vocab_size, dim=64):
        self.emb = Embedding(vocab_size, dim)
        self.ln = LayerNorm(dim)
        self.head = Linear(dim, 2)  # 0=kk, 1=ru

    def __call__(self, x: Tensor) -> Tensor:
        # x: (B, T) → mean pool → logits (B, 2)
        h = self.emb(x)           # (B, T, dim)
        h = self.ln(h.mean(axis=1))  # (B, dim)
        return self.head(h)       # (B, 2)

    def parameters(self):
        return get_parameters(self)

    def predict(self, text: str) -> str:
        ids = [w2i.get(w, PAD) for w in text.split()]
        logits = self(Tensor([ids], dtype=dtypes.int32))
        pred = int(logits[0].argmax().item())
        return "kk" if pred == 0 else "ru"

# ============================================================
# Подготовка данных
# ============================================================
# Router data: (text, label)
router_data = [(s[1], 0 if s[0] == "kk" else 1) for s in samples]

# M1: Multi→EN
m1_pairs = []
for s in samples:
    lang_tag = "<kk>" if s[0] == "kk" else "<ru>"
    pre = [w2i[lang_tag]] + [w2i[w] for w in s[1].split()]
    suf = [w2i["<en>"]] + [w2i[w] for w in s[2].split()] + [EOS]
    m1_pairs.append((pre, suf))

# M2: EN q → EN a
m2_pairs = []
for s in samples:
    pre = [w2i["<q>"]] + [w2i[w] for w in s[2].split()]
    suf = [w2i["<a>"]] + [w2i[w] for w in s[3].split()] + [EOS]
    m2_pairs.append((pre, suf))

# M3: EN → Multi (ответ на языке оригинала)
m3_pairs = []
for s in samples:
    lang_tag = "<kk>" if s[0] == "kk" else "<ru>"
    pre = [w2i["<en>"]] + [w2i[w] for w in s[3].split()]
    suf = [w2i[lang_tag]] + [w2i[w] for w in s[4].split()] + [EOS]
    m3_pairs.append((pre, suf))

# ============================================================
# Батчи
# ============================================================
def make_batch_decoder(pairs, batch_size):
    idxs = random.sample(range(len(pairs)), min(batch_size, len(pairs)))
    seqs = [pairs[i][0] + pairs[i][1] for i in idxs]
    masks = []
    for i in idxs:
        masks.append([0] * len(pairs[i][0]) + [1] * len(pairs[i][1]))
    padded = pad_batch(seqs, PAD)
    padded_masks = pad_batch(masks, 0)
    inp = [s[:-1] for s in padded]
    tgt = [s[1:] for s in padded]
    msk = [m[1:] for m in padded_masks]
    return (Tensor(inp, dtype=dtypes.int32),
            Tensor(tgt, dtype=dtypes.int32),
            Tensor(msk, dtype=dtypes.float32))

def make_batch_router(data, batch_size):
    idxs = random.sample(range(len(data)), min(batch_size, len(data)))
    texts = [data[i][0] for i in idxs]
    labels = [data[i][1] for i in idxs]
    seqs = [[w2i.get(w, PAD) for w in t.split()] for t in texts]
    padded = pad_batch(seqs, PAD)
    return Tensor(padded, dtype=dtypes.int32), Tensor(labels, dtype=dtypes.int32)

# ============================================================
# Loss
# ============================================================
def masked_ce_loss(logits, targets, mask):
    B, T, Voc = logits.shape
    lp = logits.reshape(-1, Voc).log_softmax(axis=-1)
    tf = targets.reshape(-1)
    mf = mask.reshape(-1)
    return ((-lp[Tensor.arange(lp.shape[0]), tf]) * mf).sum() / mf.sum()

def cls_loss(logits, labels):
    lp = logits.log_softmax(axis=-1)
    return -lp[Tensor.arange(lp.shape[0]), labels].mean()

# ============================================================
# Обучение
# ============================================================
def train_router(router, data, steps):
    logger = Logger("Router")
    n_params = sum(p.numel() for p in router.parameters())
    print(f"{'='*50}\nRouter ({n_params:,} params)\n{'='*50}")
    opt = Adam(router.parameters(), lr=LR)
    for step in range(steps):
        x, y = make_batch_router(data, BATCH)
        logits = router(x)
        loss = cls_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lv = loss.item()
        logger.log(step, lv)
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={lv:.3f}  [{logger.entries[-1]['t']:.1f}s]")
    s = logger.summary()
    print(f"  → {s['loss_start']:.3f} → {s['loss_end']:.3f} за {s['time_s']}s\n")
    return logger

def train_decoder(name, model, pairs, steps):
    logger = Logger(name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{'='*50}\n{name} ({n_params:,} params)\n{'='*50}")
    opt = Adam(model.parameters(), lr=LR)
    for step in range(steps):
        inp, tgt, msk = make_batch_decoder(pairs, BATCH)
        logits = model(inp)
        loss = masked_ce_loss(logits, tgt, msk)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lv = loss.item()
        logger.log(step, lv)
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={lv:.3f}  [{logger.entries[-1]['t']:.1f}s]")
    s = logger.summary()
    print(f"  → {s['loss_start']:.3f} → {s['loss_end']:.3f} за {s['time_s']}s\n")
    return logger

# ============================================================
# Greedy decode
# ============================================================
def generate(model, prefix_ids, suffix_tag, max_new=15):
    ids = prefix_ids + [w2i[suffix_tag]]
    for _ in range(max_new):
        logits = model(Tensor([ids], dtype=dtypes.int32))
        next_id = int(logits[0, -1].argmax().item())
        ids.append(next_id)
        if next_id == EOS:
            break
    tag_id = w2i[suffix_tag]
    last_tag = len(ids) - 1 - ids[::-1].index(tag_id)
    words = []
    for i in ids[last_tag + 1:]:
        w = i2w.get(i, "?")
        if w == "<eos>":
            break
        if w not in ("<pad>", "<bos>"):
            words.append(w)
    return " ".join(words)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    Tensor.training = True

    router = Router(V, DIM)
    m1 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    m2 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    m3 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)

    log_r = train_router(router, router_data, 100)  # быстро сходится
    log1 = train_decoder("M1_multi2en", m1, m1_pairs, STEPS)
    log2 = train_decoder("M2_thinker", m2, m2_pairs, STEPS)
    log3 = train_decoder("M3_en2multi", m3, m3_pairs, STEPS)

    # --- Инференс ---
    Tensor.training = False

    print(f"{'='*50}")
    print("ИНФЕРЕНС: [Router] → M1 → M2 → M3")
    print(f"{'='*50}\n")

    test_questions = [
        # Казахский
        "бір қосу екі нешеге тең ?",
        "он алу бес нешеге тең ?",
        "алты қосу үш нешеге тең ?",
        "тоғыз алу бір нешеге тең ?",
        # Русский
        "сколько будет два плюс три ?",
        "сколько будет десять минус семь ?",
        "сколько будет четыре плюс пять ?",
        "сколько будет восемь минус два ?",
    ]

    results = []
    correct = 0
    router_correct = 0

    for q in test_questions:
        # Router
        detected_lang = router.predict(q)
        # Определяем реальный язык
        real_lang = "kk" if any(w in q for w in ["қосу", "алу", "нешеге"]) else "ru"
        router_ok = detected_lang == real_lang
        if router_ok:
            router_correct += 1

        lang_tag = f"<{detected_lang}>"
        prefix = [w2i[lang_tag]] + [w2i.get(w, PAD) for w in q.split()]

        # M1: Multi→EN
        en_q = generate(m1, prefix, "<en>")
        # M2: Thinker
        m2_prefix = [w2i["<q>"]] + [w2i.get(w, PAD) for w in en_q.split()]
        en_a = generate(m2, m2_prefix, "<a>")
        # M3: EN→Multi (на языке, определённом роутером)
        m3_prefix = [w2i["<en>"]] + [w2i.get(w, PAD) for w in en_a.split()]
        src_a = generate(m3, m3_prefix, lang_tag)

        # Проверка
        expected = None
        for s in samples:
            if s[1] == q:
                expected = s[4]
                break
        is_ok = expected and src_a.strip() == expected.strip()
        if is_ok:
            correct += 1

        results.append({
            "question": q, "detected_lang": detected_lang, "real_lang": real_lang,
            "router_ok": router_ok, "en_q": en_q, "en_a": en_a,
            "answer": src_a, "expected": expected, "correct": is_ok,
        })

        r_mark = "R:ok" if router_ok else "R:FAIL"
        a_mark = "OK" if is_ok else "FAIL"
        print(f"  [{a_mark}] [{r_mark}] {q}")
        print(f"       lang={detected_lang} | M1→ {en_q}")
        print(f"       M2→ {en_a}")
        print(f"       M3→ {src_a}")
        if not is_ok:
            print(f"       exp: {expected}")
        print()

    total_time = time.time() - t0
    acc = correct / len(test_questions)
    r_acc = router_correct / len(test_questions)

    print(f"Router точность: {router_correct}/{len(test_questions)} ({100*r_acc:.0f}%)")
    print(f"Общая точность:  {correct}/{len(test_questions)} ({100*acc:.0f}%)")
    print(f"Время: {total_time:.1f}s")

    report = {
        "experiment": "exp_005",
        "description": "Router + multilingual decoder-only pipeline (KK+RU→EN→Think→KK/RU)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparams": {"dim": DIM, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                        "batch": BATCH, "lr": LR, "steps": STEPS},
        "data": {"n_samples": len(samples), "vocab_size": V},
        "training": {
            "Router": log_r.summary(), "M1": log1.summary(),
            "M2": log2.summary(), "M3": log3.summary(),
        },
        "inference": {
            "router_accuracy": r_acc, "accuracy": acc,
            "correct": correct, "total": len(test_questions),
            "results": results,
        },
        "total_time_s": round(total_time, 1),
    }
    with open(LOG_DIR / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {LOG_DIR / 'report.json'}")
