"""
exp_004/train.py — Все три модели decoder-only

M1: <kk> бір қосу екі нешеге тең ? <en> what is one plus two ? <eos>
M2: <q> what is one plus two ? <a> one plus two is three . <eos>
M3: <en> one plus two is three . <kk> бір қосу екі — үш . <eos>

Loss считается ТОЛЬКО по токенам после разделителя (target part).
"""

import json
import random
import time
import numpy as np
from pathlib import Path
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import Adam

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, build_vocabs, pad_batch
from model import DecoderOnlyModel

# ============================================================
# Гиперпараметры
# ============================================================
DIM      = 64
N_HEADS  = 4
N_LAYERS = 2
BATCH    = 16
LR       = 3e-3
STEPS    = 300  # больше шагов для M2

# ============================================================
# Логирование
# ============================================================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

class Logger:
    def __init__(self, name):
        self.name = name
        self.path = LOG_DIR / f"{name}.jsonl"
        self.path.write_text("")  # clear
        self.entries = []
        self.t0 = time.time()

    def log(self, step, loss):
        e = {"step": step, "loss": round(loss, 5), "t": round(time.time() - self.t0, 2)}
        self.entries.append(e)
        with open(self.path, "a") as f:
            f.write(json.dumps(e) + "\n")

    def summary(self):
        return {
            "model": self.name,
            "steps": len(self.entries),
            "loss_start": self.entries[0]["loss"],
            "loss_end": self.entries[-1]["loss"],
            "time_s": round(time.time() - self.t0, 1),
        }

# ============================================================
# Данные
# ============================================================
samples = make_dataset()
random.shuffle(samples)
(kk_vocab, kk_w2i, kk_i2w), (en_vocab, en_w2i, en_i2w) = build_vocabs(samples)

# Единый словарь для всех моделей (KK + EN + specials)
SPECIALS = ["<pad>", "<bos>", "<eos>", "<kk>", "<en>", "<q>", "<a>"]
all_words = set()
for s in samples:
    for text in s:
        all_words.update(text.split())

vocab = SPECIALS + sorted(all_words)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
V = len(vocab)
PAD, BOS, EOS = w2i["<pad>"], w2i["<bos>"], w2i["<eos>"]

print(f"Примеров: {len(samples)} | Общий словарь: {V} слов")
print(f"Слова: {vocab}\n")

# ============================================================
# Формирование последовательностей
# ============================================================
def make_seq(prefix_tag, prefix_text, suffix_tag, suffix_text):
    """<tag> prefix_words <tag2> suffix_words <eos>"""
    pre = [w2i[prefix_tag]] + [w2i[w] for w in prefix_text.split()]
    suf = [w2i[suffix_tag]] + [w2i[w] for w in suffix_text.split()] + [EOS]
    return pre, suf

# Подготовка данных для каждой модели
def build_pairs(samples, pre_tag, suf_tag, src_idx, tgt_idx):
    pairs = []
    for s in samples:
        pre, suf = make_seq(pre_tag, s[src_idx], suf_tag, s[tgt_idx])
        pairs.append((pre, suf))
    return pairs

m1_pairs = build_pairs(samples, "<kk>", "<en>", 0, 1)  # KK→EN
m2_pairs = build_pairs(samples, "<q>",  "<a>",  1, 2)   # EN q→EN a
m3_pairs = build_pairs(samples, "<en>", "<kk>", 2, 3)   # EN→KK

# ============================================================
# Батчи с masked loss (только target часть)
# ============================================================
def make_batch(pairs, batch_size):
    idxs = random.sample(range(len(pairs)), min(batch_size, len(pairs)))
    seqs = [pairs[i][0] + pairs[i][1] for i in idxs]
    # loss mask: 0 для prefix, 1 для suffix
    masks = []
    for i in idxs:
        pre_len = len(pairs[i][0])
        suf_len = len(pairs[i][1])
        masks.append([0] * pre_len + [1] * suf_len)

    padded = pad_batch(seqs, PAD)
    padded_masks = pad_batch(masks, 0)

    inp = [s[:-1] for s in padded]
    tgt = [s[1:] for s in padded]
    msk = [m[1:] for m in padded_masks]  # aligned with target

    return (Tensor(inp, dtype=dtypes.int32),
            Tensor(tgt, dtype=dtypes.int32),
            Tensor(msk, dtype=dtypes.float32))

# ============================================================
# Loss (masked)
# ============================================================
def masked_ce_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    B, T, Voc = logits.shape
    log_probs = logits.reshape(-1, Voc).log_softmax(axis=-1)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)
    token_losses = -log_probs[Tensor.arange(log_probs.shape[0]), targets_flat]
    # Средний loss только по target-токенам
    return (token_losses * mask_flat).sum() / mask_flat.sum()

# ============================================================
# Обучение
# ============================================================
def train_model(name, model, pairs, steps):
    logger = Logger(name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{'='*50}")
    print(f"{name} ({n_params:,} params)")
    print(f"{'='*50}")

    opt = Adam(model.parameters(), lr=LR)
    for step in range(steps):
        inp, tgt, msk = make_batch(pairs, BATCH)
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
def generate(model, prefix_tag, prefix_text, suffix_tag, max_new=15):
    ids = [w2i[prefix_tag]] + [w2i[w] for w in prefix_text.split()] + [w2i[suffix_tag]]
    for _ in range(max_new):
        logits = model(Tensor([ids], dtype=dtypes.int32))
        next_id = int(logits[0, -1].argmax().item())
        ids.append(next_id)
        if next_id == EOS:
            break
    # Извлекаем часть после suffix_tag
    tag_id = w2i[suffix_tag]
    last_tag = len(ids) - 1 - ids[::-1].index(tag_id)
    result_ids = ids[last_tag + 1:]
    words = []
    for i in result_ids:
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

    m1 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    m2 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)
    m3 = DecoderOnlyModel(V, DIM, N_HEADS, N_LAYERS, max_len=32)

    log1 = train_model("M1_kk2en", m1, m1_pairs, STEPS)
    log2 = train_model("M2_thinker", m2, m2_pairs, STEPS)
    log3 = train_model("M3_en2kk", m3, m3_pairs, STEPS)

    # --- Инференс ---
    Tensor.training = False

    print(f"{'='*50}")
    print("ИНФЕРЕНС: KK → EN → Think → KK (all decoder-only)")
    print(f"{'='*50}\n")

    test_questions = [
        "бір қосу екі нешеге тең ?",
        "үш қосу төрт нешеге тең ?",
        "он алу бес нешеге тең ?",
        "сегіз алу үш нешеге тең ?",
        "алты қосу үш нешеге тең ?",
        "жеті алу жеті нешеге тең ?",
        "нөл қосу он нешеге тең ?",
        "тоғыз алу бір нешеге тең ?",
    ]

    results = []
    correct = 0

    for kk_q in test_questions:
        en_q = generate(m1, "<kk>", kk_q, "<en>")
        en_a = generate(m2, "<q>", en_q, "<a>")
        kk_a = generate(m3, "<en>", en_a, "<kk>")

        expected = None
        for s in samples:
            if s[0] == kk_q:
                expected = s[3]
                break

        is_ok = expected and kk_a.strip() == expected.strip()
        if is_ok:
            correct += 1

        results.append({
            "kk_q": kk_q, "en_q": en_q, "en_a": en_a,
            "kk_a": kk_a, "expected": expected, "correct": is_ok,
        })

        mark = "OK" if is_ok else "FAIL"
        print(f"  [{mark}] {kk_q}")
        print(f"       M1→ {en_q}")
        print(f"       M2→ {en_a}")
        print(f"       M3→ {kk_a}")
        if not is_ok:
            print(f"       exp: {expected}")
        print()

    total_time = time.time() - t0
    acc = correct / len(test_questions)
    print(f"Точность: {correct}/{len(test_questions)} ({100*acc:.0f}%)")
    print(f"Время: {total_time:.1f}s")

    # Отчёт
    report = {
        "experiment": "exp_004",
        "description": "All decoder-only arithmetic pipeline KK→EN→Think→KK",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparams": {"dim": DIM, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                        "batch": BATCH, "lr": LR, "steps": STEPS},
        "data": {"n_samples": len(samples), "vocab_size": V},
        "training": {"M1": log1.summary(), "M2": log2.summary(), "M3": log3.summary()},
        "inference": {"accuracy": acc, "correct": correct,
                      "total": len(test_questions), "results": results},
        "total_time_s": round(total_time, 1),
    }
    report_path = LOG_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Отчёт: {report_path}")
