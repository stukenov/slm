"""
exp_003/train.py — Обучение и инференс: арифметический пайплайн KK→EN→Think→EN→KK

Три модели:
  M1: Translator KK→EN (encoder-decoder)
  M2: Thinker EN→EN (decoder-only, решает арифметику)
  M3: Translator EN→KK (encoder-decoder)

Логи сохраняются в exp/exp_003/logs/
"""

import json
import math
import random
import time
import numpy as np
from pathlib import Path
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import Adam

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, build_vocabs, encode_seq, decode_seq, pad_batch
from model import TranslatorModel, ThinkerModel

# ============================================================
# Гиперпараметры
# ============================================================
DIM      = 64
N_HEADS  = 4
N_LAYERS = 2
BATCH    = 16
LR       = 3e-3
STEPS_TRANSLATOR = 200
STEPS_THINKER    = 200

# ============================================================
# Логирование
# ============================================================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

class Logger:
    def __init__(self, name: str):
        self.name = name
        self.path = LOG_DIR / f"{name}.jsonl"
        self.entries = []
        self.start_time = time.time()

    def log_step(self, step, loss, **extra):
        entry = {
            "model": self.name,
            "step": step,
            "loss": round(loss, 5),
            "elapsed_s": round(time.time() - self.start_time, 2),
            **extra,
        }
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_meta(self, **meta):
        with open(self.path, "a") as f:
            f.write(json.dumps({"_meta": True, **meta}) + "\n")

    def summary(self):
        losses = [e["loss"] for e in self.entries]
        elapsed = time.time() - self.start_time
        return {
            "model": self.name,
            "steps": len(losses),
            "loss_start": losses[0] if losses else None,
            "loss_end": losses[-1] if losses else None,
            "elapsed_s": round(elapsed, 1),
            "steps_per_sec": round(len(losses) / elapsed, 2) if elapsed > 0 else 0,
        }

# ============================================================
# Данные
# ============================================================
samples = make_dataset()
random.shuffle(samples)
(kk_vocab, kk_w2i, kk_i2w), (en_vocab, en_w2i, en_i2w) = build_vocabs(samples)

print(f"Примеров: {len(samples)}")
print(f"KK словарь: {len(kk_vocab)} | EN словарь: {len(en_vocab)}")

m1_data = [(s[0], s[1]) for s in samples]  # KK вопрос → EN вопрос
m2_data = [(s[1], s[2]) for s in samples]  # EN вопрос → EN ответ
m3_data = [(s[2], s[3]) for s in samples]  # EN ответ → KK ответ

# ============================================================
# Батч-семплеры
# ============================================================
def make_batch_translator(data, src_w2i, tgt_w2i, batch_size):
    idxs = random.sample(range(len(data)), min(batch_size, len(data)))
    src_seqs = [encode_seq(data[i][0], src_w2i, add_bos=False, add_eos=True) for i in idxs]
    tgt_in = [encode_seq(data[i][1], tgt_w2i, add_bos=True, add_eos=False) for i in idxs]
    tgt_out = [encode_seq(data[i][1], tgt_w2i, add_bos=False, add_eos=True) for i in idxs]

    return (Tensor(pad_batch(src_seqs, src_w2i["<pad>"]), dtype=dtypes.int32),
            Tensor(pad_batch(tgt_in, tgt_w2i["<pad>"]), dtype=dtypes.int32),
            Tensor(pad_batch(tgt_out, tgt_w2i["<pad>"]), dtype=dtypes.int32))


def make_batch_thinker(data, w2i, batch_size):
    sep_id = w2i["<sep>"]
    idxs = random.sample(range(len(data)), min(batch_size, len(data)))
    seqs = []
    for i in idxs:
        q_ids = [w2i[w] for w in data[i][0].split()]
        a_ids = [w2i[w] for w in data[i][1].split()]
        seqs.append([w2i["<bos>"]] + q_ids + [sep_id] + a_ids + [w2i["<eos>"]])
    t = Tensor(pad_batch(seqs, w2i["<pad>"]), dtype=dtypes.int32)
    return t[:, :-1], t[:, 1:]


# <sep> для M2
en_vocab.append("<sep>")
en_w2i["<sep>"] = len(en_vocab) - 1
en_i2w[len(en_vocab) - 1] = "<sep>"

# ============================================================
# Loss
# ============================================================
def compute_ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    B, T, V = logits.shape
    log_probs = logits.reshape(-1, V).log_softmax(axis=-1)
    targets_flat = targets.reshape(-1)
    return -log_probs[Tensor.arange(log_probs.shape[0]), targets_flat].mean()

# ============================================================
# Обучение
# ============================================================
def train_translator(name, model, data, src_w2i, tgt_w2i, steps):
    logger = Logger(name)
    n_params = sum(p.numel() for p in model.parameters())
    logger.log_meta(params=n_params, dim=DIM, heads=N_HEADS, layers=N_LAYERS, lr=LR, batch=BATCH)
    print(f"\n{'='*50}")
    print(f"Обучение {name} ({n_params:,} params)")
    print(f"{'='*50}")

    opt = Adam(model.parameters(), lr=LR)
    for step in range(steps):
        src, tgt_in, tgt_out = make_batch_translator(data, src_w2i, tgt_w2i, BATCH)
        logits = model(src, tgt_in)
        loss = compute_ce_loss(logits, tgt_out)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = loss.item()
        logger.log_step(step, loss_val)
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={loss_val:.3f}  [{logger.entries[-1]['elapsed_s']:.1f}s]")

    s = logger.summary()
    print(f"  → Итого: {s['loss_start']:.3f} → {s['loss_end']:.3f} за {s['elapsed_s']}s ({s['steps_per_sec']} steps/s)")
    return logger


def train_thinker(model, data, w2i, steps):
    logger = Logger("M2_thinker")
    n_params = sum(p.numel() for p in model.parameters())
    logger.log_meta(params=n_params, dim=DIM, heads=N_HEADS, layers=N_LAYERS, lr=LR, batch=BATCH)
    print(f"\n{'='*50}")
    print(f"Обучение M2 Thinker ({n_params:,} params)")
    print(f"{'='*50}")

    opt = Adam(model.parameters(), lr=LR)
    for step in range(steps):
        inp, tgt = make_batch_thinker(data, w2i, BATCH)
        logits = model(inp)
        loss = compute_ce_loss(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = loss.item()
        logger.log_step(step, loss_val)
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={loss_val:.3f}  [{logger.entries[-1]['elapsed_s']:.1f}s]")

    s = logger.summary()
    print(f"  → Итого: {s['loss_start']:.3f} → {s['loss_end']:.3f} за {s['elapsed_s']}s ({s['steps_per_sec']} steps/s)")
    return logger

# ============================================================
# Greedy decode
# ============================================================
def greedy_translate(model, src_text, src_w2i, tgt_w2i, tgt_i2w, max_len=15):
    src_ids = encode_seq(src_text, src_w2i, add_bos=False, add_eos=True)
    enc_out = model.encode(Tensor([src_ids], dtype=dtypes.int32))
    tgt_ids = [tgt_w2i["<bos>"]]
    for _ in range(max_len):
        logits = model.decode(Tensor([tgt_ids], dtype=dtypes.int32), enc_out)
        next_id = int(logits[0, -1].argmax().item())
        tgt_ids.append(next_id)
        if next_id == tgt_w2i["<eos>"]:
            break
    return decode_seq(tgt_ids, tgt_i2w)


def greedy_think(model, question_text, w2i, i2w, max_len=15):
    q_ids = [w2i["<bos>"]] + [w2i[w] for w in question_text.split()] + [w2i["<sep>"]]
    for _ in range(max_len):
        logits = model(Tensor([q_ids], dtype=dtypes.int32))
        next_id = int(logits[0, -1].argmax().item())
        q_ids.append(next_id)
        if next_id == w2i["<eos>"]:
            break
    sep_idx = q_ids.index(w2i["<sep>"])
    return decode_seq(q_ids[sep_idx + 1:], i2w)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    Tensor.training = True

    m1 = TranslatorModel(len(kk_vocab), len(en_vocab), DIM, N_HEADS, N_LAYERS)
    m2 = ThinkerModel(len(en_vocab), DIM, N_HEADS, N_LAYERS, max_len=32)
    m3 = TranslatorModel(len(en_vocab), len(kk_vocab), DIM, N_HEADS, N_LAYERS)

    log1 = train_translator("M1_kk2en", m1, m1_data, kk_w2i, en_w2i, STEPS_TRANSLATOR)
    log2 = train_thinker(m2, m2_data, en_w2i, STEPS_THINKER)
    log3 = train_translator("M3_en2kk", m3, m3_data, en_w2i, kk_w2i, STEPS_TRANSLATOR)

    # --- Инференс ---
    Tensor.training = False

    print(f"\n{'='*50}")
    print("ИНФЕРЕНС: KK → EN → Think → KK")
    print(f"{'='*50}\n")

    test_questions = [
        "бір қосу екі нешеге тең ?",
        "үш қосу төрт нешеге тең ?",
        "он алу бес нешеге тең ?",
        "сегіз алу үш нешеге тең ?",
        "алты қосу үш нешеге тең ?",
        "жеті алу жеті нешеге тең ?",
    ]

    results = []
    correct = 0

    for kk_q in test_questions:
        en_q = greedy_translate(m1, kk_q, kk_w2i, en_w2i, en_i2w)
        en_a = greedy_think(m2, en_q, en_w2i, en_i2w)
        kk_a = greedy_translate(m3, en_a, en_w2i, kk_w2i, kk_i2w)

        expected = None
        for s in samples:
            if s[0] == kk_q:
                expected = s[3]
                break

        is_correct = expected and kk_a.strip() == expected.strip()
        if is_correct:
            correct += 1

        result = {
            "kk_question": kk_q,
            "en_question": en_q,
            "en_answer": en_a,
            "kk_answer": kk_a,
            "expected": expected,
            "correct": is_correct,
        }
        results.append(result)

        mark = "OK" if is_correct else "FAIL"
        print(f"  [{mark}] {kk_q}")
        print(f"       M1→ {en_q}")
        print(f"       M2→ {en_a}")
        print(f"       M3→ {kk_a}")
        if not is_correct:
            print(f"       ожидалось: {expected}")
        print()

    total_time = time.time() - t0
    accuracy = correct / len(test_questions)

    print(f"Точность: {correct}/{len(test_questions)} ({100*accuracy:.0f}%)")
    print(f"Общее время: {total_time:.1f}s")

    # --- Сохраняем итоговый отчёт ---
    report = {
        "experiment": "exp_003",
        "description": "Arithmetic pipeline KK→EN→Think→KK",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparams": {
            "dim": DIM, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            "batch": BATCH, "lr": LR,
            "steps_translator": STEPS_TRANSLATOR, "steps_thinker": STEPS_THINKER,
        },
        "data": {
            "n_samples": len(samples),
            "kk_vocab_size": len(kk_vocab),
            "en_vocab_size": len(en_vocab),
        },
        "training": {
            "M1_kk2en": log1.summary(),
            "M2_thinker": log2.summary(),
            "M3_en2kk": log3.summary(),
        },
        "inference": {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_questions),
            "results": results,
        },
        "total_time_s": round(total_time, 1),
    }

    report_path = LOG_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {report_path}")
