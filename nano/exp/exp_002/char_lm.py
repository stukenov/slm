"""
nano/char_lm.py — Character-level Language Model на tinygrad.

Архитектура: мини-Transformer (decoder-only, GPT-style)
  - Embedding(vocab, dim) + positional embedding
  - N transformer блоков: LayerNorm → MultiHeadAttention → LayerNorm → FFN
  - Linear head → logits over vocab

Обучение на сыром Unicode-тексте (символьный уровень, без токенизатора).
"""

import math
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, Embedding, LayerNorm
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

# ============================================================
# Гиперпараметры
# ============================================================
CTX_LEN   = 64      # длина контекста (символов)
DIM       = 64      # размер эмбеддинга
N_HEADS   = 4       # голов внимания
N_LAYERS  = 2       # трансформер-блоков
FFN_MUL   = 4       # множитель для FFN
BATCH     = 32
STEPS     = 600
LR        = 3e-3

# ============================================================
# Корпус — казахский текст (можно заменить на любой Unicode)
# ============================================================
CORPUS = """
Қазақстан — Орталық Азиядағы мемлекет. Астанасы — Астана қаласы.
Ресми тілі — қазақ тілі. Халық саны — 20 миллионнан астам.
Қазақстан аумағы жағынан әлемдегі тоғызыншы ірі мемлекет.
Батысында Каспий теңізі, шығысында Алтай тауы, оңтүстігінде
Тянь-Шань тауы, солтүстігінде Батыс Сібір жазығы орналасқан.
Қазақ халқы көшпенді өмір салтын ұстанған. Мал шаруашылығы,
аңшылық пен саудагерлік негізгі кәсіптері болған.
Қазіргі Қазақстан — тәуелсіз, демократиялық мемлекет.
Экономикасы мұнай, газ, уран және басқа пайдалы қазбаларға негізделген.
Білім беру жүйесі дамыған, университеттер мен ғылыми орталықтар көп.
Мәдениеті бай — домбыра, күй, ақындар дәстүрі, көкпар, бәйге.
Тамақ мәдениеті — бешбармақ, қуырдақ, баурсақ, құрт, қымыз.
""".strip()

# ============================================================
# Символьный словарь
# ============================================================
chars = sorted(set(CORPUS))
VOCAB = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}

def encode(s):  return [c2i[c] for c in s]
def decode(ids): return "".join(i2c[i] for i in ids)

data = encode(CORPUS)
print(f"Корпус: {len(CORPUS)} символов, словарь: {VOCAB} уникальных")

# ============================================================
# Модель
# ============================================================
class Attention:
    def __init__(self, dim, n_heads):
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = Linear(dim, dim * 3)
        self.out = Linear(dim, dim)

    def __call__(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # (B, heads, T, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        att = q @ k.permute(0, 1, 3, 2) * scale
        # causal mask
        mask = Tensor.ones(T, T).triu(1).reshape(1, 1, T, T) * -1e9
        att = (att + mask).softmax(axis=-1)
        out = (att @ v).permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out(out)


class FFN:
    def __init__(self, dim):
        self.up = Linear(dim, dim * FFN_MUL)
        self.down = Linear(dim * FFN_MUL, dim)

    def __call__(self, x: Tensor) -> Tensor:
        return self.down(self.up(x).relu())


class Block:
    def __init__(self, dim, n_heads):
        self.ln1 = LayerNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ln2 = LayerNorm(dim)
        self.ffn = FFN(dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CharTransformer:
    def __init__(self):
        self.tok_emb = Embedding(VOCAB, DIM)
        self.pos_emb = Embedding(CTX_LEN, DIM)
        self.blocks = [Block(DIM, N_HEADS) for _ in range(N_LAYERS)]
        self.ln_f = LayerNorm(DIM)
        self.head = Linear(DIM, VOCAB)

    def __call__(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        pos = Tensor.arange(T, dtype=dtypes.int32).reshape(1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def parameters(self):
        return get_parameters(self)


model = CharTransformer()
n_params = sum(p.numel() for p in model.parameters())
print(f"Модель: {N_LAYERS}L, {N_HEADS}H, dim={DIM}, {n_params:,} параметров\n")

# ============================================================
# Батч-семплер
# ============================================================
def get_batch():
    starts = np.random.randint(0, len(data) - CTX_LEN - 1, size=BATCH)
    x = np.array([data[s:s+CTX_LEN] for s in starts])
    y = np.array([data[s+1:s+CTX_LEN+1] for s in starts])
    return Tensor(x, dtype=dtypes.int32), Tensor(y, dtype=dtypes.int32)

# ============================================================
# Обучение
# ============================================================
Tensor.training = True
opt = Adam(model.parameters(), lr=LR)

print("=== Обучение ===\n")
for step in range(STEPS):
    x, y = get_batch()
    logits = model(x)                        # (B, T, VOCAB)
    logits_flat = logits.reshape(-1, VOCAB)   # (B*T, VOCAB)
    targets_flat = y.reshape(-1)              # (B*T,)

    # cross-entropy
    log_probs = logits_flat.log_softmax(axis=-1)
    loss = -log_probs[Tensor.arange(logits_flat.shape[0]), targets_flat].mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0 or step == STEPS - 1:
        print(f"  step {step:3d}  loss={loss.item():.3f}  ppl={math.exp(loss.item()):.1f}")

# ============================================================
# Генерация (инференс)
# ============================================================
Tensor.training = False

def generate(prompt: str, max_new: int = 200, temperature: float = 0.8) -> str:
    ids = encode(prompt)
    for _ in range(max_new):
        # обрезаем до CTX_LEN
        ctx = ids[-CTX_LEN:]
        x = Tensor([ctx], dtype=dtypes.int32)
        logits = model(x)
        # берём логиты последнего токена
        next_logits = logits[0, -1] / temperature
        probs = next_logits.softmax()
        probs_np = probs.numpy().astype(np.float64)
        probs_np /= probs_np.sum()
        next_id = int(np.random.choice(len(probs_np), p=probs_np))
        ids.append(next_id)
    return decode(ids)


print("\n=== Генерация ===\n")
for prompt in ["Қазақ", "Астана", "Мәдени"]:
    text = generate(prompt, max_new=150)
    print(f"[{prompt}] → {text}\n")
