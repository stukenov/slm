"""
exp_006/model.py — Decoder-only transformer (общая архитектура для всех 4 моделей)
"""

from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, Embedding, LayerNorm
from tinygrad.nn.state import get_parameters


class SelfAttention:
    def __init__(self, dim, n_heads):
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = Linear(dim, dim * 3)
        self.out = Linear(dim, dim)

    def __call__(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        att = q @ k.permute(0, 1, 3, 2) * (self.head_dim ** -0.5)
        mask = Tensor.ones(T, T).triu(1).reshape(1, 1, T, T) * -1e9
        att = (att + mask).softmax(axis=-1)
        return self.out((att @ v).permute(0, 2, 1, 3).reshape(B, T, C))


class Block:
    def __init__(self, dim, n_heads):
        self.ln1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.ln2 = LayerNorm(dim)
        self.up = Linear(dim, dim * 4)
        self.down = Linear(dim * 4, dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.down(self.up(self.ln2(x)).relu())
        return x


class DecoderOnlyModel:
    def __init__(self, vocab_size, dim=64, n_heads=4, n_layers=2, max_len=32):
        self.tok_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(max_len, dim)
        self.blocks = [Block(dim, n_heads) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, vocab_size)

    def __call__(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(Tensor.arange(T, dtype=dtypes.int32).reshape(1, T))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def parameters(self):
        return get_parameters(self)
