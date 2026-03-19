"""
exp_003/model.py — Модели для арифметического пайплайна

M1 (KK→EN): Encoder-Decoder Transformer
M2 (EN→EN): Decoder-only Transformer (думает/считает)
M3 (EN→KK): Encoder-Decoder Transformer
"""

from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, Embedding, LayerNorm
from tinygrad.nn.state import get_parameters


# ============================================================
# Общие блоки
# ============================================================

class SelfAttention:
    def __init__(self, dim, n_heads, causal=False):
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
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
        if self.causal:
            mask = Tensor.ones(T, T).triu(1).reshape(1, 1, T, T) * -1e9
            att = att + mask
        att = att.softmax(axis=-1)
        out = (att @ v).permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out(out)


class CrossAttention:
    def __init__(self, dim, n_heads):
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = Linear(dim, dim)
        self.kv_proj = Linear(dim, dim * 2)
        self.out = Linear(dim, dim)

    def __call__(self, x: Tensor, enc_out: Tensor) -> Tensor:
        B, T, C = x.shape
        S = enc_out.shape[1]
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(enc_out).reshape(B, S, 2, self.n_heads, self.head_dim)
        k, v = kv[:, :, 0].permute(0, 2, 1, 3), kv[:, :, 1].permute(0, 2, 1, 3)

        att = (q @ k.permute(0, 1, 3, 2) * (self.head_dim ** -0.5)).softmax(axis=-1)
        out = (att @ v).permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out(out)


class FFN:
    def __init__(self, dim, mult=4):
        self.up = Linear(dim, dim * mult)
        self.down = Linear(dim * mult, dim)

    def __call__(self, x: Tensor) -> Tensor:
        return self.down(self.up(x).relu())


# ============================================================
# Encoder Block (bidirectional self-attention)
# ============================================================

class EncoderBlock:
    def __init__(self, dim, n_heads):
        self.ln1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, causal=False)
        self.ln2 = LayerNorm(dim)
        self.ffn = FFN(dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# Decoder Block (causal self-attention + cross-attention)
# ============================================================

class DecoderBlock:
    def __init__(self, dim, n_heads):
        self.ln1 = LayerNorm(dim)
        self.self_attn = SelfAttention(dim, n_heads, causal=True)
        self.ln2 = LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, n_heads)
        self.ln3 = LayerNorm(dim)
        self.ffn = FFN(dim)

    def __call__(self, x: Tensor, enc_out: Tensor) -> Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), enc_out)
        x = x + self.ffn(self.ln3(x))
        return x


# ============================================================
# M1/M3: Encoder-Decoder Transformer (Translation)
# ============================================================

class TranslatorModel:
    def __init__(self, src_vocab, tgt_vocab, dim=64, n_heads=4, n_layers=2, max_len=20):
        self.src_emb = Embedding(src_vocab, dim)
        self.tgt_emb = Embedding(tgt_vocab, dim)
        self.src_pos = Embedding(max_len, dim)
        self.tgt_pos = Embedding(max_len, dim)
        self.enc_blocks = [EncoderBlock(dim, n_heads) for _ in range(n_layers)]
        self.dec_blocks = [DecoderBlock(dim, n_heads) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, tgt_vocab)

    def encode(self, src: Tensor) -> Tensor:
        B, S = src.shape
        x = self.src_emb(src) + self.src_pos(Tensor.arange(S, dtype=dtypes.int32).reshape(1, S))
        for block in self.enc_blocks:
            x = block(x)
        return x

    def decode(self, tgt: Tensor, enc_out: Tensor) -> Tensor:
        B, T = tgt.shape
        x = self.tgt_emb(tgt) + self.tgt_pos(Tensor.arange(T, dtype=dtypes.int32).reshape(1, T))
        for block in self.dec_blocks:
            x = block(x, enc_out)
        return self.head(self.ln_f(x))

    def __call__(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.decode(tgt, self.encode(src))

    def parameters(self):
        return get_parameters(self)


# ============================================================
# M2: Decoder-Only Transformer (Thinker — EN→EN)
# ============================================================

class CausalBlock:
    def __init__(self, dim, n_heads):
        self.ln1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, causal=True)
        self.ln2 = LayerNorm(dim)
        self.ffn = FFN(dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ThinkerModel:
    def __init__(self, vocab_size, dim=64, n_heads=4, n_layers=2, max_len=32):
        self.tok_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(max_len, dim)
        self.blocks = [CausalBlock(dim, n_heads) for _ in range(n_layers)]
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
