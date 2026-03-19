"""OmniAudio: Kazakh ASR decode-only omni-model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(self, n_mels=80, d_model=384, n_heads=6, n_layers=4, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)

        # Sinusoidal positional encoding
        max_len = 5000
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, time)
        x = self.gelu(self.conv1(mel))
        x = self.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, seq_len, d_model)
        x = self.ln(x)
        x = x + self.pe[:, :x.size(1)]
        x = self.transformer(x)
        return x


class AudioProjector(nn.Module):
    def __init__(self, audio_dim=384, llm_dim=576):
        super().__init__()
        self.linear = nn.Linear(audio_dim, llm_dim)
        self.ln = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.linear(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class LlamaDecoderBlock(nn.Module):
    def __init__(self, d_model=576, n_heads=8, intermediate_size=1536, dropout=0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.attn_norm(x)
        if attn_mask is None:
            seq_len = h.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=h.device)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=True)
        x = x + h
        # SwiGLU FFN with pre-norm
        h = self.ffn_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class OmniAudioModel(nn.Module):
    def __init__(
        self,
        n_mels=80, audio_d_model=384, audio_n_heads=6, audio_n_layers=4,
        llm_vocab_size=32000, llm_d_model=576, llm_n_heads=8, llm_n_layers=8,
        llm_intermediate_size=1536, dropout=0.1,
    ):
        super().__init__()
        self.audio_encoder = AudioEncoder(n_mels, audio_d_model, audio_n_heads, audio_n_layers, dropout)
        self.projector = AudioProjector(audio_d_model, llm_d_model)
        self.text_embed = nn.Embedding(llm_vocab_size, llm_d_model)
        self.layers = nn.ModuleList([
            LlamaDecoderBlock(llm_d_model, llm_n_heads, llm_intermediate_size, dropout)
            for _ in range(llm_n_layers)
        ])
        self.norm = RMSNorm(llm_d_model)
        self.lm_head = nn.Linear(llm_d_model, llm_vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.text_embed.weight

    def forward(self, mel: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        audio_embeds = self.projector(self.audio_encoder(mel))
        text_embeds = self.text_embed(text_ids)
        combined = torch.cat([audio_embeds, text_embeds], dim=1)

        x = combined
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        audio_len = audio_embeds.size(1)
        # Logits at positions [audio_len-1:-1] predict text_ids
        text_logits = logits[:, audio_len - 1:-1]  # (B, text_len, vocab)
        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_ids.reshape(-1),
            ignore_index=-100,
        )
        return loss

    @torch.no_grad()
    def generate(self, mel: torch.Tensor, tokenizer, max_new_tokens: int = 200) -> list[int]:
        self.eval()
        audio_embeds = self.projector(self.audio_encoder(mel))
        generated: list[int] = []
        text_ids = torch.zeros(1, 0, dtype=torch.long, device=mel.device)

        for _ in range(max_new_tokens):
            if text_ids.size(1) > 0:
                text_embeds = self.text_embed(text_ids)
            else:
                text_embeds = torch.zeros(1, 0, audio_embeds.size(2), device=mel.device)
            combined = torch.cat([audio_embeds, text_embeds], dim=1)
            x = combined
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            logits = self.lm_head(x[:, -1:])
            next_token = logits.argmax(dim=-1).item()
            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)
            text_ids = torch.cat([text_ids, torch.tensor([[next_token]], device=mel.device)], dim=1)

        return generated
