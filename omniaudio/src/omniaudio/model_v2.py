"""OmniAudio V2: Kazakh ASR with RoPE, RMSNorm, SwiGLU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim

    def forward(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x: (B, heads, seq, head_dim). cos/sin: (seq, head_dim)."""
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # SwiGLU FFN
        ffn_dim = int(d_model * 8 / 3)  # ~2.67x, standard SwiGLU ratio
        # Round to multiple of 64 for efficiency
        ffn_dim = ((ffn_dim + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Pre-norm + self-attention with RoPE
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.o_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out

        # Pre-norm + SwiGLU FFN
        h = self.norm2(x)
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + self.ffn_dropout(self.down_proj(gate * up))

        return x


class AudioEncoderV2(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 6, n_conv: int = 2, dropout: float = 0.1):
        super().__init__()
        # Conv stack
        convs = []
        in_ch = n_mels
        for i in range(n_conv):
            out_ch = d_model
            convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            convs.append(nn.GELU())
            in_ch = d_model
        self.conv_stack = nn.Sequential(*convs)
        self.norm = RMSNorm(d_model)

        # RoPE
        head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(dim=head_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, time)
        x = self.conv_stack(mel)  # (B, d_model, time // 2^n_conv)
        x = x.transpose(1, 2)    # (B, seq_len, d_model)
        x = self.norm(x)

        cos, sin = self.rope(x.size(1))
        for layer in self.layers:
            x = layer(x, cos, sin)

        return x


class AudioProjectorV2(nn.Module):
    def __init__(self, audio_dim: int, llm_dim: int):
        super().__init__()
        self.linear = nn.Linear(audio_dim, llm_dim)
        self.norm = RMSNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class OmniAudioV2Model(nn.Module):
    def __init__(self, encoder_config: dict, llm_name: str | None,
                 vocab_size: int, llm_dim: int = 768):
        super().__init__()
        self.encoder = AudioEncoderV2(**encoder_config)
        self.projector = AudioProjectorV2(encoder_config["d_model"], llm_dim)
        self.ctc_head = nn.Linear(encoder_config["d_model"], vocab_size)

        if llm_name:
            from transformers import LlamaForCausalLM
            self.llm = LlamaForCausalLM.from_pretrained(llm_name)
            for p in self.llm.parameters():
                p.requires_grad = False
        else:
            self.llm = None

    def forward_ctc(self, mel: torch.Tensor, targets: torch.Tensor,
                    target_lengths: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(mel)  # (B, T, d_model)
        logits = self.ctc_head(enc_out)  # (B, T, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, vocab_size)
        input_lengths = torch.full((mel.size(0),), log_probs.size(0),
                                   dtype=torch.long, device=mel.device)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          blank=0, zero_infinity=True)
        return loss

    def forward_e2e(self, mel: torch.Tensor, text_ids: torch.Tensor,
                    ctc_weight: float = 0.0, ctc_targets: torch.Tensor | None = None,
                    ctc_target_lengths: torch.Tensor | None = None) -> torch.Tensor:
        assert self.llm is not None, "LLM required for E2E forward"

        enc_out = self.encoder(mel)  # (B, T_audio, d_model)
        audio_embeds = self.projector(enc_out)  # (B, T_audio, llm_dim)

        # Replace -100 padding with 0 for embedding lookup (keep -100 only in labels)
        text_ids_safe = text_ids.clone()
        text_ids_safe[text_ids_safe < 0] = 0
        text_embeds = self.llm.model.embed_tokens(text_ids_safe)  # (B, T_text, llm_dim)

        # Concat [audio, text]
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

        B, T_audio = audio_embeds.shape[:2]

        # Labels: -100 for audio positions, text_ids for text positions
        ignore = torch.full((B, T_audio), -100, dtype=torch.long, device=mel.device)
        labels = torch.cat([ignore, text_ids], dim=1)

        outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
        ce_loss = outputs.loss

        if ctc_weight > 0.0 and ctc_targets is not None and ctc_target_lengths is not None:
            ctc_loss = self.forward_ctc(mel, ctc_targets, ctc_target_lengths)
            return (1.0 - ctc_weight) * ce_loss + ctc_weight * ctc_loss

        return ce_loss

    @torch.no_grad()
    def generate(self, mel: torch.Tensor, max_new_tokens: int = 200,
                 eos_token_id: int = 0) -> list[int]:
        assert self.llm is not None, "LLM required for generation"
        self.llm.eval()

        enc_out = self.encoder(mel)
        audio_embeds = self.projector(enc_out)  # (1, T_audio, llm_dim)

        # First pass with audio embeddings
        outputs = self.llm(inputs_embeds=audio_embeds, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)  # (1,)

        generated = [next_token_id.item()]

        for _ in range(max_new_tokens - 1):
            if generated[-1] == eos_token_id:
                break
            token_embeds = self.llm.model.embed_tokens(next_token_id.unsqueeze(0))
            outputs = self.llm(inputs_embeds=token_embeds, past_key_values=past_key_values,
                               use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token_id.item())

        return generated


# ============================================================================
# OmniAudio Scratch: encoder + decoder from scratch (no pretrained LLM)
# ============================================================================

class DecoderBlock(nn.Module):
    """Causal transformer decoder block with RoPE and SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        ffn_dim = int(d_model * 8 / 3)
        ffn_dim = ((ffn_dim + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        # Causal attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.o_proj(attn_out)
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class OmniAudioScratchModel(nn.Module):
    """Full model from scratch: encoder + projector + decoder, no pretrained LLM.

    Args:
        encoder_config: dict with n_mels, d_model, n_heads, n_layers, n_conv
        decoder_config: dict with d_model, n_heads, n_layers
        vocab_size: tokenizer vocabulary size
    """

    def __init__(self, encoder_config: dict, decoder_config: dict, vocab_size: int):
        super().__init__()
        enc_dim = encoder_config["d_model"]
        dec_dim = decoder_config["d_model"]
        dec_heads = decoder_config["n_heads"]
        dec_layers = decoder_config["n_layers"]

        self.encoder = AudioEncoderV2(**encoder_config)
        self.projector = AudioProjectorV2(enc_dim, dec_dim)
        self.ctc_head = nn.Linear(enc_dim, vocab_size)

        # Decoder from scratch
        self.embed_tokens = nn.Embedding(vocab_size, dec_dim)
        self.decoder_rope = RotaryEmbedding(dec_dim // dec_heads)
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(dec_dim, dec_heads) for _ in range(dec_layers)
        ])
        self.decoder_norm = RMSNorm(dec_dim)
        self.lm_head = nn.Linear(dec_dim, vocab_size, bias=False)
        # Tie embeddings
        self.lm_head.weight = self.embed_tokens.weight

    def forward_ctc(self, mel: torch.Tensor, targets: torch.Tensor,
                    target_lengths: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(mel)
        logits = self.ctc_head(enc_out)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = torch.full((mel.size(0),), log_probs.size(0),
                                   dtype=torch.long, device=mel.device)
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          blank=0, zero_infinity=True)

    def forward(self, mel: torch.Tensor, text_ids: torch.Tensor,
                ctc_weight: float = 0.0, ctc_targets: torch.Tensor | None = None,
                ctc_target_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Forward: encode audio, concat with text embeddings, decode causally."""
        enc_out = self.encoder(mel)
        audio_embeds = self.projector(enc_out)  # (B, T_audio, dec_dim)

        text_ids_safe = text_ids.clone()
        text_ids_safe[text_ids_safe < 0] = 0
        text_embeds = self.embed_tokens(text_ids_safe)  # (B, T_text, dec_dim)

        combined = torch.cat([audio_embeds, text_embeds], dim=1)  # (B, T_total, dec_dim)

        cos, sin = self.decoder_rope(combined.size(1))
        x = combined
        for layer in self.decoder_layers:
            x = layer(x, cos, sin)
        x = self.decoder_norm(x)
        logits = self.lm_head(x)

        # Loss only on text positions
        audio_len = audio_embeds.size(1)
        text_logits = logits[:, audio_len - 1:-1]  # predict text_ids from audio[-1] onward
        ce_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_ids.reshape(-1),
            ignore_index=-100,
            label_smoothing=0.1,
        )

        if ctc_weight > 0.0 and ctc_targets is not None and ctc_target_lengths is not None:
            ctc_loss = self.forward_ctc(mel, ctc_targets, ctc_target_lengths)
            return (1.0 - ctc_weight) * ce_loss + ctc_weight * ctc_loss

        return ce_loss

    @torch.no_grad()
    def generate(self, mel: torch.Tensor, max_new_tokens: int = 200,
                 eos_token_id: int = 0, repetition_penalty: float = 1.2) -> list[int]:
        self.eval()
        enc_out = self.encoder(mel)
        audio_embeds = self.projector(enc_out)  # (1, T_audio, dec_dim)

        generated: list[int] = []
        combined = audio_embeds  # start with just audio

        for _ in range(max_new_tokens):
            cos, sin = self.decoder_rope(combined.size(1))
            x = combined
            for layer in self.decoder_layers:
                x = layer(x, cos, sin)
            x = self.decoder_norm(x)
            logits = self.lm_head(x[:, -1:]).squeeze(0).squeeze(0)  # (vocab,)

            # Repetition penalty
            if repetition_penalty != 1.0 and generated:
                for prev_token in set(generated):
                    if logits[prev_token] > 0:
                        logits[prev_token] /= repetition_penalty
                    else:
                        logits[prev_token] *= repetition_penalty

            next_token = logits.argmax(dim=-1).item()
            if next_token == eos_token_id:
                break
            generated.append(next_token)
            token_embed = self.embed_tokens(torch.tensor([[next_token]], device=mel.device))
            combined = torch.cat([combined, token_embed], dim=1)

        return generated
