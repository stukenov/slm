# OmniAudio v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an ASR model that connects a custom audio encoder to a frozen pretrained Llama 150M decoder, with CTC auxiliary loss, A/B encoder configs, and deploy to kaznu for training.

**Architecture:** Audio (mel spectrogram) -> Custom Encoder (CNN+Transformer with RoPE) -> Projector -> Frozen Llama 150M (from HF). 3-stage training: CTC pretrain -> alignment -> end-to-end hybrid. Two encoder configs (Small 10M / Medium 25M) for A/B comparison.

**Tech Stack:** PyTorch, transformers (LlamaForCausalLM), torchaudio, datasets, jiwer, Ansible

**Spec:** `docs/superpowers/specs/2026-03-28-omniaudio-v2-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `omniaudio/src/omniaudio/model_v2.py` | AudioEncoderV2 (RoPE, configurable), AudioProjector, CTC head, OmniAudioV2Model (wraps HF Llama) |
| `omniaudio/src/omniaudio/augment.py` | SpecAugment and speed perturbation transforms |
| `omniaudio/src/omniaudio/data_v2.py` | AudioCollatorV2 with augmentation support and 50k tokenizer |
| `omniaudio/src/omniaudio/train_v2.py` | 3-stage training loop (CTC pretrain, alignment, end-to-end) |
| `omniaudio/src/omniaudio/evaluate_v2.py` | WER/CER assessment for v2 model |
| `omniaudio/configs/v2_base.yaml` | Base config for v2 |
| `omniaudio/configs/v2_small_s1.yaml` | Config S, stage 1 (CTC pretrain) |
| `omniaudio/configs/v2_small_s2.yaml` | Config S, stage 2 (alignment) |
| `omniaudio/configs/v2_small_s3.yaml` | Config S, stage 3 (end-to-end) |
| `omniaudio/configs/v2_medium_s1.yaml` | Config M, stage 1 |
| `omniaudio/configs/v2_medium_s2.yaml` | Config M, stage 2 |
| `omniaudio/configs/v2_medium_s3.yaml` | Config M, stage 3 |
| `omniaudio/tests/test_model_v2.py` | Unit tests for v2 model |
| `omniaudio/tests/test_augment.py` | Unit tests for augmentation |
| `ansible/run_omniaudio_v2.yml` | Ansible playbook for deploying and running on kaznu |

---

### Task 1: AudioEncoderV2 with RoPE

**Files:**
- Create: `omniaudio/src/omniaudio/model_v2.py`
- Create: `omniaudio/tests/test_model_v2.py`

- [ ] **Step 1: Write failing tests for RotaryEmbedding**

```python
# omniaudio/tests/test_model_v2.py
import torch
from omniaudio.model_v2 import RotaryEmbedding

def test_rotary_embedding_shape():
    rope = RotaryEmbedding(dim=64)
    cos, sin = rope(seq_len=100)
    assert cos.shape == (100, 64)
    assert sin.shape == (100, 64)

def test_rotary_embedding_different_lengths():
    rope = RotaryEmbedding(dim=64)
    cos1, sin1 = rope(seq_len=50)
    cos2, sin2 = rope(seq_len=100)
    assert torch.allclose(cos1, cos2[:50])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd omniaudio && python -m pytest tests/test_model_v2.py::test_rotary_embedding_shape -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement RotaryEmbedding**

```python
# omniaudio/src/omniaudio/model_v2.py
"""OmniAudio v2: ASR with pretrained Llama 150M decoder."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos(), freqs.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query or key tensor. x: (B, heads, seq, head_dim)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd omniaudio && python -m pytest tests/test_model_v2.py -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for AudioEncoderV2**

Add to `omniaudio/tests/test_model_v2.py`:

```python
from omniaudio.model_v2 import AudioEncoderV2

def test_encoder_v2_small_shape():
    enc = AudioEncoderV2(n_mels=80, d_model=256, n_heads=4, n_layers=6, n_conv=2)
    mel = torch.randn(2, 80, 1000)
    out = enc(mel)
    assert out.dim() == 3
    assert out.size(0) == 2
    assert out.size(2) == 256
    assert out.size(1) == 250  # 1000 / 4x downsampling

def test_encoder_v2_medium_shape():
    enc = AudioEncoderV2(n_mels=80, d_model=384, n_heads=6, n_layers=8, n_conv=3)
    mel = torch.randn(2, 80, 1000)
    out = enc(mel)
    assert out.size(2) == 384
    assert out.size(1) == 125  # 1000 / 8x downsampling

def test_encoder_v2_param_count_small():
    enc = AudioEncoderV2(n_mels=80, d_model=256, n_heads=4, n_layers=6, n_conv=2)
    total = sum(p.numel() for p in enc.parameters())
    assert 5_000_000 < total < 15_000_000, f"Small encoder params {total}"

def test_encoder_v2_param_count_medium():
    enc = AudioEncoderV2(n_mels=80, d_model=384, n_heads=6, n_layers=8, n_conv=3)
    total = sum(p.numel() for p in enc.parameters())
    assert 15_000_000 < total < 35_000_000, f"Medium encoder params {total}"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd omniaudio && python -m pytest tests/test_model_v2.py::test_encoder_v2_small_shape -v`
Expected: FAIL with ImportError

- [ ] **Step 7: Implement AudioEncoderV2**

Add to `omniaudio/src/omniaudio/model_v2.py`:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class EncoderBlock(nn.Module):
    """Transformer encoder block with RoPE and SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = RMSNorm(d_model)
        ff_dim = d_model * 4
        self.gate_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.o_proj(attn)
        h = self.ffn_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class AudioEncoderV2(nn.Module):
    """Audio encoder: Conv1d downsampling + Transformer with RoPE."""

    def __init__(self, n_mels: int = 80, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 6, n_conv: int = 2, dropout: float = 0.1):
        super().__init__()
        convs = []
        in_ch = n_mels
        for _ in range(n_conv):
            convs.append(nn.Conv1d(in_ch, d_model, kernel_size=3, stride=2, padding=1))
            convs.append(nn.GELU())
            in_ch = d_model
        self.conv_stack = nn.Sequential(*convs)
        self.ln = RMSNorm(d_model)
        self.rope = RotaryEmbedding(d_model // n_heads)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(mel)
        x = x.transpose(1, 2)
        x = self.ln(x)
        cos, sin = self.rope(x.size(1))
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x
```

- [ ] **Step 8: Run all encoder tests**

Run: `cd omniaudio && python -m pytest tests/test_model_v2.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add omniaudio/src/omniaudio/model_v2.py omniaudio/tests/test_model_v2.py
git commit -m "feat(omniaudio): add AudioEncoderV2 with RoPE and configurable conv stack"
```

---

### Task 2: OmniAudioV2Model with HF Llama and CTC head

**Files:**
- Modify: `omniaudio/src/omniaudio/model_v2.py`
- Modify: `omniaudio/tests/test_model_v2.py`

- [ ] **Step 1: Write failing tests for projector and full model**

Add to `omniaudio/tests/test_model_v2.py`:

```python
from omniaudio.model_v2 import AudioProjectorV2, OmniAudioV2Model

def test_projector_v2():
    proj = AudioProjectorV2(audio_dim=256, llm_dim=768)
    x = torch.randn(2, 100, 256)
    out = proj(x)
    assert out.shape == (2, 100, 768)

def test_omniaudio_v2_forward_ctc():
    model = OmniAudioV2Model(
        encoder_config=dict(n_mels=80, d_model=256, n_heads=4, n_layers=2, n_conv=2),
        llm_name=None, vocab_size=100,
    )
    mel = torch.randn(2, 80, 500)
    targets = torch.randint(1, 100, (2, 20))
    target_lengths = torch.tensor([20, 15])
    loss = model.forward_ctc(mel, targets, target_lengths)
    assert loss.dim() == 0
    assert loss.item() > 0

def test_omniaudio_v2_forward_e2e():
    from transformers import LlamaConfig, LlamaForCausalLM
    tiny_config = LlamaConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    tiny_llm = LlamaForCausalLM(tiny_config)
    model = OmniAudioV2Model(
        encoder_config=dict(n_mels=80, d_model=32, n_heads=2, n_layers=2, n_conv=2),
        llm_name=None, vocab_size=100, llm_dim=64,
    )
    model.llm = tiny_llm
    for p in model.llm.parameters():
        p.requires_grad = False
    mel = torch.randn(2, 80, 200)
    text_ids = torch.randint(0, 100, (2, 10))
    loss = model.forward_e2e(mel, text_ids)
    assert loss.dim() == 0
    assert loss.item() > 0

def test_omniaudio_v2_generate():
    from transformers import LlamaConfig, LlamaForCausalLM
    tiny_config = LlamaConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    tiny_llm = LlamaForCausalLM(tiny_config)
    model = OmniAudioV2Model(
        encoder_config=dict(n_mels=80, d_model=32, n_heads=2, n_layers=2, n_conv=2),
        llm_name=None, vocab_size=100, llm_dim=64,
    )
    model.llm = tiny_llm
    mel = torch.randn(1, 80, 200)
    tokens = model.generate(mel, max_new_tokens=10, eos_token_id=0)
    assert isinstance(tokens, list)
    assert len(tokens) <= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd omniaudio && python -m pytest tests/test_model_v2.py::test_projector_v2 -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement AudioProjectorV2 and OmniAudioV2Model**

Add to `omniaudio/src/omniaudio/model_v2.py`:

```python
class AudioProjectorV2(nn.Module):
    def __init__(self, audio_dim: int, llm_dim: int):
        super().__init__()
        self.linear = nn.Linear(audio_dim, llm_dim)
        self.norm = RMSNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class OmniAudioV2Model(nn.Module):
    """OmniAudio v2: audio encoder + projector + frozen pretrained Llama decoder."""

    def __init__(self, encoder_config: dict, llm_name: str | None,
                 vocab_size: int, llm_dim: int = 768):
        super().__init__()
        self.encoder = AudioEncoderV2(**encoder_config)
        self.projector = AudioProjectorV2(encoder_config["d_model"], llm_dim)
        self.ctc_head = nn.Linear(encoder_config["d_model"], vocab_size)
        self.llm = None
        if llm_name:
            from transformers import LlamaForCausalLM
            self.llm = LlamaForCausalLM.from_pretrained(llm_name)
            for p in self.llm.parameters():
                p.requires_grad = False

    def forward_ctc(self, mel: torch.Tensor, targets: torch.Tensor,
                    target_lengths: torch.Tensor) -> torch.Tensor:
        """CTC loss on encoder output (stage 1)."""
        enc_out = self.encoder(mel)
        logits = self.ctc_head(enc_out)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, vocab)
        input_lengths = torch.full((mel.size(0),), log_probs.size(0), dtype=torch.long)
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=True)

    def forward_e2e(self, mel: torch.Tensor, text_ids: torch.Tensor,
                    ctc_weight: float = 0.0, ctc_targets: torch.Tensor | None = None,
                    ctc_target_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """End-to-end forward: encoder -> projector -> frozen LLM decoder."""
        enc_out = self.encoder(mel)
        audio_embeds = self.projector(enc_out)
        text_embeds = self.llm.model.embed_tokens(text_ids)
        combined = torch.cat([audio_embeds, text_embeds], dim=1)

        audio_len = audio_embeds.size(1)
        total_len = combined.size(1)

        labels = torch.full((mel.size(0), total_len), -100, dtype=torch.long, device=mel.device)
        labels[:, audio_len:] = text_ids

        outputs = self.llm(inputs_embeds=combined, labels=labels)
        ce_loss = outputs.loss

        if ctc_weight > 0 and ctc_targets is not None:
            ctc_loss = self.forward_ctc(mel, ctc_targets, ctc_target_lengths)
            return (1 - ctc_weight) * ce_loss + ctc_weight * ctc_loss
        return ce_loss

    @torch.no_grad()
    def generate(self, mel: torch.Tensor, max_new_tokens: int = 200,
                 eos_token_id: int = 0) -> list[int]:
        self.llm.eval() if self.llm else None
        enc_out = self.encoder(mel)
        audio_embeds = self.projector(enc_out)
        generated: list[int] = []
        outputs = self.llm(inputs_embeds=audio_embeds, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1).squeeze().item()
        if next_token == eos_token_id:
            return generated
        generated.append(next_token)

        for _ in range(max_new_tokens - 1):
            input_ids = torch.tensor([[next_token]], device=mel.device)
            outputs = self.llm(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:].argmax(dim=-1).squeeze().item()
            if next_token == eos_token_id:
                break
            generated.append(next_token)
        return generated
```

- [ ] **Step 4: Run all tests**

Run: `cd omniaudio && python -m pytest tests/test_model_v2.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add omniaudio/src/omniaudio/model_v2.py omniaudio/tests/test_model_v2.py
git commit -m "feat(omniaudio): add OmniAudioV2Model with CTC head and HF Llama integration"
```

---

### Task 3: Data augmentation (SpecAugment + speed perturbation)

**Files:**
- Create: `omniaudio/src/omniaudio/augment.py`
- Create: `omniaudio/tests/test_augment.py`

- [ ] **Step 1: Write failing tests**

```python
# omniaudio/tests/test_augment.py
import torch
from omniaudio.augment import spec_augment, speed_perturb

def test_spec_augment_shape():
    mel = torch.randn(80, 500)
    augmented = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                             num_freq_masks=2, num_time_masks=2)
    assert augmented.shape == mel.shape

def test_spec_augment_has_zeros():
    torch.manual_seed(42)
    mel = torch.ones(80, 500)
    augmented = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                             num_freq_masks=2, num_time_masks=2)
    assert (augmented == 0).any()

def test_spec_augment_no_masks():
    mel = torch.randn(80, 500)
    augmented = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                             num_freq_masks=0, num_time_masks=0)
    assert torch.equal(mel, augmented)

def test_speed_perturb_shape():
    waveform = torch.randn(16000)
    perturbed = speed_perturb(waveform, sample_rate=16000, factor=0.9)
    assert perturbed.shape[0] > waveform.shape[0]

def test_speed_perturb_identity():
    waveform = torch.randn(16000)
    perturbed = speed_perturb(waveform, sample_rate=16000, factor=1.0)
    assert perturbed.shape[0] == waveform.shape[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd omniaudio && python -m pytest tests/test_augment.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement augmentations**

```python
# omniaudio/src/omniaudio/augment.py
"""Data augmentation for audio: SpecAugment and speed perturbation."""

import random
import torch
import torchaudio


def spec_augment(mel: torch.Tensor, freq_mask_param: int = 27, time_mask_param: int = 100,
                 num_freq_masks: int = 2, num_time_masks: int = 2) -> torch.Tensor:
    """Apply SpecAugment to mel spectrogram. mel: (n_mels, time)."""
    augmented = mel.clone()
    n_mels, n_time = augmented.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, n_mels - 1))
        f0 = random.randint(0, n_mels - f)
        augmented[f0:f0 + f, :] = 0

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, n_time - 1))
        t0 = random.randint(0, n_time - t)
        augmented[:, t0:t0 + t] = 0

    return augmented


def speed_perturb(waveform: torch.Tensor, sample_rate: int = 16000, factor: float = 1.0) -> torch.Tensor:
    """Speed perturbation via resampling. waveform: (samples,)."""
    if factor == 1.0:
        return waveform
    new_sr = int(sample_rate * factor)
    return torchaudio.functional.resample(waveform, new_sr, sample_rate)
```

- [ ] **Step 4: Run tests**

Run: `cd omniaudio && python -m pytest tests/test_augment.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add omniaudio/src/omniaudio/augment.py omniaudio/tests/test_augment.py
git commit -m "feat(omniaudio): add SpecAugment and speed perturbation"
```

---

### Task 4: Data pipeline v2

**Files:**
- Create: `omniaudio/src/omniaudio/data_v2.py`
- Create: `omniaudio/tests/test_data_v2.py`

- [ ] **Step 1: Write failing tests**

```python
# omniaudio/tests/test_data_v2.py
import torch
from omniaudio.data_v2 import AudioCollatorV2

def test_collator_v2_output_keys():
    collator = AudioCollatorV2(
        tokenizer_path="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        n_mels=80, sample_rate=16000, max_audio_len=15.0, max_text_len=256,
        augment=False,
    )
    fake_sample = {
        "audio": {"array": torch.randn(16000).numpy(), "sampling_rate": 16000},
        "sentence": "test text",
    }
    batch = collator([fake_sample])
    assert "mel" in batch
    assert "text_ids" in batch
    assert "ctc_targets" in batch
    assert "ctc_target_lengths" in batch

def test_collator_v2_mel_shape():
    collator = AudioCollatorV2(
        tokenizer_path="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        n_mels=80, sample_rate=16000, max_audio_len=15.0, max_text_len=256,
        augment=False,
    )
    fake_sample = {
        "audio": {"array": torch.randn(16000).numpy(), "sampling_rate": 16000},
        "sentence": "test",
    }
    batch = collator([fake_sample, fake_sample])
    assert batch["mel"].dim() == 3
    assert batch["mel"].size(0) == 2
    assert batch["mel"].size(1) == 80

def test_collator_v2_ctc_targets_valid():
    collator = AudioCollatorV2(
        tokenizer_path="saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1",
        n_mels=80, sample_rate=16000, max_audio_len=15.0, max_text_len=256,
        augment=False,
    )
    fake_sample = {
        "audio": {"array": torch.randn(16000).numpy(), "sampling_rate": 16000},
        "sentence": "test",
    }
    batch = collator([fake_sample])
    assert (batch["ctc_targets"] >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd omniaudio && python -m pytest tests/test_data_v2.py::test_collator_v2_output_keys -v`
Expected: FAIL

- [ ] **Step 3: Implement data_v2.py**

```python
# omniaudio/src/omniaudio/data_v2.py
"""Data loading and collation for OmniAudio v2."""

import random
import torch
import torchaudio
from datasets import load_dataset
from transformers import AutoTokenizer
from omniaudio.augment import spec_augment, speed_perturb


def load_commonvoice_kk(split="train", max_samples=None):
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "kk",
                      split=split, trust_remote_code=True)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


class AudioCollatorV2:
    """Collate audio+text for OmniAudio v2. Produces mel, text_ids, and CTC targets."""

    SPEED_FACTORS = [0.9, 1.0, 1.1]

    def __init__(self, tokenizer_path: str, n_mels: int = 80, sample_rate: int = 16000,
                 max_audio_len: float = 15.0, max_text_len: int = 256, augment: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.augment = augment
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160,
        )

    def __call__(self, batch):
        max_audio_samples = int(self.max_audio_len * self.sample_rate)
        mels, text_ids_list, ctc_targets_list = [], [], []

        for sample in batch:
            audio = sample["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32)
            sr = audio["sampling_rate"]

            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            if self.augment:
                factor = random.choice(self.SPEED_FACTORS)
                waveform = speed_perturb(waveform, self.sample_rate, factor)

            waveform = waveform[:max_audio_samples]
            mel = self.mel_transform(waveform.unsqueeze(0))
            mel = torch.log(torch.clamp(mel, min=1e-10)).squeeze(0)

            if self.augment:
                mel = spec_augment(mel, freq_mask_param=27, time_mask_param=100,
                                   num_freq_masks=2, num_time_masks=2)
            mels.append(mel)

            tokens = self.tokenizer(sample["sentence"], max_length=self.max_text_len,
                                    truncation=True, return_tensors="pt")
            ids = tokens["input_ids"].squeeze(0)
            text_ids_list.append(ids)
            ctc_targets_list.append(ids.clone())

        max_t = max(m.shape[1] for m in mels)
        padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_t)
        for i, m in enumerate(mels):
            padded_mels[i, :, :m.shape[1]] = m

        padded_text = torch.nn.utils.rnn.pad_sequence(text_ids_list, batch_first=True, padding_value=-100)
        ctc_target_lengths = torch.tensor([len(t) for t in ctc_targets_list])
        padded_ctc = torch.nn.utils.rnn.pad_sequence(ctc_targets_list, batch_first=True, padding_value=0)

        return {
            "mel": padded_mels,
            "text_ids": padded_text,
            "ctc_targets": padded_ctc,
            "ctc_target_lengths": ctc_target_lengths,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd omniaudio && python -m pytest tests/test_data_v2.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add omniaudio/src/omniaudio/data_v2.py omniaudio/tests/test_data_v2.py
git commit -m "feat(omniaudio): add data pipeline v2 with augmentation and CTC targets"
```

---

### Task 5: Training script v2 (3-stage)

**Files:**
- Create: `omniaudio/src/omniaudio/train_v2.py`

- [ ] **Step 1: Implement train_v2.py**

```python
# omniaudio/src/omniaudio/train_v2.py
"""Training script for OmniAudio v2: 3-stage (CTC pretrain, alignment, E2E)."""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from omniaudio.data_v2 import AudioCollatorV2, load_commonvoice_kk
from omniaudio.model_v2 import OmniAudioV2Model

logger = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    inherits = config.pop("inherits", None)
    if inherits:
        base_path = Path(path).parent / f"{inherits}.yaml"
        base = load_config(base_path)
        base.update(config)
        return base
    return config


def get_trainable_params(model, config):
    stage = config["stage"]
    for p in model.parameters():
        p.requires_grad = False

    if stage == "ctc_pretrain":
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.ctc_head.parameters():
            p.requires_grad = True
    elif stage == "alignment":
        for p in model.projector.parameters():
            p.requires_grad = True
    elif stage == "e2e":
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.projector.parameters():
            p.requires_grad = True
        for p in model.ctc_head.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Stage: %s | Params: %.2fM total, %.2fM trainable", stage, total / 1e6, trainable / 1e6)


def train(config):
    experiment_name = config["experiment_name"]
    output_dir = Path(config["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = config["stage"]
    logger.info("Device: %s | Stage: %s", device, stage)

    encoder_config = {
        "n_mels": config["n_mels"],
        "d_model": config["audio_d_model"],
        "n_heads": config["audio_n_heads"],
        "n_layers": config["audio_n_layers"],
        "n_conv": config["audio_n_conv"],
    }

    llm_name = config.get("llm_name") if stage != "ctc_pretrain" else None
    model = OmniAudioV2Model(
        encoder_config=encoder_config, llm_name=llm_name,
        vocab_size=config["vocab_size"], llm_dim=config.get("llm_dim", 768),
    )

    init_from = config.get("init_from")
    if init_from:
        ckpt = Path(init_from) / "model.pt"
        logger.info("Loading checkpoint: %s", ckpt)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("Loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))

    get_trainable_params(model, config)
    model = model.to(device)
    use_bf16 = config.get("bf16", True) and device.type == "cuda"

    augment = config.get("augment", stage != "ctc_pretrain")
    collator = AudioCollatorV2(
        tokenizer_path=config["tokenizer_path"], n_mels=config["n_mels"],
        sample_rate=config["sample_rate"], max_audio_len=config["max_audio_len"],
        max_text_len=config["max_text_len"], augment=augment,
    )
    train_ds = load_commonvoice_kk("train", max_samples=config.get("max_train_samples"))
    val_ds = load_commonvoice_kk("validation", max_samples=config.get("max_eval_samples"))
    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    batch_size = config["per_device_train_batch_size"]
    num_workers = config.get("dataloader_num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collator,
                            num_workers=num_workers)

    grad_accum = config.get("gradient_accumulation_steps", 1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    num_epochs = config["num_train_epochs"]
    total_steps = len(train_loader) * num_epochs // grad_accum
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.05)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    ctc_weight = float(config.get("ctc_weight", 0.0))

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            mel = batch["mel"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if stage == "ctc_pretrain":
                    loss = model.forward_ctc(mel, batch["ctc_targets"].to(device),
                                             batch["ctc_target_lengths"].to(device))
                else:
                    text_ids = batch["text_ids"].to(device)
                    ctc_t = batch["ctc_targets"].to(device) if ctc_weight > 0 else None
                    ctc_l = batch["ctc_target_lengths"].to(device) if ctc_weight > 0 else None
                    loss = model.forward_e2e(mel, text_ids, ctc_weight=ctc_weight,
                                            ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
                loss = loss / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
            num_batches += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.get("logging_steps", 50) == 0:
                    avg = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    logger.info("Step %d | Loss: %.4f | LR: %.2e", global_step, avg, lr)

                save_steps = config.get("save_steps")
                if save_steps and global_step % save_steps == 0:
                    _save_checkpoint(model, output_dir, global_step)

        avg_train = epoch_loss / max(num_batches, 1)
        logger.info("Epoch %d/%d | Train loss: %.4f", epoch + 1, num_epochs, avg_train)

        val_loss = _run_validation(model, val_loader, device, stage, use_bf16, ctc_weight)
        logger.info("Epoch %d/%d | Val loss: %.4f", epoch + 1, num_epochs, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, output_dir, "best")
            logger.info("New best (val_loss=%.4f)", val_loss)

    _save_checkpoint(model, output_dir, "final")
    logger.info("Done: %s", experiment_name)


def _run_validation(model, val_loader, device, stage, use_bf16, ctc_weight):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if stage == "ctc_pretrain":
                    loss = model.forward_ctc(mel, batch["ctc_targets"].to(device),
                                             batch["ctc_target_lengths"].to(device))
                else:
                    text_ids = batch["text_ids"].to(device)
                    ctc_t = batch["ctc_targets"].to(device) if ctc_weight > 0 else None
                    ctc_l = batch["ctc_target_lengths"].to(device) if ctc_weight > 0 else None
                    loss = model.forward_e2e(mel, text_ids, ctc_weight=ctc_weight,
                                            ctc_targets=ctc_t, ctc_target_lengths=ctc_l)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def _save_checkpoint(model, output_dir, step):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(exist_ok=True)
    state = {name: param.data for name, param in model.named_parameters() if not name.startswith("llm.")}
    torch.save(state, ckpt_dir / "model.pt")
    logger.info("Saved checkpoint-%s", step)


def main():
    parser = argparse.ArgumentParser(description="OmniAudio v2 Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add omniaudio/src/omniaudio/train_v2.py
git commit -m "feat(omniaudio): add 3-stage training script (CTC, alignment, E2E)"
```

---

### Task 6: Assessment script v2

**Files:**
- Create: `omniaudio/src/omniaudio/evaluate_v2.py`

- [ ] **Step 1: Implement evaluate_v2.py**

```python
# omniaudio/src/omniaudio/evaluate_v2.py
"""WER/CER measurement for OmniAudio v2 on Common Voice kk test."""

import argparse
import logging
import random
from pathlib import Path

import torch
from jiwer import cer, wer
from transformers import AutoTokenizer

from omniaudio.data_v2 import AudioCollatorV2, load_commonvoice_kk
from omniaudio.model_v2 import OmniAudioV2Model
from omniaudio.train_v2 import load_config

logger = logging.getLogger(__name__)


def run_assessment(config, model_path, max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_config = {
        "n_mels": config["n_mels"], "d_model": config["audio_d_model"],
        "n_heads": config["audio_n_heads"], "n_layers": config["audio_n_layers"],
        "n_conv": config["audio_n_conv"],
    }

    model = OmniAudioV2Model(
        encoder_config=encoder_config, llm_name=config["llm_name"],
        vocab_size=config["vocab_size"], llm_dim=config.get("llm_dim", 768),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    test_ds = load_commonvoice_kk("test", max_samples=max_samples)
    collator = AudioCollatorV2(
        tokenizer_path=config["tokenizer_path"], n_mels=config["n_mels"],
        sample_rate=config["sample_rate"], max_audio_len=config["max_audio_len"],
        max_text_len=config["max_text_len"], augment=False,
    )

    all_refs, all_hyps = [], []
    logger.info("Running on %d samples...", len(test_ds))

    with torch.no_grad():
        for i, sample in enumerate(test_ds):
            batch = collator([sample])
            mel = batch["mel"].to(device)
            tokens = model.generate(mel, max_new_tokens=config.get("max_text_len", 256),
                                    eos_token_id=tokenizer.eos_token_id or 0)
            hyp = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            ref = sample["sentence"].strip()
            all_refs.append(ref)
            all_hyps.append(hyp)
            if (i + 1) % 100 == 0:
                logger.info("Processed %d/%d", i + 1, len(test_ds))

    results = {"wer": wer(all_refs, all_hyps), "cer": cer(all_refs, all_hyps), "n": len(all_refs)}

    print(f"\n{'='*40}")
    print(f"OmniAudio v2 Results")
    print(f"{'='*40}")
    print(f"Samples:  {results['n']}")
    print(f"WER:      {results['wer']:.2%}")
    print(f"CER:      {results['cer']:.2%}")
    print(f"{'='*40}\n")

    indices = random.sample(range(len(all_refs)), min(5, len(all_refs)))
    for idx in indices:
        print(f"REF: {all_refs[idx]}")
        print(f"HYP: {all_hyps[idx]}\n")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)
    run_assessment(config, args.model_path, args.max_samples)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add omniaudio/src/omniaudio/evaluate_v2.py
git commit -m "feat(omniaudio): add WER/CER measurement script for v2"
```

---

### Task 7: YAML configs for A/B test

**Files:**
- Create: `omniaudio/configs/v2_base.yaml`
- Create: `omniaudio/configs/v2_small_s1.yaml`, `v2_small_s2.yaml`, `v2_small_s3.yaml`
- Create: `omniaudio/configs/v2_medium_s1.yaml`, `v2_medium_s2.yaml`, `v2_medium_s3.yaml`

- [ ] **Step 1: Create all config files**

`omniaudio/configs/v2_base.yaml`:
```yaml
seed: 42
output_dir: ./outputs
logging_dir: ./logs
n_mels: 80
sample_rate: 16000
max_audio_len: 15.0
max_text_len: 256
llm_name: saken-tukenov/sozkz-core-llama-150m-kk-base-v1
llm_dim: 768
vocab_size: 50257
tokenizer_path: saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1
bf16: true
dataloader_num_workers: 4
max_grad_norm: 1.0
label_smoothing: 0.1
```

`omniaudio/configs/v2_small_s1.yaml`:
```yaml
inherits: v2_base
experiment_name: omniaudio_v2_small_s1_ctc
stage: ctc_pretrain
audio_d_model: 256
audio_n_heads: 4
audio_n_layers: 6
audio_n_conv: 2
augment: false
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
learning_rate: 1e-3
warmup_ratio: 0.10
weight_decay: 0.01
num_train_epochs: 20
save_steps: 1000
logging_steps: 50
```

`omniaudio/configs/v2_small_s2.yaml`:
```yaml
inherits: v2_base
experiment_name: omniaudio_v2_small_s2_align
stage: alignment
audio_d_model: 256
audio_n_heads: 4
audio_n_layers: 6
audio_n_conv: 2
init_from: ./outputs/omniaudio_v2_small_s1_ctc/final
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
learning_rate: 1e-3
warmup_ratio: 0.05
weight_decay: 0.01
num_train_epochs: 5
save_steps: 500
logging_steps: 50
```

`omniaudio/configs/v2_small_s3.yaml`:
```yaml
inherits: v2_base
experiment_name: omniaudio_v2_small_s3_e2e
stage: e2e
audio_d_model: 256
audio_n_heads: 4
audio_n_layers: 6
audio_n_conv: 2
ctc_weight: 0.3
init_from: ./outputs/omniaudio_v2_small_s2_align/final
per_device_train_batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 2e-5
warmup_ratio: 0.05
weight_decay: 0.01
num_train_epochs: 15
save_steps: 500
eval_steps: 500
logging_steps: 50
```

`omniaudio/configs/v2_medium_s1.yaml`:
```yaml
inherits: v2_base
experiment_name: omniaudio_v2_medium_s1_ctc
stage: ctc_pretrain
audio_d_model: 384
audio_n_heads: 6
audio_n_layers: 8
audio_n_conv: 3
augment: false
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
learning_rate: 1e-3
warmup_ratio: 0.10
weight_decay: 0.01
num_train_epochs: 20
save_steps: 1000
logging_steps: 50
```

`omniaudio/configs/v2_medium_s2.yaml`:
```yaml
inherits: v2_base
experiment_name: omniaudio_v2_medium_s2_align
stage: alignment
audio_d_model: 384
audio_n_heads: 6
audio_n_layers: 8
audio_n_conv: 3
init_from: ./outputs/omniaudio_v2_medium_s1_ctc/final
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
learning_rate: 1e-3
warmup_ratio: 0.05
weight_decay: 0.01
num_train_epochs: 5
save_steps: 500
logging_steps: 50
```

`omniaudio/configs/v2_medium_s3.yaml`:
```yaml
inherits: v2_base
experiment_name: omniaudio_v2_medium_s3_e2e
stage: e2e
audio_d_model: 384
audio_n_heads: 6
audio_n_layers: 8
audio_n_conv: 3
ctc_weight: 0.3
init_from: ./outputs/omniaudio_v2_medium_s2_align/final
per_device_train_batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 2e-5
warmup_ratio: 0.05
weight_decay: 0.01
num_train_epochs: 15
save_steps: 500
eval_steps: 500
logging_steps: 50
```

- [ ] **Step 2: Commit**

```bash
git add omniaudio/configs/v2_*.yaml
git commit -m "feat(omniaudio): add v2 configs for Small/Medium A/B test (3 stages each)"
```

---

### Task 8: Ansible playbook for kaznu deployment

**Files:**
- Create: `ansible/run_omniaudio_v2.yml`

- [ ] **Step 1: Create playbook**

```yaml
# ansible/run_omniaudio_v2.yml
# Usage:
#   ansible-playbook ansible/run_omniaudio_v2.yml -i ansible/inventory.ini \
#     -e config=v2_small_s1 -e screen_name=omniaudio_v2_small_s1
#   ansible-playbook ansible/run_omniaudio_v2.yml -i ansible/inventory.ini \
#     -e action=logs -e screen_name=omniaudio_v2_small_s1
---
- hosts: kaznu
  vars:
    work_dir: /root/slm
    venv: /root/slm/.venv
    config: v2_small_s1
    screen_name: "omniaudio_v2_{{ config }}"
    action: train
  tasks:
    - name: Sync omniaudio code
      ansible.posix.synchronize:
        src: "{{ playbook_dir }}/../omniaudio/"
        dest: "{{ work_dir }}/omniaudio/"
        rsync_opts:
          - "--exclude=.venv"
          - "--exclude=__pycache__"
          - "--exclude=outputs"
          - "--exclude=logs"
          - "--exclude=*.pyc"
      when: action == "train"

    - name: Install omniaudio
      shell: "cd {{ work_dir }} && {{ venv }}/bin/pip install -e omniaudio/ --quiet 2>&1 | tail -3"
      when: action == "train"

    - name: Start training in screen
      shell: |
        screen -dmS {{ screen_name }} bash -c '
          cd {{ work_dir }} &&
          {{ venv }}/bin/python -m omniaudio.train_v2 \
            --config omniaudio/configs/{{ config }}.yaml \
            2>&1 | tee logs/{{ screen_name }}.log
        '
      when: action == "train"

    - name: Training started
      debug:
        msg: "Screen '{{ screen_name }}' launched. Monitor: ssh kaznu \"tail -f {{ work_dir }}/logs/{{ screen_name }}.log\""
      when: action == "train"

    - name: Show logs
      shell: "tail -30 {{ work_dir }}/logs/{{ screen_name }}.log 2>/dev/null || echo 'No log yet'"
      register: log_out
      when: action == "logs"

    - name: Display
      debug:
        msg: "{{ log_out.stdout_lines }}"
      when: action == "logs"

    - name: Check status
      shell: "screen -ls | grep {{ screen_name }} || echo 'No session'"
      register: status_out
      when: action == "status"

    - name: Show status
      debug:
        msg: "{{ status_out.stdout }}"
      when: action == "status"
```

- [ ] **Step 2: Commit**

```bash
git add ansible/run_omniaudio_v2.yml
git commit -m "feat(ansible): add OmniAudio v2 deployment playbook for kaznu"
```

---

### Task 9: Local smoke test

- [ ] **Step 1: Run all unit tests**

```bash
cd omniaudio && python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Smoke test with tiny model**

```bash
cd omniaudio && python -c "
import torch
from omniaudio.model_v2 import OmniAudioV2Model
from transformers import LlamaConfig, LlamaForCausalLM

model = OmniAudioV2Model(
    encoder_config=dict(n_mels=80, d_model=32, n_heads=2, n_layers=2, n_conv=2),
    llm_name=None, vocab_size=100, llm_dim=32,
)
cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                  max_position_embeddings=512)
model.llm = LlamaForCausalLM(cfg)
for p in model.llm.parameters():
    p.requires_grad = False

mel = torch.randn(2, 80, 500)

targets = torch.randint(1, 100, (2, 15))
target_lengths = torch.tensor([15, 10])
ctc_loss = model.forward_ctc(mel, targets, target_lengths)
print(f'CTC loss: {ctc_loss.item():.4f}')

text_ids = torch.randint(0, 100, (2, 20))
e2e_loss = model.forward_e2e(mel, text_ids)
print(f'E2E loss: {e2e_loss.item():.4f}')

tokens = model.generate(mel[:1], max_new_tokens=5, eos_token_id=0)
print(f'Generated {len(tokens)} tokens: {tokens}')
print('SMOKE TEST PASSED')
"
```

Expected: Prints "SMOKE TEST PASSED"

- [ ] **Step 3: Final commit**

```bash
git add -A omniaudio/
git commit -m "feat(omniaudio): OmniAudio v2 complete - ready for kaznu deployment"
```

---

### Task 10: Deploy and launch on kaznu

- [ ] **Step 1: Deploy and start Config S Stage 1**

```bash
ansible-playbook ansible/run_omniaudio_v2.yml -i ansible/inventory.ini \
  -e config=v2_small_s1 -e screen_name=omniaudio_v2_small_s1
```

- [ ] **Step 2: Verify training started**

```bash
ansible-playbook ansible/run_omniaudio_v2.yml -i ansible/inventory.ini \
  -e action=logs -e screen_name=omniaudio_v2_small_s1
```

- [ ] **Step 3: After S1 completes, launch S2 (alignment)**

```bash
ansible-playbook ansible/run_omniaudio_v2.yml -i ansible/inventory.ini \
  -e config=v2_small_s2 -e screen_name=omniaudio_v2_small_s2
```

- [ ] **Step 4: After S2, launch S3 (E2E)**

```bash
ansible-playbook ansible/run_omniaudio_v2.yml -i ansible/inventory.ini \
  -e config=v2_small_s3 -e screen_name=omniaudio_v2_small_s3
```

- [ ] **Step 5: Repeat for Config M (Medium)**

```bash
ansible-playbook ansible/run_omniaudio_v2.yml -i ansible/inventory.ini \
  -e config=v2_medium_s1 -e screen_name=omniaudio_v2_medium_s1
# ... then s2 and s3
```
