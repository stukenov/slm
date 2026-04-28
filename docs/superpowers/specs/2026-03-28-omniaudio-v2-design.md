# OmniAudio v2 — Native ASR with Pretrained Llama 150M Decoder

## Goal

Modify the existing OmniAudio subproject to use the **pretrained Llama 150M** (`saken-tukenov/sozkz-core-llama-150m-kk-base-v1`) as a frozen decoder instead of a random-init Llama 50M. Train only the audio encoder + projector. A/B test two encoder configurations.

## Architecture

```
Audio (16kHz mono)
  -> Mel Spectrogram (80 bins, hop=160, n_fft=400)
  -> [Audio Encoder: Conv1d downsampling + Transformer layers]   trainable
  -> [Projector: Linear(enc_dim -> 768) + LayerNorm]             trainable
  -> prefix tokens
  -> [Llama 150M from HF (FROZEN)]                              frozen
  -> autoregressive text generation (kazakh-bpe-50k tokenizer)
```

### Pretrained Decoder (frozen)

Loaded from HuggingFace via `transformers.LlamaForCausalLM`:

| Param | Value |
|---|---|
| HF repo | `saken-tukenov/sozkz-core-llama-150m-kk-base-v1` |
| hidden_size | 768 |
| num_hidden_layers | 16 |
| num_attention_heads | 12 |
| num_key_value_heads | 12 |
| intermediate_size | 2048 |
| vocab_size | 50257 |
| max_position_embeddings | 1024 |
| tie_word_embeddings | true |
| Tokenizer | `saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1` |

All decoder parameters are **frozen** (no gradients).

### A/B Test: Two Encoder Configs

| | **Config S (Small)** | **Config M (Medium)** |
|---|---|---|
| Conv layers | 2x Conv1d (stride 2 each) | 3x Conv1d (stride 2 each) |
| Total downsampling | 4x | 8x |
| d_model | 256 | 384 |
| n_heads | 4 | 6 |
| n_layers | 6 | 8 |
| FFN dim | 1024 (4x) | 1536 (4x) |
| Positional encoding | RoPE | RoPE |
| Encoder params | ~10M | ~25M |
| CTC head (training only) | Linear(256, 50257) | Linear(384, 50257) |
| CTC head params | ~13M | ~19M |
| Projector | Linear(256, 768) + RMSNorm | Linear(384, 768) + RMSNorm |
| Projector params | ~0.2M | ~0.3M |
| **Total trainable** | **~23M** | **~44M** |
| 10s audio -> tokens | ~250 | ~125 |
| 30s audio -> tokens | ~750 | ~375 |

**Research-informed changes from OmniAudio v1:**
- RoPE instead of sinusoidal (variable-length support, Moonshine uses this)
- More encoder layers (6/8 vs 4) — depth > width at small scale
- CTC auxiliary loss on encoder output (stabilizes training, acts as regularizer — proven by ESPnet/OWSM)
- RMSNorm instead of LayerNorm in projector (consistency with Llama)
- Label smoothing 0.1 (prevents overconfident predictions on small datasets)
- 4x downsampling preferred for Kazakh (agglutinative morphology needs higher resolution)

**Note on sequence length:** Llama 150M has `max_position_embeddings=1024`. With Config S, 30s audio produces ~937 audio tokens + text tokens — this may exceed 1024. Two mitigations:
1. Cap `max_audio_len` at 15s for Config S (-> ~468 tokens, safe)
2. Config M's 8x downsampling naturally fits 30s (~468 tokens)

**Decision:** Cap audio at **15 seconds** for both configs. This covers most Common Voice utterances (median ~5s). Config S: ~234 tokens, Config M: ~117 tokens. Plenty of room for text.

## Training Strategy

### Stage 1: CTC Pre-training (encoder only)

Train encoder + CTC head to learn good acoustic representations before connecting to LLM.

| Param | Value |
|---|---|
| Trainable | Encoder + CTC head |
| Loss | CTC |
| LR | 1e-3 |
| Epochs | 20 |
| Batch size | 32 |
| Optimizer | AdamW |
| Warmup | 10% of steps |
| Weight decay | 0.01 |
| bf16 | yes |
| Save checkpoints | every 1000 steps |

Goal: encoder learns frame-level acoustic representations. Validate with CTC greedy decode WER — if WER > 80%, encoder has fundamental problems.

### Stage 2: Alignment (projector only)

Freeze: encoder + decoder. Train: projector only (~0.2-0.3M params).

| Param | Value |
|---|---|
| Trainable | Projector only |
| LR | 1e-3 |
| Epochs | 5 |
| Batch size | 32 |
| Grad accumulation | 1 |
| Optimizer | AdamW |
| Warmup | 5% of steps |
| bf16 | yes |

Goal: learn the linear mapping from CTC-pretrained encoder space to Llama embedding space.

### Stage 3: End-to-end (encoder + projector, hybrid loss)

Freeze: decoder only. Train: encoder + projector.

| Param | Value |
|---|---|
| Trainable | Encoder + Projector + CTC head |
| Loss | 0.7 * CE (decoder) + 0.3 * CTC (encoder) |
| Label smoothing | 0.1 |
| LR | 2e-5 |
| Epochs | 15 |
| Batch size | 16 |
| Grad accumulation | 2 (effective 32) |
| Optimizer | AdamW |
| Warmup | 5% of steps |
| Weight decay | 0.01 |
| bf16 | yes |
| Save checkpoints | every 500 steps |
| Eval | every 500 steps |

Goal: encoder and projector learn end-to-end to produce features that work with Llama decoder for transcription. CTC auxiliary loss keeps encoder representations grounded.

### Data Augmentation

Applied during training to compensate for limited data (100h):
- **SpecAugment:** 2 frequency masks (F=27), 2 time masks (T=100)
- **Speed perturbation:** 0.9x, 1.0x, 1.1x (3x data effectively)

## Data

- **Dataset:** `mozilla-foundation/common_voice_17_0`, language `kk`
- **Train:** train split
- **Val:** validation split
- **Test:** test split (final WER/CER evaluation)
- **Audio:** resampled to 16kHz mono
- **Text:** tokenized with `saken-tukenov/sozkz-core-gpt2-50k-kk-base-v1`
- **Max audio length:** 15 seconds
- **Max text length:** 256 tokens

## Model Implementation

### Key change from OmniAudio v1

v1 uses a custom `LlamaDecoderBlock` stack built from scratch. v2 replaces this with HuggingFace `LlamaForCausalLM.from_pretrained()`.

The new `OmniAudioV2Model`:
1. `AudioEncoder` — same architecture as v1, but configurable (S/M)
2. `AudioProjector` — Linear + LayerNorm, maps to 768-dim
3. `llm` — `LlamaForCausalLM.from_pretrained(...)`, all params frozen
4. Forward: encode audio -> project -> concat with text embeddings -> pass through frozen LLM -> compute loss on text positions only

```python
class OmniAudioV2Model(nn.Module):
    def __init__(self, encoder_config, llm_name):
        self.encoder = AudioEncoder(**encoder_config)
        self.projector = AudioProjector(encoder_config["d_model"], 768)
        self.llm = LlamaForCausalLM.from_pretrained(llm_name)
        # Freeze LLM
        for p in self.llm.parameters():
            p.requires_grad = False

    def forward(self, mel, text_ids):
        audio_embeds = self.projector(self.encoder(mel))
        text_embeds = self.llm.model.embed_tokens(text_ids)
        combined = torch.cat([audio_embeds, text_embeds], dim=1)
        # Build causal attention mask for full sequence
        # Pass combined embeddings through LLM (inputs_embeds)
        outputs = self.llm(inputs_embeds=combined, ...)
        # Loss only on text token positions
```

### Evaluation

- **WER** (Word Error Rate) via `jiwer` on Common Voice test split
- **CER** (Character Error Rate) — more relevant for agglutinative Kazakh
- Greedy decoding (argmax) for baseline, beam search (beam=5) for final eval

## Infrastructure

- **Server:** kaznu (2x A10 23GB, SSH alias `kaznu`)
- **Deployment:** via Ansible playbook
- **Training:** single GPU (A10), ~23GB VRAM
  - Llama 150M bf16: ~300MB
  - Encoder (22M bf16): ~44MB
  - Optimizer states + activations: ~2-4GB
  - Total: well within A10 23GB
- **Detached session:** screen
- **Logging:** TensorBoard + stdout log file

## File Changes

All changes within `omniaudio/` subproject:

| Action | File | Description |
|---|---|---|
| Create | `omniaudio/src/omniaudio/model_v2.py` | New model with HF LlamaForCausalLM |
| Modify | `omniaudio/src/omniaudio/data.py` | Add SpecAugment, speed perturbation, use 50k tokenizer |
| Create | `omniaudio/src/omniaudio/train_v2.py` | Training script for v2 (loads pretrained LLM) |
| Create | `omniaudio/configs/v2_base.yaml` | Base config for v2 |
| Create | `omniaudio/configs/v2_small_stage1.yaml` | Config S, stage 1 |
| Create | `omniaudio/configs/v2_small_stage2.yaml` | Config S, stage 2 |
| Create | `omniaudio/configs/v2_medium_stage1.yaml` | Config M, stage 1 |
| Create | `omniaudio/configs/v2_medium_stage2.yaml` | Config M, stage 2 |
| Create | `omniaudio/tests/test_model_v2.py` | Tests for v2 model |
| Create | `ansible/run_omniaudio_v2.yml` | Ansible playbook for training |

## Scaling Path

After proof-of-concept on Common Voice (100h):

1. **More data:** Add KSC2 (1200h) + OpenSLR-140 (554h) — modify `data.py` to support multiple datasets
2. **Bigger decoder:** Swap Llama 150M for Llama 600M — change `llm_name` in config, adjust projector to `Linear(enc_dim, 1280)`
3. **Bigger encoder:** Increase layers/dims, add Conformer-style convolution modules
4. **Unfreeze decoder:** LoRA or full fine-tune of decoder for speech-specific adaptation
5. **Streaming:** Add causal encoder variant for real-time ASR

## Success Criteria

- Model produces recognizable Kazakh text from Common Voice audio (not garbage)
- WER under 80% on Common Voice kk test (baseline — better than random)
- CER under 50% on Common Voice kk test
- A/B comparison shows clear winner between Config S and Config M
- Training completes in under 24 hours per config on kaznu A10
