# OmniAudio v2 — Path to WER ≤10% Without External Encoders

**Created:** 2026-04-10
**Context:** Follow-up to the `morphbpe_fleursnorm_long15` run. 150M hit WER 43.07% on FLEURS kk_kz test (200 samples). 300M, 50M and 600M underperformed. Target: **WER ≤10% on FLEURS**, keeping the "from scratch audio encoder" philosophy — no Whisper, no MMS, no HuBERT.

## 1. Current state (baseline)

| Model | Checkpoint | WER | CER | Notes |
|---|---|---|---|---|
| **150M** `morphbpe_fleursnorm_long15` | ckpt-284000 | **43.07%** | **30.18%** | current best, still training at time of writing |
| 150M scratch full E2E (earlier) | ckpt-220000 | 45.33% | 31.70% | no pretrained LLM decoder |
| 300M `morphbpe_fleursnorm_long15` | ckpt-92000 | 49.69% | 35.02% | `grad_accum=4` → only 1/3 of gradient updates vs 150M |
| 300M `e2e_fast` (previous) | ckpt-best | 109.41% | 99.77% | repetition collapse |
| 50M `morphbpe_fleursnorm_long15` | ckpt-best (full FLEURS) | 98.95% | 81.25% | too small |
| **600M** `morphbpe_fleursnorm_long15_multigpu` | ckpt-40000 | **98.18%** | **78.15%** | only ~0.5 effective epochs seen, severely undertrained — run abandoned |

**Architecture / recipe (150M):**

- Custom encoder: `384d, 6 heads, 10 layers, 2 conv layers`
- Frozen Llama 150M decoder (`stukenov/sozkz-core-llama-150m-kk-base-v1`)
- Tokenizer `stukenov/sozkz-morphbpe-100k-kk-v1` (100k vocab)
- Training data: `stukenov/sozkz-asr-mels-kk-v1` (~1M samples, ~2100 h)
- `batch 16 / accum 1 / workers 6 / bf16`
- Loss: `0.7 * CE + 0.3 * CTC`, label smoothing 0.1
- `learning_rate 3e-4`, cosine, warmup 5%, 5 epochs
- Decode: `repetition_penalty 1.15`, `no_repeat_ngram_size 3`
- Text normalization: lowercase, strip punctuation, collapse whitespace

## 2. Why 43% is the ceiling for the current setup

1. **Audio encoder is trained from scratch with only 2100 h of labeled data.** Whisper saw 680 000 h. MMS saw 55 000 h. Our encoder has ~1/300 of the "speech experience" before it starts seeing transcripts.
2. **Frozen Llama decoder.** Text prior is strong, but the model never adapts the decoder to audio features, so audio grounding stays weak.
3. **Data volume is small.** 2100 h is ~3 % of what a SOTA Kazakh ASR model typically trains on.
4. **No language-model fusion at decoding.** Pure greedy + repetition hacks, no external n-gram LM, no neural rescoring.
5. **Train/eval normalization mismatch.** FLEURS references use digits (`70 км`), model outputs words (`жетпіс километр`), every number is penalised by WER.

SOTA on FLEURS kk_kz for reference:

- Whisper-large-v3 (off-the-shelf): ~18–25 % WER
- MMS-1B (Meta, off-the-shelf): ~20–30 % WER
- Kazakh-specific fine-tuned models: **8–15 % WER** (best published)

**Target ≤10 % is realistic only if we (a) close the encoder gap, (b) add LM fusion, (c) add at least ~3× more labeled speech, and (d) clean up the decoding pipeline.**

## 3. Non-encoder-related cheap wins (Phase 0)

Applied to the current model as-is. Zero or near-zero cost. Run after the current 150M training finishes.

| Fix | Expected ΔWER | Effort | $ | Risk |
|---|---|---|---|---|
| **Text norm audit** — fix digit↔word normalization on both `ref` and `hyp` (`70` ↔ `жетпіс`) | **−3 … −6 pp** | 1 h | 0 | none |
| **Beam search** `num_beams=8`, `length_penalty=1.0` | −1 … −2 pp | 30 min | 0 | none |
| **CTC + CE joint decoding / hybrid rescoring** — already have a CTC head from training, currently unused at inference | −1 … −3 pp | 2 h | 0 | none |
| **Polishing pass** at `lr 3e-6`, 1 epoch, `ctc_weight 0.0`, init from best ckpt | −1 … −3 pp | 4 h | ~$2.60 | low |

**Phase 0 total: ~$3, ~8 h, target WER ~33–35 %.**

These are sanity checks — they tell us how much of the 43 % is "real encoder gap" vs "broken pipeline".

## 4. Improving our own encoder (the honest path, no external models)

Key insight: **SSL pretraining of your own encoder on unlabeled audio is not the same as using a third-party encoder.** It is what Whisper/HuBERT/wav2vec2 did — they pretrained their own encoder on large unlabeled corpora. We can do the same at Kazakh scale. It is strictly more work, but it keeps the model 100 % ours.

### 4.1 Level 1: architecture tweaks (cheap, −3 … −8 pp)

- **Conformer blocks instead of pure transformer**
  - Add a Convolution module: `pointwise → GLU → depthwise conv → swish → pointwise`
  - Captures local phonetic patterns much better than self-attention alone
  - Standard ASR architecture since 2020 (Google, NeMo, ESPnet)
  - Params +5–10 %, compute +15 %
  - **ΔWER −3 … −6 pp** on the same dataset
- **More conv front-end / deeper downsampling**
  - Currently 2 conv layers (downsample 4×)
  - Try 4 conv layers or strided conv → downsample 8×
  - Fewer frames into attention → longer effective receptive field and lower memory
  - **ΔWER −1 … −2 pp**
- **Squeezeformer or Branchformer blocks**
  - Squeezeformer: Conformer + squeeze-expand + downsample-upsample → faster and slightly better
  - Branchformer: parallel attention and conv branches → better gradient flow
  - **ΔWER −1 … −3 pp** on top of plain Conformer
- **Intermediate CTC losses (iCTC)**
  - CTC head not only on the final layer but also on intermediate layers (e.g. after layers 4, 8, 12)
  - Total loss: `CE + Σ αᵢ · CTCᵢ`
  - Every layer gets explicit supervision → faster convergence, better representations
  - Standard in NeMo/ESPnet
  - **ΔWER −1 … −3 pp**, zero new parameters
- **Scale the encoder**
  - Current 384d / 10L / ~20M params is too small for 2100 h
  - A reasonable scale is 50–150M params of encoder for ~2000 h of data
  - Target: 512d / 14L / ~50M or 768d / 16L / ~120M
  - **ΔWER −2 … −5 pp** if data is enough to avoid overfit
- **More mel bins**
  - Currently `n_mels=80`
  - Try `n_mels=128` for higher spectral resolution
  - Phonetically rich languages like Kazakh benefit slightly
  - **ΔWER −0.5 … −1 pp**

### 4.2 Level 2: self-supervised pretraining (biggest lever, −10 … −20 pp)

Pretrain our own encoder on unlabeled Kazakh audio before E2E fine-tuning. Three main SSL objectives that work on audio:

- **BEST-RQ (recommended)**
  - Random-projection quantizer: a frozen random matrix projects mel slices into a fixed discrete codebook
  - Mask 50 % of mel frames, encoder predicts the code of every masked frame
  - No contrastive learning (no negative sampling)
  - No k-means (no proxy labels from a previous model version)
  - Used by Google in USM and Gemini audio; proven at scale
  - Minimal reference implementation: <https://github.com/sooftware/BEST-RQ>
- **wav2vec 2.0 contrastive pretraining**
  - Mask spans → encoder → contrastive loss between the masked slot and negatives
  - Works, but temperature and negative sampling need careful tuning
  - Same expected effect as BEST-RQ with more hyperparameter pain
- **data2vec v2**
  - Teacher / student: encoder predicts its own EMA-teacher representations at masked positions
  - Regression loss, no discrete codes
  - Works well but less common in ASR tooling

**Recommended path: BEST-RQ.** Simplest, fewest hyperparameters, proven at scale.

**Pipeline:**

1. Collect 5 000–20 000 h of unlabeled Kazakh audio (YouTube, radio, podcasts — just download, no transcription)
2. VAD + segment into 10–15 s chunks, convert to mel
3. Pretrain the encoder on the BEST-RQ objective, ~200 K–500 K steps (~3–7 days on 3× RTX 5090)
4. Fine-tune on `sozkz-asr-mels-kk-v1` with CTC + CE, using the pretrained encoder weights as init
5. Apply the decoder (frozen Llama or unfrozen) the same way as v2

**Expected ΔWER −10 … −20 pp.** This is our equivalent of Whisper's pretraining.

Cost: ~$100–200 compute + ~1–2 days of scraping + ~1 day of preprocessing.

### 4.3 Level 3: augmentation and data (−2 … −5 pp)

- **Heavier SpecAugment**
  - Time mask: 10 × 40 frames (check current config)
  - Freq mask: 2 × 27 bins
  - Consider adaptive SpecAugment (scales with sequence length)
- **Speed perturbation** 0.9× / 1.0× / 1.1×
- **Noise injection** (MUSAN or in-the-wild noise)
- **Room impulse response / reverb** augmentation
- **Volume and channel perturbation**
- **Multi-task encoder objectives** during pretrain:
  - Frame-level phoneme prediction (needs a kk phonemizer)
  - Speaker classification head (improves normalization)
  - VAD head (speech/silence)

Each auxiliary task gives the encoder an extra gradient signal.

### 4.4 Level 4: joint CTC + RNN-T + AED (harder, −2 … −4 pp)

- Keep the current attention-based decoder (AED via Llama) and the CTC head, add an **RNN-Transducer (RNNT) head** on top of the same encoder
- Three decoders share the same encoder: CTC, RNNT, AED
- Inference: MBR decoding or rescoring across all three heads
- Standard NeMo "Hybrid CTC-Transducer" recipe

### 4.5 Level 5: self-distillation

- Train encoder + decoder as usual
- Use the trained model as a teacher for a second iteration
- Self-distillation loop: student + teacher predictions + raw labels
- **ΔWER −1 … −2 pp** typically

## 5. Phased plan to reach WER ≤10 % without external encoders

| Phase | Scope | Expected WER | $ | Time |
|---|---|---|---|---|
| **0** | Polish the current 150M: beam search + CTC rescore + text norm + low-LR polishing | 43 % → **~33–35 %** | ~$3 | 8 h |
| **1** | `OmniAudio v3` with Conformer blocks + iCTC + ~50M encoder + 8 epochs, same labeled data | 35 % → **~27–30 %** | ~$20 | 1 day |
| **2** | **BEST-RQ SSL pretrain** on 5 000 h unlabeled kk → fine-tune on 2 100 h labeled | 27 % → **~18–22 %** | ~$150 | 5–7 days |
| **3** | Scale encoder to 100–150M, expand SSL corpus to 20 000 h, re-pretrain, re-finetune | 20 % → **~13–17 %** | ~$300 | 2 weeks |
| **4** | Joint CTC + RNNT + AED head, FLEURS-train fine-tune, KenLM n-gram shallow fusion, neural LM rescoring | 15 % → **~9–12 %** ✅ | ~$50 | 4 days |
| **5** *(optional)* | 2–3 seed ensemble, self-distillation iteration, broader augmentation | 10 % → **~7–9 %** | ~$100 | 1 week |

**Total to reach WER ≤10 % without external encoders: ~$620, ~5 weeks of active work.**

For reference the "use Whisper encoder" path is ~$315 and ~3 weeks, so the self-pretrained approach costs roughly $300 more and two extra weeks of compute + data collection. That delta is entirely BEST-RQ pretraining.

## 6. Critical blockers and decisions

1. **Unlabeled Kazakh audio corpus** (5–20 kh). This is the single biggest non-compute blocker. Nothing SSL-related can start without it. Sources:
   - Kazakh YouTube (news, podcasts, education): 5–20 k h accessible
   - Kazakh radio online streams (record): 1–5 k h
   - Kazakh TV archives: 2–10 k h
   - Pipeline: youtube-dl / yt-dlp → VAD → 10–15 s segments → mel
2. **Compute budget.** Phases 2 and 3 dominate. Need ~$450 for two SSL training rounds.
3. **Is FLEURS the optimization target?** FLEURS-train fine-tuning (Phase 4) is legitimate for production but not for "independent benchmark" claims. We need to decide whether we report with or without FLEURS-train.
4. **Decoder strategy.** Frozen Llama 150M may cap the last 1–2 pp. Consider unfreezing the last 2–4 Llama layers at Phase 4 or switching to `sozkz-core-llama-300m-kk-base-v1`. The 300M sibling experiment showed that bigger frozen decoders need way more gradient updates, so **if we move to a larger decoder we must drop `grad_accum`**.

## 7. Phase 0 — concrete follow-ups we can do immediately

These do not wait for anything:

- **`evaluate_v2.py` patches**
  - Add `--num-beams`, `--length-penalty` flags (pass through to `model.generate()`)
  - Add CTC + CE hybrid decoding: for each beam candidate, combine the AED log-prob with the CTC head forward score at α = 0.3, rerank
  - Add digit ↔ word normalization helper applied symmetrically to `ref` and `hyp`
- **New config** `v2_llm150m_e2e_polish.yaml`
  - Inherits current 150M config
  - `learning_rate: 3e-6`, `num_train_epochs: 1`, `ctc_weight: 0.0`, `label_smoothing: 0.0`
  - `init_from: outputs/.../checkpoint-best`
- **Eval script** that runs the four Phase 0 deltas independently and reports each contribution

## 8. Phase 1 — concrete follow-ups once current 150M finishes

- **`model_v2.py`** — add `ConformerBlock` class:
  - LayerNorm → MultiHeadAttention → LayerNorm → ConvModule → LayerNorm → FFN → residuals
  - ConvModule: pointwise conv → GLU → depthwise conv (kernel 31) → BatchNorm → Swish → pointwise conv → dropout
- **`model_v2.py`** — add `IntermediateCTCHead(layer_idx, weight)` list, registered at construction time
- **`train_v2.py`** — aggregate loss `CE + Σ αᵢ · CTCᵢ`
- **New config** `v3_llm150m_conformer_icct.yaml`
  - `audio_d_model: 512`, `audio_n_heads: 8`, `audio_n_layers: 14`, `audio_n_conv: 4`
  - `ctc_weight: 0.3`, `intermediate_ctc: [4, 8, 12]`, `intermediate_ctc_weight: 0.1`
  - `num_train_epochs: 8`, `batch: 12`, `accum: 1`

Expected budget for Phase 1: ~24 hours on a single RTX 5090 pod (~$15).

## 9. Phase 2 — BEST-RQ pretrain scaffolding

- New module `omniaudio/src/omniaudio/ssl_bestrq.py`:
  - Random projection quantizer (`W ∈ ℝ^{mel_dim × codebook_dim}`, fixed at init)
  - Codebook (`K = 8192` codes) with `L2-normalized` rows
  - Masking: contiguous spans of `10 % of T`, 5 % of mask tokens
  - Loss: cross-entropy over masked positions vs nearest codebook entry of the projected mel
- New script `omniaudio/scripts/ssl_pretrain.py`:
  - Streams mels from a HF dataset of unlabeled kk audio
  - Optimizer `AdamW`, `lr 5e-4`, warmup 5 %, cosine
  - 200 k steps, `batch 32`, grad accum 1
- Data prep pipeline `omniaudio/scripts/collect_unlabeled_kk.py`:
  - yt-dlp + Silero VAD + 15-second segmentation + mel computation
  - Target: 5 000 h as first goal, 20 000 h as stretch
- Finetune config `v3_llm150m_from_bestrq.yaml`:
  - `init_from: outputs/ssl_bestrq_v1/checkpoint-best/encoder.pt`
  - Everything else mirrors the best v2 recipe

## 10. Recommended execution order

1. **Now:** finish the running 150M training and apply Phase 0 polish. Record every delta.
2. **Next:** implement Conformer + iCTC as Phase 1, launch a 24 h `v3_llm150m_conformer` run. Verify the architectural lever is real.
3. **In parallel:** start collecting unlabeled Kazakh audio. The slowest part of the whole plan is not compute, it is scraping and filtering — it is pure wall-clock latency.
4. **When ≥5 000 h of unlabeled audio is ready:** run Phase 2 BEST-RQ pretrain. This is the real investment.
5. **After BEST-RQ:** fine-tune on labeled data, add LM fusion, land near 10 % WER.
6. **Optional Phase 5:** ensemble and self-distillation for anything below 10 %.

## 11. What this document is not

- Not a promise. WER projections are based on published scaling trends and could miss the target by 3–5 pp.
- Not a replacement for an ablation. Each delta should be measured independently on the same 200-sample FLEURS subset before committing to the next phase.
- Not an endorsement of using FLEURS train data for public benchmarks. Phase 4 FLEURS-train fine-tuning is for production-oriented evaluation; "honest academic" numbers must exclude it.

## 12. Related artifacts

| Artifact | Path |
|---|---|
| OmniAudio v2 whitepaper | `docs/omniaudio_v2_whitepaper.md` |
| OmniAudio v2 architecture | `docs/omniaudio_v2_architecture.md` |
| 150M model card | `docs/cards/README_sozkz-core-omniaudio-150m-kk-asr-v1.md` |
| 300M model card | `docs/cards/README_sozkz-core-omniaudio-300m-kk-asr-v1.md` |
| Training script | `omniaudio/src/omniaudio/train_v2.py` |
| Eval script | `omniaudio/src/omniaudio/evaluate_v2.py` |
| Model module | `omniaudio/src/omniaudio/model_v2.py` |
| CloudRift ops | `cloudrift.md` |
