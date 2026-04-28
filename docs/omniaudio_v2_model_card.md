---
license: mit
language:
  - kk
tags:
  - speech-recognition
  - asr
  - kazakh
  - audio
  - omniaudio
pipeline_tag: automatic-speech-recognition
metrics:
  - wer
  - cer
model-index:
  - name: sozkz-core-omniaudio-70m-kk-asr-v1
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: kzcalm-tts-kk-v1 (test split)
          type: stukenov/kzcalm-tts-kk-v1
        metrics:
          - type: wer
            value: 21.28
            name: WER
          - type: cer
            value: 12.80
            name: CER
---

# SozKZ OmniAudio v2 — Kazakh ASR (70M)

A native Kazakh automatic speech recognition model trained from scratch. No pretrained components — both encoder and decoder are trained entirely on Kazakh speech data.

## Model Details

| Parameter | Value |
|-----------|-------|
| **Architecture** | Custom encoder-decoder with cross-attention |
| **Total params** | 69.58M (266 MB) |
| **Encoder** | 5.1M params — 256d, 4 heads, 6 layers, 2 conv (4x downsample), bidirectional |
| **Decoder** | 51.4M params — 512d, 8 heads, 8 layers, causal, tied embeddings |
| **Components** | RoPE, RMSNorm, SwiGLU (Llama-style) |
| **Tokenizer** | kazakh-gpt2-50k (50,257 vocab, BPE) |
| **Training data** | kzcalm-tts-kk-v1 (232K samples, 439 hours) |
| **Input** | 16kHz mono audio, 80-bin log-mel spectrogram |
| **Output** | Kazakh text |

## Results

| Metric | Value |
|--------|-------|
| **WER** | 21.28% |
| **CER** | 12.80% |
| **Val loss** | 2.3547 |

Assessed on 50 test samples from kzcalm-tts-kk-v1.

### Example Outputs

| Reference | Prediction |
|-----------|------------|
| Тоқ етер түйіні – адам өзін–өзі зерттеуі керек! | Тоқ етер түйіні – адам өзін–өзі зерттеуі керек! |
| Тамыр емес, бейтаныс қонжықтың жүніне қолы тигенін ол енді ғана сезгендей болды. | Тамыр емес, бейтаныс қонжықтың жүніне қолы тигенін ол енді ғана сезгендей болды. |
| Ол сонда да есін жоғалтпай тұрды. | Ол сонда да есін жоғалтпай тұрды. |
| Сондай–ақ барша Қазақстан халқына шексіз алғысын жолдап, бақ–береке тіледі. | Сондай–ақ, барша Қазақстан халқына шексіз алғысын жолдап, бақ-береке тіледі. |

## Training

### 3-Stage Pipeline

1. **CTC Pretraining** (5 epochs): Encoder + CTC head only. Teaches the encoder to produce meaningful audio representations.
2. **End-to-End with CTC** (4+ epochs): Full model with hybrid loss (0.7 CE + 0.3 CTC). Trains cross-attention and decoder.
3. **Pure CE Fine-tuning** (1 epoch): CE loss only, LR=5e-6, label_smoothing=0.05. Final refinement.

### Training Details

- **Hardware:** 1x NVIDIA RTX 4090 (24 GB)
- **Augmentation:** SpecAugment (freq=27, time=100, 2 masks each) + speed perturbation (0.9x, 1.0x, 1.1x)
- **Optimizer:** AdamW, weight_decay=0.01
- **Schedule:** Linear warmup + linear decay
- **Precision:** bfloat16

### Key Findings

- **EOS token is critical:** Without EOS in training targets, the model generates correct text followed by infinite garbage. Adding EOS reduced WER from 855% to ~21%.
- **DDP hurts this model:** DistributedDataParallel consistently degraded WER (tested twice). Single GPU training works best for this model size.
- **Lower LR for fine-tuning:** 5e-6 >> 1e-5 >> 1e-4 when continuing from a good checkpoint.
- **Pure CE > hybrid CTC+CE** for final fine-tuning stage.

## Architecture

```
Audio (16kHz)
  -> Mel Spectrogram (80 bins)
  -> Conv1 (1->256, k=3, s=2) -> Conv2 (256->256, k=3, s=2)  [4x downsample]
  -> 6x Encoder Blocks (bidirectional self-attention + SwiGLU FFN)
  -> Projector (256->512)
  -> 8x Decoder Blocks (causal self-attention + cross-attention + SwiGLU FFN)
  -> LM Head (512->50257, tied with embeddings)
  -> Text tokens -> Tokenizer decode -> Kazakh text
```

## Usage

```python
import torch
import torchaudio
from omniaudio.model_v2 import OmniAudioScratchModel
from transformers import PreTrainedTokenizerFast

# Load model
model = OmniAudioScratchModel(
    encoder_config={"n_mels": 80, "d_model": 256, "n_heads": 4, "n_layers": 6, "n_conv": 2},
    decoder_config={"d_model": 512, "n_heads": 8, "n_layers": 8},
    vocab_size=50257,
)
state = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=False)
model.cuda()

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/")

# Transcribe
waveform, sr = torchaudio.load("audio.wav")
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_mels=80, n_fft=400, hop_length=160
)
mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-10)).cuda()

with torch.no_grad():
    tokens = model.generate(mel, max_len=256)
text = tokenizer.decode(tokens, skip_special_tokens=True)
print(text)
```

## Limitations

- Trained on studio-quality TTS data (kzcalm) — may perform worse on noisy real-world audio
- 70M model has limited capacity — larger models with more diverse data would improve results
- Only Kazakh language supported
- Max audio length: 10 seconds (during training)

## Citation

```bibtex
@misc{omniaudio-v2-2026,
  title={OmniAudio v2: Native Kazakh ASR from Scratch},
  author={Saken Tukenov},
  year={2026},
  url={https://huggingface.co/stukenov/sozkz-core-omniaudio-70m-kk-asr-v1}
}
```
