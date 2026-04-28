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
  - name: sozkz-core-omniaudio-50m-kk-asr-v1
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: google/fleurs (kk_kz, test)
          type: google/fleurs
        metrics:
          - type: wer
            value: 101.53
            name: WER
          - type: cer
            value: 128.02
            name: CER
---

# SozKZ OmniAudio v2 50M — Kazakh ASR

Compact Kazakh automatic speech recognition model built with a custom audio encoder plus a frozen SozKZ 50M Llama decoder.

## Model Details

| Parameter | Value |
|-----------|-------|
| Model repo | `stukenov/sozkz-core-omniaudio-50m-kk-asr-v1` |
| Decoder | `stukenov/sozkz-core-llama-50m-kk-base-v2` |
| Training stage | OmniAudio v2 E2E |
| Input | 16 kHz mono audio |
| Audio features | 80-bin log-mel spectrogram |
| Encoder | 256d, 4 heads, 6 layers, 2 conv layers |
| Projector | 256 -> 512 |
| Decoder hidden size | 512 |
| Tokenizer | `kazakh-gpt2-50k` |
| Max audio length | 15 seconds |

## Evaluation

Evaluated on `google/fleurs`, config `kk_kz`, `test` split.

| Metric | Value |
|--------|-------|
| WER | 101.53% |
| CER | 128.02% |
| Samples | 853 |

The model performs poorly on `FLEURS` despite being functional. Typical failures are short degenerate fragments and repeated token bursts rather than usable transcriptions.

## Training Setup

| Parameter | Value |
|-----------|-------|
| Stage | E2E |
| Dataset used for training | `stukenov/sozkz-asr-mels-kk-v1` |
| CTC weight | 0.3 |
| Learning rate | 3e-4 |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Precision | bf16 |

## Notes

- This README reports `FLEURS` test metrics for the `checkpoint-best` model.
- FLEURS loading was evaluated through a direct Hub-file fallback because newer `datasets` versions no longer support the original script-based loader.
- The model was trained on `sozkz_mels`, so the `FLEURS` domain shift appears to be severe.

## Example Predictions

| Reference | Prediction |
|-----------|------------|
| виртуалды топтар кәдімгі топтар сияқты бірдей жоғары сапа стандарттарын ұстанады дегенмен мұнда кішігірім айырмашылықтар бар | айтқанымменменменменді. |
| матаның тым ыстық болуына жол бермеңіз бұл қысқаруға немесе күюіне себеп болуы мүмкін | iмiмiмiмiмiмiмiмiмiмiмiммеiн дебемекемдер. |
| жиырмасыншы ғасырдағы зерттеу генетикалық вариацияның екі пулы бар екенін көрсетті жасырын және көрінетін | айтқанымменменменменді. |
