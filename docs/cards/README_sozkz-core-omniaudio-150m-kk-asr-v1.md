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
library_name: transformers
pipeline_tag: automatic-speech-recognition
metrics:
  - wer
  - cer
model-index:
  - name: sozkz-core-omniaudio-150m-kk-asr-v1
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: google/fleurs (kk_kz, sample-200)
          type: google/fleurs
          config: kk_kz
          split: test
        metrics:
          - type: wer
            value: 46.42
            name: WER
          - type: cer
            value: 32.16
            name: CER
---

# SozKZ OmniAudio 150M Kazakh ASR

Kazakh ASR model built with OmniAudio v2 scratch training on SozKZ mels. This variant uses a from-scratch encoder-decoder stack with about 147M parameters and no pretrained text LLM.

## Model Details

| Parameter | Value |
|-----------|-------|
| Model repo | `stukenov/sozkz-core-omniaudio-150m-kk-asr-v1` |
| Architecture | OmniAudio v2 scratch encoder-decoder |
| Total params | `147.32M` |
| Encoder | `384d`, `6 heads`, `10 layers`, `2 conv layers` |
| Decoder | `768d`, `12 heads`, `10 layers` |
| Audio features | 80-bin log-mel spectrogram |
| Input audio | 16 kHz mono |
| Tokenizer | `kazakh-gpt2-50k` |
| Max audio length | 10 seconds |

## Evaluation

Quick evaluation on `google/fleurs`, config `kk_kz`, `test` split, `200` samples, using a strong intermediate `checkpoint-best`.

| Metric | Value |
|--------|-------|
| WER | 46.42% |
| CER | 32.16% |
| Samples | 200 |

This is a partial sample-based estimate, not the final full-test benchmark.

## Real Inference Examples

The following examples are direct `REF/HYP` pairs from actual evaluation runs on `FLEURS`.

| Reference | Prediction |
|-----------|------------|
| ақш-тың гимнастика федерациясы ақш олимпиада комитетінің хатын қолдайды және біздің барлық спортсмендер үшін қоршаған ортаның қауіпсіздігін арттырудың абсолютті қажеттілігін мойындайды | ақш тың гимнастика федерациясы ақш олимпиада комитетінің хатын қолдайды және біздің барлық спортсмендері үшін қоршаған ортаны қорғау |
| барлық оңтүстік африка ұлттық саябақтарындағыдай саябаққа кіруде экологиялық салым және кіру ақысы төленеді | барлық оңтүстік африка ұлттық саябақтарындадай саябаққа кіруіне экологиялық саяси қызмет көрсету үшін де |
| бар болғаны кеме экскурсиялары арқылы жағаға барсаңыз сізге бөлек виза қажет болмайды 2009 жылдан | бар болғаны кеме экскурсиялары арқылы жағаға барсаңыз сізге бөлек виза қажет болмайды екі мың тоғызыншы жылдан |
| көп неміс пісірілген өнімдерінде сонымен бірге бадам орман жаңғақтары және басқа ағаш жаңғақтары болады танымал торттар жиі қою кофенің саптыаяғымен өте жақсы желінеді | көп неміс пісірілген өнімдерді сонымен бірге баланың орман жаңғақтары және басқа да шаңғақтары болады танымал тортар жүйе |
| бұл есеп жетекшінің иракқа қатысты ағымдағы саясатының әрбір аспектісін қатты сынайды және бағытты дереу өзгертуге шақырады | бұл есеп жетекшісінің иракқа қатысты ағымдағы саясатының әрбір аспектісін қатты сынайды және бағытта |

## Interpretation

The model already shows real audio grounding and often keeps the structure of the source sentence. Typical remaining issues are:

- truncation near the end of an utterance
- lexical substitutions
- occasional grammatical drift
- weaker robustness on out-of-domain speech

## Training Summary

| Parameter | Value |
|-----------|-------|
| Training mode | scratch encoder-decoder |
| Training dataset | `stukenov/sozkz-asr-mels-kk-v1` |
| Loss | `0.7 * CE + 0.3 * CTC` |
| Best practical config | `batch 24`, `grad_accum 1`, `workers 6`, `torch_compile false` |
| Precision | bf16 |

## Limitations

- Current metric in this card is a `200`-sample estimate, not full-test `FLEURS`.
- Quality is not yet production-level.
- Out-of-domain robustness is still limited.

## Related Artifacts

| Artifact | Link |
|----------|------|
| Training mel dataset | `stukenov/sozkz-asr-mels-kk-v1` |
| OmniAudio v2 whitepaper | local project whitepaper |
