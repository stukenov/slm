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
  - name: sozkz-core-omniaudio-300m-kk-asr-v1
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
            value: 49.69
            name: WER
          - type: cer
            value: 35.02
            name: CER
---

# SozKZ OmniAudio 300M Kazakh ASR

End-to-end Kazakh ASR built with the OmniAudio v2 recipe: a large custom audio encoder feeding a frozen SozKZ Llama 300M text decoder, trained with morph-BPE 100k tokens and FLEURS-style text normalization over 15-second audio windows.

The canonical checkpoint for this model card is `checkpoint-92000`, which is the latest evaluated checkpoint from the `morphbpe100k_fleursnorm_long15` run. Earlier `checkpoint-100000` through `checkpoint-150000` in the repo are leftover artifacts from the deprecated `e2e_fast` run and should be ignored.

## Model Details

| Parameter | Value |
|-----------|-------|
| Model repo | `stukenov/sozkz-core-omniaudio-300m-kk-asr-v1` |
| Architecture | OmniAudio v2 encoder + frozen Llama 300M decoder |
| Decoder base | `stukenov/sozkz-core-llama-300m-kk-base-v1` |
| Encoder | `640d`, `10 heads`, `14 layers`, `2 conv layers` |
| Projector | `640 -> 1024` |
| Decoder hidden size | `1024` |
| Audio features | 80-bin log-mel spectrogram |
| Input audio | 16 kHz mono |
| Tokenizer | `stukenov/sozkz-morphbpe-100k-kk-v1` |
| Vocab size | `100000` |
| Max audio length | 15 seconds |
| Canonical checkpoint | `checkpoint-92000` |

## Evaluation

Evaluated on `google/fleurs`, config `kk_kz`, `test` split, `200` samples, using `checkpoint-92000`. Text was lowercased, stripped of punctuation, and whitespace-collapsed to match the training target distribution.

| Metric | Value |
|--------|-------|
| WER | 49.69% |
| CER | 35.02% |
| Samples | 200 |

### Comparison with prior 300M runs and sibling models

| Model / Run | Checkpoint | WER | CER | Notes |
|---|---|---|---|---|
| **This card (morphbpe_fleursnorm_long15)** | ckpt-92000 | **49.69%** | **35.02%** | current best 300M |
| 300M `e2e_fast` (previous recipe) | ckpt-best | 109.41% | 99.77% | repetition collapse |
| 300M `e2e_fast` (previous recipe) | ckpt-150000 | 101.97% | 93.84% | |
| 150M `morphbpe_fleursnorm_long15` (sister) | ckpt-284000 | 43.07% | 30.18% | outperforms this 300M |
| 150M scratch full E2E (earlier) | ckpt-220000 | 45.33% | 31.70% | |

The new recipe (`morph-BPE 100k` tokenizer, FLEURS-normalized targets, 15 s max audio, longer training) moved this model from catastrophic (WER > 100 %) to the first usable 300M ASR in this project.

### Why does the 150M sibling beat this 300M?

Not a dataset issue — both saw the same `sozkz-asr-mels-kk-v1` (~2100 hours). The primary cause is fewer gradient updates: this 300M run used `per_device_train_batch_size: 12` with `gradient_accumulation_steps: 4` (effective batch 48) due to single-GPU VRAM constraints with a frozen Llama 300M decoder. The sibling 150M ran `batch 16 / accum 1` (effective batch 16). Across 5 epochs, the 300M received roughly **one third of the gradient updates** that the 150M did, so it is effectively undertrained relative to its capacity. Training was also stopped at `step 92800 / ~107000` (~87% of the planned 5-epoch schedule) after the 150M already demonstrated a better WER with the same recipe.

## Real Inference Examples

Direct `REF/HYP` pairs from `FLEURS kk_kz test` with `checkpoint-92000`.

| Reference | Prediction |
|-----------|------------|
| апат биік таулы жерлерде орын алған және дұшпандық мағынадағы өрттің салдарынан болған деп болжануда | апат биік таулы жерлерде орын алған жәнеинарлықдылық ғылымдағы өрттің салдарынан болған деп зіында |
| көп неміс пісірілген өнімдерінде сонымен бірге бадам орман жаңғақтары және басқа ағаш жаңғақтары болады танымал торттар жиі қою кофенің саптыаяғымен өте жақсы желінеді | көп неміс түсірілген өнімдерінде сонымен біргесарыбаев жаңтары және басқа ағаш жаңтары болады танымал хабарлан ә ме қою сипаттамен өте жақсыжоғарыдағы |
| цунами туралы ескерту берілмеді және джакартаның геофизика агенттігінің деректеріне сай цунами туралы ескерту берілмейді өйткені жер сілкінісі 6 5 магнитудасы талабына сай болмады | соналған туралы ескерту берілмеді жәнероарыпыңды физика агенттігінің деректеріне сайэк туралы ескерту ұсынмейді өйткені жерпарде алты бүтін оннан |
| кейбір адамдар оныкі дұрыс деп ойлады бірақ көп адам қарама қарсы нәрсеге сенді күн жүйесі соның ішінде күн және тіпті басқа жұлдыздар жерді айналды деп ойлады | кейбір адамдар он екі дұрыс деп ойлады бірақ көп адам қарама нәрсе тиімділіг жандандыр тоқта күн жүйесі соның ішінде күн |
| бар болғаны кеме экскурсиялары арқылы жағаға барсаңыз сізге бөлек виза қажет болмайды 2009 жылдан | бар болғаны кемеастана арқылы жағаға бараңыз сізге бөлек риза қажет жен екі мың тоғызыншы жылдан |

Typical failure modes: dropped or mangled rare words, drift toward phonetic near-neighbors, and premature truncation on long (>10 s) utterances where attention over the audio encoder weakens.

## Training Summary

| Parameter | Value |
|-----------|-------|
| Stage | OmniAudio v2 E2E |
| Init | `stukenov/sozkz-core-omniaudio-300m-kk-ctc-v1` `checkpoint-best` |
| Training dataset | `stukenov/sozkz-asr-mels-kk-v1` (~1M samples, ~2100 h) |
| Target normalization | lowercase, strip punctuation, collapse whitespace (FLEURS-style) |
| Max audio length | 15 s |
| Batch size | 12 per device |
| Gradient accumulation | 4 (effective batch 48) |
| Num workers | 6 |
| Learning rate (peak) | 2e-4 |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Loss | `0.7 * CE + 0.3 * CTC` |
| Label smoothing | 0.1 |
| Precision | bf16 |
| Epochs (planned) | 5 |
| Step at final checkpoint | 92000 |
| Training progress at stop | ~87% of planned schedule |
| Hardware | 1 × RTX 5090 32 GB (CloudRift) |
| Decode | `repetition_penalty 1.15`, `no_repeat_ngram_size 3` |

## Limitations

- 200-sample estimate, not full-test FLEURS. Expect ±2-4 pp variance vs. full test.
- Undertrained relative to its capacity — see "Why does the 150M sibling beat this 300M?" above.
- Weaker robustness on out-of-domain speech and long (> 10 s) utterances.
- Number handling is brittle; digits in the reference are often spelled out and penalized by WER.
- Not production-ready. The 150M sibling is the recommended OmniAudio v2 ASR model at this scale tier.

## Related Artifacts

| Artifact | Link |
|----------|------|
| Decoder base model | `stukenov/sozkz-core-llama-300m-kk-base-v1` |
| CTC pretrain stage | `stukenov/sozkz-core-omniaudio-300m-kk-ctc-v1` |
| Training mel dataset | `stukenov/sozkz-asr-mels-kk-v1` |
| Sister model (better) | `stukenov/sozkz-core-omniaudio-150m-kk-asr-v1` |
| Tokenizer | `stukenov/sozkz-morphbpe-100k-kk-v1` |
