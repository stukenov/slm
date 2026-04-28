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
  - name: sozkz-core-omniaudio-50m-kk-asr-v1
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: google/fleurs (kk_kz, test)
          type: google/fleurs
          config: kk_kz
          split: test
        metrics:
          - type: wer
            value: 98.95
            name: WER
          - type: cer
            value: 81.25
            name: CER
---

# SozKZ OmniAudio 50M Kazakh ASR

Compact Kazakh ASR model built with the OmniAudio v2 stack: a custom audio encoder plus a frozen SozKZ 50M Llama decoder. The latest run uses a MorphBPE 100k tokenizer and FLEURS-style text normalization.

## Model Details

| Parameter | Value |
|-----------|-------|
| Model repo | `stukenov/sozkz-core-omniaudio-50m-kk-asr-v1` |
| Architecture | OmniAudio v2 encoder + frozen Llama decoder |
| Decoder base | `stukenov/sozkz-core-llama-50m-kk-base-v2` |
| Encoder | 256d, 4 heads, 6 layers, 2 conv layers |
| Projector | 256 -> 512 |
| Decoder hidden size | 512 |
| Audio features | 80-bin log-mel spectrogram |
| Input audio | 16 kHz mono |
| Tokenizer | `stukenov/sozkz-morphbpe-100k-kk-v1` |
| Max audio length | 15 seconds |

## Evaluation

Evaluated on `google/fleurs`, config `kk_kz`, `test` split, using the finished MorphBPE follow-up run `checkpoint-best`.

| Metric | Value |
|--------|-------|
| WER | 98.95% |
| CER | 81.25% |
| Samples | 853 |

This is still a weak `FLEURS` result, but it is better than the earlier frozen-decoder 50M baseline.

## Comparison

`google/fleurs`, `kk_kz`, `test`:

| Variant | Description | WER | CER |
|--------|-------------|-----|-----|
| `old-baseline-best` | original frozen-decoder 50M, `checkpoint-best` | 101.53% | 128.02% |
| `old-baseline-ckpt26000` | original frozen-decoder 50M, `checkpoint-26000` | 125.39% | 107.00% |
| `morphbpe-fleursnorm-ckpt26000` | new tokenizer + normalization, `checkpoint-26000` | 98.36% | 83.58% |
| `morphbpe-fleursnorm-best` | finished MorphBPE follow-up, `checkpoint-best` | 98.95% | 81.25% |

Takeaway:
- The `unfreeze2` recipe was a failure and is no longer the preferred direction.
- The MorphBPE 100k + lowercased/punctuation-stripped recipe is better than the old 50M baseline on `FLEURS`.
- Absolute quality is still poor, and repetitive collapse remains the main failure mode.

## Real Inference Examples

### Old Baseline 50M (`checkpoint-26000`)

| Reference | Prediction |
|-----------|------------|
| мұнда осы уақытқа дейін осында өмір сүрген көптеген ерлер мен әйелдер сондай-ақ жақындары қаза болған немесе соңғы демі қалғанша қызмет еткен еврей еврей емес тірі адамдар көп | iмнiңiмдi, бұл осы тiлшімгерлiктердiң, мен – бұл-да, ол – қазақ тiлiнде жазылған. |
| ол оңтүстік африканың негізгі көрнекі жерлерінің бірі болып табылады және оңтүстік африка ұлттық парктерінің sanparks жетекшісі болып есептеледі | бөлімінен бөлімінен бөлімінен және, ең көп бөлігі (қазақ. |
| айдың беті тау жынысы мен шаңнан тұрады айдың сыртқы қабаты қабық деп аталады | бөлімінен бөлімінен бөлімінен және, ең көп бөлігі (қазақ. |

### New MorphBPE + FLEURS-Norm 50M (`checkpoint-best`)

| Reference | Prediction |
|-----------|------------|
| бұл жұптар баласы үшін асырап алу жоспарын жасауды таңдауы мүмкін | бұлныңдыдығанғағағады |
| цунами туралы ескерту берілмеді және джакартаның геофизика агенттігінің деректеріне сай цунами туралы ескерту берілмейді өйткені жер сілкінісі 6 5 магнитудасы талабына сай болмады | сонға және және жәнедіменді және және ге мендыңды және жәнеғағаін |
| іс жүзінде тіпті адам оның бар екенін білсе де оны табу оңай емес үңгірге түскеннен кейін толығымен оқшауланасыз | өз барге мен менменас де жәнеғағаіндіпды |

## Training Summary

| Parameter | Value |
|-----------|-------|
| Stage | OmniAudio v2 E2E |
| Training dataset | `stukenov/sozkz-asr-mels-kk-v1` |
| Text normalization | lowercase + strip punctuation + collapse whitespace |
| CTC weight | 0.3 |
| Learning rate | 3e-4 |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Precision | bf16 |

## Usage

```python
import torch
from transformers import AutoTokenizer
from omniaudio.model_v2 import OmniAudioV2Model

model = OmniAudioV2Model(
    encoder_config={
        "n_mels": 80,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 6,
        "n_conv": 2,
    },
    llm_name="stukenov/sozkz-core-llama-50m-kk-base-v2",
    vocab_size=50257,
    llm_dim=512,
)
state = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=False)
model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained("./tokenizers/kazakh-gpt2-50k")
```

## Limitations

- Weak generalization to `FLEURS` test audio.
- Strong domain sensitivity relative to the training mixture.
- Output can still collapse into repetitive fragments or short pseudo-morpheme bursts.
- Intended for research and debugging, not production ASR.

## Related Artifacts

| Artifact | Link |
|----------|------|
| Decoder base model | `stukenov/sozkz-core-llama-50m-kk-base-v2` |
| Tokenizer | `stukenov/sozkz-morphbpe-100k-kk-v1` |
| Training mel dataset | `stukenov/sozkz-asr-mels-kk-v1` |
