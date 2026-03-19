# Experiment Log: exp004_scratch_kk

Custom Llama 50M trained from scratch on Kazakh text.

## Model

| Parameter | Value |
|---|---|
| Architecture | LlamaForCausalLM |
| Parameters | 50.29M |
| Layers | 8 |
| Hidden size | 576 |
| Attention heads | 8 (MHA, no GQA) |
| Intermediate size | 1536 (SwiGLU) |
| Max positions | 1024 |
| Tied embeddings | Yes |
| Tokenizer | kazakh-bpe-32k (ByteLevel BPE, 32K vocab) |

## Training Config

| Parameter | Value |
|---|---|
| Dataset | kz-transformers/multidomain-kazakh-dataset (23.6M samples) |
| Block size | 512 |
| Batch size | 16 x 2 GPUs x 2 grad_accum = 64 effective |
| Learning rate | 6e-4 (cosine schedule) |
| Warmup | 700 steps |
| Weight decay | 0.1 |
| Precision | bf16 |
| Total steps | 87,770 (1 epoch) |
| Hardware | 2x NVIDIA A10 (23GB each) |

## Training Curve (Train Loss)

| Step | Train Loss | Grad Norm | LR | Epoch |
|---|---|---|---|---|
| 50 | ~10.4 | - | ~4.3e-5 | 0.001 |
| 700 | ~4.7 | - | 6e-4 | 0.008 |
| 2000 | 4.269 | - | 5.98e-4 | 0.023 |
| 4000 | 3.899 | 0.350 | 5.97e-4 | 0.046 |
| 8000 | 3.659 | 0.353 | 5.90e-4 | 0.091 |
| 12000 | 3.545 | - | 5.73e-4 | 0.137 |
| 16000 | 3.478 | - | 5.49e-4 | 0.182 |
| 20000 | 3.432 | - | 5.19e-4 | 0.228 |
| 24000 | 3.389 | 0.365 | 4.96e-4 | 0.273 |

## Eval Loss Curve

| Step | Eval Loss | Perplexity |
|---|---|---|
| 2000 | 4.249 | ~70 |
| 4000 | 3.886 | ~49 |
| 6000 | 3.739 | ~42 |
| 8000 | 3.654 | ~39 |
| 10000 | 3.592 | ~36 |
| 12000 | 3.545 | ~35 |
| 14000 | 3.508 | ~33 |
| 16000 | 3.478 | ~33 |
| 18000 | 3.455 | ~32 |
| 20000 | 3.432 | ~31 |
| 22000 | 3.414 | ~30 |
| 24000 | 3.391 | ~30 |

Speed: ~3.2 it/s (training) + ~462s per eval pass. Each eval ~7.7 min.

## Checkpoint Evaluations

### checkpoint-2000 (step 2000, loss 4.269)

- Avg Kazakh char ratio: **0.169**
- Generation quality: model produces grammatically plausible Kazakh text, but:
  - Semantically incoherent (rambles, topic drift)
  - Repetitive patterns ("сөз, сөз, сөз")
  - Sometimes switches to Russian mid-sentence
  - No factual accuracy
  - News-style prompts produce the most natural output

**Sample generations:**

> **Prompt:** Қазақстан — бұл
> **Output:** Қазақстан — бұл бір жылда екі жыл бойы бір-ақ күнде бірінші болып келеді...

> **Prompt:** Бүгін ауа райы
> **Output:** Бүгін ауа райы болжамын. Аймақтан ауа райы жауын-шашын күтіледі. Бұл туралы бүгін облыс әкімі Нұрлан Байбек айтты...

> **Prompt:** Президент бүгін халыққа үндеу жасап,
> **Output:** Президент бүгін халыққа үндеу жасап, осы мәселеге қатысты пікір білдірді. «Бұл әрине, бүгінгі қазақ тіліндегі маңызды мәселеге тоқталып...

**Observations at step 2000:**
- Model already learned Kazakh morphology and common phrases
- Heavily biased toward news/government press release style (dominant in training data)
- Repetition is the main failure mode
- Loss 4.27 = perplexity ~71

### checkpoint-8000 (step 8000, eval_loss 3.654)

- Generation quality improved noticeably vs step 2000:
  - More coherent sentences
  - Less Russian code-switching
  - COVID statistics appeared (reflects training data)
  - Repetitions still present

**Sample generations:**

> **Prompt:** Қазақстан — бұл
> **Output:** Қазақстан — бұл жылдың 1 маусымынан бастап 23 маусым аралығында COVID-19 індетін жұқтырған 12 наурыздағы...

> **Prompt:** Бүгін ауа райы
> **Output:** Бүгін ауа райы бойынша 2019 жылғы 29 сәуір – 30 мамыр аралығында болатын бір қатар іс-шаралар...

> **Prompt:** Жігіт атқа мініп, далаға шықты.
> **Output:** Жігіт атқа мініп, далаға шықты. Бұл жөнінде ҚР Денсаулық сақтау министрлігінің ресми өкілі Бағдат Қожахметов...

### checkpoint-22000 (step 22000, eval_loss 3.414)

- More complex academic Kazakh vocabulary
- Better narrative structure (esp. literary prompts)
- Still repetitive, but longer coherent spans
- Perplexity ~30

**Sample generations:**

> **Prompt:** Қазақстан — бұл
> **Output:** Қазақстан — бұл бағдарлама бойынша оқытудың мазмұнын анықтау үшін білім берудің педагогикалық тұрғыдағы тиімділігін...

> **Prompt:** Бүгін ауа райы
> **Output:** Бүгін ауа райы бойынша жалпы білім беретін мектептердегі оқу-тәрбие процесінде қазақ тілін...

> **Prompt:** Жігіт атқа мініп, далаға шықты.
> **Output:** Жігіт атқа мініп, далаға шықты. Қазіргі кезде еліміздегі ауылдық жерлерде тұратын халықтың...

### checkpoint-24000 (pending inference)

## Notes

- Training started: 2026-02-07 ~10:30 UTC
- Inference runs on CPU (CUDA_VISIBLE_DEVICES="") to not disturb training
- Each prompt takes ~80s on CPU for 100 tokens (50M model)
- Using 3 prompts for quick inference (was 18, reduced for speed)
- save_total_limit=3 means only latest 3 checkpoints are kept on disk
- Eval runs every 2000 steps (~462s per eval pass)
