# exp_018: Morpheme BPE 256K Tokenizer

## Context
In exp_017 we trained a morpheme-aware BPE tokenizer with 100K vocab using the HyperCLOVA X approach (qazcorpora BiLSTM segmenter). The tokenizer was published as `stukenov/sozkz-morphbpe-100k-kk-v1`.

The segmented corpus from exp_017 was lost when the training instance was destroyed.

256K chosen to match Gemma-level vocab size — large enough for full morpheme coverage in agglutinative Kazakh.

## Goal
1. Train a **256K vocab** morpheme-aware BPE tokenizer for Kazakh
2. **Save the segmented corpus to HuggingFace** so it's not lost again
3. Upload the tokenizer to HuggingFace as `stukenov/sozkz-morphbpe-256k-kk-v1`
4. Run on AWS EC2 and terminate immediately after upload

## Architecture
Same as exp_017:
- Pre-segment text into morphemes using qazcorpora BiLSTM model
- MORPH_SEP (`\x1F`) inserted between morphemes
- ByteLevel BPE trained on segmented text — merges cannot cross morpheme boundaries
- Dataset: `stukenov/ekitil-corpus-annotated-kk-v1` (filtered: kaz, confidence >= 0.95)

## Changes from exp_017
- `vocab_size`: 100K -> 256K
- New `--save-corpus-hf` flag to upload segmented corpus as HF dataset
- Output naming uses actual vocab size (not hardcoded "100k")
- `launch_aws.sh` for one-shot EC2 training + upload + terminate

## How to run

### Locally (testing)
```bash
python train_tokenizer.py --max-samples 10000
```

### On AWS
```bash
bash launch_aws.sh
```

## HuggingFace targets
- Tokenizer: `stukenov/sozkz-morphbpe-256k-kk-v1`
- Segmented corpus: `stukenov/sozkz-corpus-segmented-kk-v1`
