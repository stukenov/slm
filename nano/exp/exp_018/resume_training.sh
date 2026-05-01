#!/bin/bash
set -euo pipefail

cd /home/ubuntu/exp018
source .venv/bin/activate

SKIP=55539970
CORPUS=/home/ubuntu/exp018/output/corpus_segmented.txt
TOKENIZER_REPO="stukenov/sozkz-morphbpe-256k-kk-v1"
CORPUS_REPO="stukenov/sozkz-corpus-segmented-kk-v1"

echo "=== Resuming segmentation from doc $SKIP ==="
echo "Started: $(date)"

python3 -c "
import time, sys
sys.path.insert(0, '.')
from train_tokenizer import load_texts
from morpheme_segmenter import MorphemeSegmenter

SKIP = $SKIP
seg = MorphemeSegmenter(backend='qazcorpora')
datasets = load_texts()

processed = 0
skipped_total = 0
t0 = time.time()

with open('$CORPUS', 'a', encoding='utf-8') as f:
    for ds, text_col in datasets:
        for row in ds:
            if skipped_total < SKIP:
                skipped_total += 1
                continue
            text = row[text_col]
            if text and text.strip():
                f.write(seg.segment(text) + '\n')
                processed += 1
                if processed % 100_000 == 0:
                    elapsed = time.time() - t0
                    total = SKIP + processed
                    print(f'  [seg] {total:,} docs (+{processed:,} new) | {elapsed:.0f}s', flush=True)

total = SKIP + processed
print(f'  [seg] DONE: {total:,} total docs ({processed:,} new) | {time.time()-t0:.0f}s', flush=True)
"

echo ""
echo "=== Segmentation complete. Uploading corpus to HF ==="

python3 -c "
from train_tokenizer import upload_corpus_to_hf
upload_corpus_to_hf('$CORPUS', '$CORPUS_REPO')
"

echo ""
echo "=== Training BPE 256K ==="

python3 -c "
import os, time
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders, processors
from train_tokenizer import SPECIAL_TOKENS, get_unicode_digits, verify_tokenizer

vocab_size = 256000
corpus_file = '$CORPUS'
output_dir = '/home/ubuntu/exp018/output/morphbpe-qazcorpora-256k'
os.makedirs(output_dir, exist_ok=True)

extra_tokens = SPECIAL_TOKENS + get_unicode_digits()
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(pattern=Regex(r'\x1F'), behavior='removed'),
    pre_tokenizers.ByteLevel(add_prefix_space=False),
])
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=extra_tokens,
    min_frequency=2,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

t0 = time.time()
print(f'Training BPE (vocab={vocab_size})...', flush=True)
tokenizer.train([corpus_file], trainer=trainer)
elapsed = time.time() - t0
print(f'Done in {elapsed:.0f}s. Vocab: {tokenizer.get_vocab_size()}')

tokenizer.save(os.path.join(output_dir, 'tokenizer.json'))

from transformers import PreTrainedTokenizerFast
hf_tok = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    eos_token='<|endoftext|>',
    bos_token='<|startoftext|>',
    pad_token='<|padding|>',
    unk_token=None,
    model_max_length=4096,
)
hf_tok.save_pretrained(output_dir)

fertility = verify_tokenizer(hf_tok, 'morphbpe-qazcorpora-256k')
print(f'Pushing to $TOKENIZER_REPO...')
hf_tok.push_to_hub('$TOKENIZER_REPO')
print('Upload complete!')
"

echo ""
echo "=== ALL DONE ==="
echo "Finished: $(date)"
