#!/bin/bash
set -euo pipefail

cd /home/ubuntu/exp018
source .venv/bin/activate

CORPUS=/home/ubuntu/exp018/output/corpus_segmented.txt
TOKENIZER_REPO="stukenov/sozkz-morphbpe-256k-kk-v1"
CORPUS_REPO="stukenov/sozkz-corpus-segmented-kk-v1"

echo "=== Step 1: Upload corpus in chunks ==="
echo "Started: $(date)"

python3 -c "
import os
from datasets import Dataset
from huggingface_hub import HfApi

corpus = '$CORPUS'
repo = '$CORPUS_REPO'
chunk_size = 5_000_000

api = HfApi()
try:
    api.create_repo(repo, repo_type='dataset', exist_ok=True)
except Exception as e:
    print(f'Repo create: {e}')

shard = 0
buf = []
total = 0

with open(corpus, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if line:
            buf.append(line)
            total += 1
        if len(buf) >= chunk_size:
            ds = Dataset.from_dict({'text_segmented': buf})
            fname = f'data/train-{shard:05d}-of-99999.parquet'
            ds.to_parquet(f'/tmp/shard_{shard}.parquet')
            api.upload_file(
                path_or_fileobj=f'/tmp/shard_{shard}.parquet',
                path_in_repo=fname,
                repo_id=repo,
                repo_type='dataset',
            )
            os.remove(f'/tmp/shard_{shard}.parquet')
            print(f'  Uploaded shard {shard}: {len(buf):,} rows (total {total:,})', flush=True)
            buf = []
            shard += 1

if buf:
    ds = Dataset.from_dict({'text_segmented': buf})
    fname = f'data/train-{shard:05d}-of-99999.parquet'
    ds.to_parquet(f'/tmp/shard_{shard}.parquet')
    api.upload_file(
        path_or_fileobj=f'/tmp/shard_{shard}.parquet',
        path_in_repo=fname,
        repo_id=repo,
        repo_type='dataset',
    )
    os.remove(f'/tmp/shard_{shard}.parquet')
    print(f'  Uploaded shard {shard}: {len(buf):,} rows (total {total:,})', flush=True)

print(f'Corpus upload DONE: {total:,} rows in {shard+1} shards', flush=True)
"

echo ""
echo "=== Step 2: Train BPE 256K ==="

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
