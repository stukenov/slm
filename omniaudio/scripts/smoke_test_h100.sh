#!/bin/bash
# Smoke test: verify full pipeline for all 3 models before real training
# Run on cloud instance BEFORE launching full training
set -e

cd /root/slm
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

echo "========================================"
echo "Step 1: Verify packages"
echo "========================================"
python -c "import omniaudio; print('omniaudio OK')"
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.device_count()} GPUs')"
python -c "from huggingface_hub import HfApi; HfApi().whoami(); print('HF auth OK')"
python -c "from transformers import AutoModelForCausalLM; print('transformers OK')"

echo ""
echo "========================================"
echo "Step 2: Verify dataset loads"
echo "========================================"
python << 'PYEOF'
from omniaudio.data_v2 import load_speech_dataset
ds = load_speech_dataset('sozkz_mels', 'train', max_samples=4)
print(f'Dataset OK: {len(ds)} samples, keys={list(ds[0].keys())[:5]}')
PYEOF

echo ""
echo "========================================"
echo "Step 3: Verify LLM loading"
echo "========================================"
python << 'PYEOF'
import torch
from transformers import LlamaForCausalLM, AutoConfig

for name in [
    "stukenov/sozkz-core-llama-150m-kk-base-v1",
    "stukenov/sozkz-core-llama-600m-kk-base-v1",
    "stukenov/sozkz-core-llama-1b-kk-base-v1",
]:
    c = AutoConfig.from_pretrained(name)
    print(f'{name}: hidden={c.hidden_size}, layers={c.num_hidden_layers}')
    m = LlamaForCausalLM.from_pretrained(name, torch_dtype="auto")
    total = sum(p.numel() for p in m.parameters())
    print(f'  Loaded OK: {total/1e6:.1f}M params')
    del m
    torch.cuda.empty_cache()
PYEOF

echo ""
echo "========================================"
echo "Step 4: Micro-train 150M (GPU 0)"
echo "========================================"
python << 'PYEOF'
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
from omniaudio.train_v2 import load_config, train

config = load_config('omniaudio/configs/v2_llm150m_h100.yaml')
config['num_train_epochs'] = 1
config['max_train_samples'] = 64
config['max_eval_samples'] = 16
config['per_device_train_batch_size'] = 4
config['save_steps'] = 5
config['logging_steps'] = 1
config['hf_repo_id'] = None
config['experiment_name'] = 'smoke_test_150m'
train(config)
print('SMOKE TEST 150M: PASS')
PYEOF

echo ""
echo "========================================"
echo "Step 5: Micro-train 600M (GPU 0)"
echo "========================================"
python << 'PYEOF'
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
from omniaudio.train_v2 import load_config, train

config = load_config('omniaudio/configs/v2_llm600m_h100.yaml')
config['num_train_epochs'] = 1
config['max_train_samples'] = 64
config['max_eval_samples'] = 16
config['per_device_train_batch_size'] = 4
config['save_steps'] = 5
config['logging_steps'] = 1
config['hf_repo_id'] = None
config['experiment_name'] = 'smoke_test_600m'
train(config)
print('SMOKE TEST 600M: PASS')
PYEOF

echo ""
echo "========================================"
echo "Step 6: Micro-train 1B (GPU 0)"
echo "========================================"
python << 'PYEOF'
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
from omniaudio.train_v2 import load_config, train

config = load_config('omniaudio/configs/v2_llm1b_h100.yaml')
config['num_train_epochs'] = 1
config['max_train_samples'] = 64
config['max_eval_samples'] = 16
config['per_device_train_batch_size'] = 4
config['save_steps'] = 5
config['logging_steps'] = 1
config['hf_repo_id'] = None
config['experiment_name'] = 'smoke_test_1b'
train(config)
print('SMOKE TEST 1B: PASS')
PYEOF

echo ""
echo "========================================"
echo "Step 7: Verify HF repos"
echo "========================================"
python << 'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
for repo in [
    'stukenov/sozkz-core-omniaudio-150m-kk-asr-v1',
    'stukenov/sozkz-core-omniaudio-600m-kk-asr-v1',
    'stukenov/sozkz-core-omniaudio-1b-kk-asr-v1',
]:
    api.create_repo(repo_id=repo, exist_ok=True, repo_type='model')
    print(f'Repo OK: {repo}')
PYEOF

echo ""
echo "========================================"
echo "Step 8: Checkpoint round-trip test"
echo "========================================"
python << 'PYEOF'
import torch
from omniaudio.model_v2 import OmniAudioV2Model

enc_cfg = {'n_mels': 80, 'd_model': 512, 'n_heads': 8, 'n_layers': 12, 'n_conv': 2}
model = OmniAudioV2Model(
    encoder_config=enc_cfg,
    llm_name='stukenov/sozkz-core-llama-150m-kk-base-v1',
    vocab_size=50257, llm_dim=768,
)
state = torch.load('outputs/smoke_test_150m/checkpoint-best/model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=False)
model = model.cuda()
model.train(False)

mel = torch.randn(1, 80, 200).cuda()
tokens = model.generate(mel, max_new_tokens=10, eos_token_id=0)
print(f'Round-trip OK: generated {len(tokens)} tokens')
PYEOF

echo ""
echo "========================================"
echo "Step 9: Cleanup"
echo "========================================"
rm -rf outputs/smoke_test_*

echo ""
echo "========================================"
echo "ALL SMOKE TESTS PASSED"
echo "========================================"
echo "Ready: bash omniaudio/scripts/launch_h100_all.sh"
