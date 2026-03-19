"""Diagnostic: test different decode paths to find the correct one."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import MimiModel

device = 'cuda'

# Load one sample
print('Loading sample...')
ds = load_dataset('stukenov/kzcalm-mimi-codes-kk-v1', split='train', streaming=True)
sample = next(iter(ds))
codes_raw = sample['codes'][:8]  # 8 codebooks
T = sample['num_frames']
print(f'Text: "{sample["text"][:80]}"')
print(f'Frames: {T}, Codebooks: {len(codes_raw)}')

codes = torch.tensor(codes_raw, dtype=torch.long, device=device).unsqueeze(0)  # (1, 8, T)
print(f'codes shape: {codes.shape}')

# Load Mimi
mimi = MimiModel.from_pretrained('kyutai/mimi').to(device)
for p in mimi.parameters():
    p.requires_grad = False
q = mimi.quantizer

os.makedirs('/root/kzcalm/diag_out', exist_ok=True)

# ========== TEST 1: Full Mimi encode->decode (ground truth) ==========
# We don't have the original audio, so skip this.
# Instead, test: codes -> quantizer.decode -> decoder -> audio

# ========== TEST 2: Codes through proper quantizer decode ==========
print('\n--- Test 2: codes -> quantizer.decode -> decoder ---')
# Mimi quantizer decode expects codes as (B, K, T) but uses its own internal decode
# Let's trace what quantizer.decode does
sem_q = q.semantic_residual_vector_quantizer
ac_q = q.acoustic_residual_vector_quantizer

# Semantic: code 0
sem_codes = codes[:, 0:1, :]  # (1, 1, T)
# Acoustic: codes 1-7
ac_codes = codes[:, 1:8, :]   # (1, 7, T)

# Semantic decode path
sem_emb = sem_q.layers[0].codebook.embed[sem_codes[:, 0, :]]  # (1, T, 256)
sem_emb_t = sem_emb.transpose(1, 2)  # (1, 256, T)
sem_out = sem_q.output_proj(sem_emb_t)  # (1, 512, T)
print(f'sem_out: {sem_out.shape}, range=[{sem_out.min():.3f}, {sem_out.max():.3f}]')

# Acoustic decode path
ac_emb = torch.zeros(1, T, 256, device=device)
for i in range(7):
    ac_emb += ac_q.layers[i].codebook.embed[ac_codes[:, i, :]]
ac_emb_t = ac_emb.transpose(1, 2)  # (1, 256, T)
ac_out = ac_q.output_proj(ac_emb_t)  # (1, 512, T)
print(f'ac_out: {ac_out.shape}, range=[{ac_out.min():.3f}, {ac_out.max():.3f}]')

# Sum and decode
decoder_input = sem_out + ac_out  # (1, 512, T)
print(f'decoder_input: {decoder_input.shape}')
with torch.no_grad():
    wav2 = mimi.decoder(decoder_input)
wav2 = wav2.squeeze().cpu()
if wav2.dim() == 1:
    wav2 = wav2.unsqueeze(0)
torchaudio.save('/root/kzcalm/diag_out/test2_proper_decode.wav', wav2, 24000)
print(f'Saved test2: {wav2.shape[-1]/24000:.1f}s')

# ========== TEST 3: Our current approach (sum all -> sem_proj + ac_proj) ==========
print('\n--- Test 3: sum all 8 codebooks -> sem_proj + ac_proj (our current method) ---')
all_emb = torch.zeros(1, T, 256, device=device)
# codebook 0 (semantic)
all_emb += sem_q.layers[0].codebook.embed[codes[:, 0, :]]
# codebooks 1-7 (acoustic)
for i in range(7):
    all_emb += ac_q.layers[i].codebook.embed[codes[:, i+1, :]]
all_emb_t = all_emb.transpose(1, 2)  # (1, 256, T)
mixed_out = sem_q.output_proj(all_emb_t) + ac_q.output_proj(all_emb_t)
print(f'mixed_out: {mixed_out.shape}, range=[{mixed_out.min():.3f}, {mixed_out.max():.3f}]')
with torch.no_grad():
    wav3 = mimi.decoder(mixed_out)
wav3 = wav3.squeeze().cpu()
if wav3.dim() == 1:
    wav3 = wav3.unsqueeze(0)
torchaudio.save('/root/kzcalm/diag_out/test3_mixed_decode.wav', wav3, 24000)
print(f'Saved test3: {wav3.shape[-1]/24000:.1f}s')

# ========== TEST 4: Semantic-only decode ==========
print('\n--- Test 4: semantic codebook only -> sem_proj -> decoder ---')
sem_only = sem_q.output_proj(sem_emb_t)
with torch.no_grad():
    wav4 = mimi.decoder(sem_only)
wav4 = wav4.squeeze().cpu()
if wav4.dim() == 1:
    wav4 = wav4.unsqueeze(0)
torchaudio.save('/root/kzcalm/diag_out/test4_sem_only.wav', wav4, 24000)
print(f'Saved test4: {wav4.shape[-1]/24000:.1f}s')

# ========== TEST 5: Use Mimi's own decode method with codes ==========
print('\n--- Test 5: mimi.decode(audio_codes=...) ---')
# Mimi.decode expects audio_codes (B, K, T)
# Need to figure out what format it expects
# The full 32 codebooks: pad missing ones with 0
full_codes = torch.zeros(1, 32, T, dtype=torch.long, device=device)
full_codes[:, :8, :] = codes
try:
    with torch.no_grad():
        wav5 = mimi.decode(audio_codes=full_codes).audio_values
    wav5 = wav5.squeeze().cpu()
    if wav5.dim() == 1:
        wav5 = wav5.unsqueeze(0)
    torchaudio.save('/root/kzcalm/diag_out/test5_mimi_decode.wav', wav5, 24000)
    print(f'Saved test5: {wav5.shape[-1]/24000:.1f}s')
except Exception as e:
    print(f'Test 5 failed: {e}')

# ========== TEST 6: Use only 8 codebooks via mimi.decode ==========
print('\n--- Test 6: mimi.decode with 8 codebooks ---')
try:
    with torch.no_grad():
        wav6 = mimi.decode(audio_codes=codes).audio_values
    wav6 = wav6.squeeze().cpu()
    if wav6.dim() == 1:
        wav6 = wav6.unsqueeze(0)
    torchaudio.save('/root/kzcalm/diag_out/test6_mimi_decode_8cb.wav', wav6, 24000)
    print(f'Saved test6: {wav6.shape[-1]/24000:.1f}s')
except Exception as e:
    print(f'Test 6 failed: {e}')

print('\n=== Summary ===')
print('test2: proper separate sem/ac decode path')
print('test3: our current mixed approach (likely broken)')
print('test4: semantic only')
print('test5: mimi.decode with 32 codebooks (padded)')
print('test6: mimi.decode with 8 codebooks')
print('\nListen and compare!')
