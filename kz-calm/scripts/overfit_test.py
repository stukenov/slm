"""Overfit test v2: 10 samples with correct 512-dim latent extraction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from kzcalm.model.backbone import TTSBackbone
from kzcalm.model.code_embedding import MimiLatentExtractor
from kzcalm.model.flow_head import FlowMatchingLoss, sample_euler
from kzcalm.tokenizer.sp_tokenizer import KazakhTokenizer

device = 'cuda'

# --- Load 10 samples ---
print('Loading 10 samples...')
ds = load_dataset('stukenov/kzcalm-mimi-codes-kk-v1', split='train', streaming=True)
samples = []
for s in ds:
    if s['num_frames'] <= 300:
        samples.append(s)
    if len(samples) >= 10:
        break
print(f'Got {len(samples)} samples')
for i, s in enumerate(samples):
    print(f'  [{i}] frames={s["num_frames"]} text="{s["text"][:60]}"')

# --- Tokenizer ---
tok_path = hf_hub_download('stukenov/kzcalm-sp-tokenizer-4k-kk-v1', 'tokenizer.model')
tokenizer = KazakhTokenizer(tok_path)

# --- Prepare codes batch ---
text_ids_list = []
codes_list = []
for s in samples:
    ids = tokenizer.encode(s['text'])[:256]
    text_ids_list.append(torch.tensor(ids, dtype=torch.long))
    codes_list.append(torch.tensor(s['codes'][:8], dtype=torch.long))

max_text = max(t.shape[0] for t in text_ids_list)
max_frames = max(c.shape[1] for c in codes_list)
B = len(samples)

text_ids = torch.zeros(B, max_text, dtype=torch.long, device=device)
text_mask = torch.ones(B, max_text, dtype=torch.bool, device=device)
codes = torch.zeros(B, 8, max_frames, dtype=torch.long, device=device)
codes_mask = torch.zeros(B, max_frames, device=device)

for i in range(B):
    T_t = text_ids_list[i].shape[0]
    text_ids[i, :T_t] = text_ids_list[i]
    text_mask[i, :T_t] = False
    T_c = codes_list[i].shape[1]
    codes[i, :, :T_c] = codes_list[i]
    codes_mask[i, :T_c] = 1.0

print(f'Codes batch: text={text_ids.shape}, codes={codes.shape}')

# --- Extract 512-dim latents via MimiLatentExtractor ---
extractor = MimiLatentExtractor(device=device)
with torch.no_grad():
    x1 = extractor(codes)  # (B, 2T, 512)
latent_mask = codes_mask.repeat_interleave(2, dim=1)  # (B, 2T)
print(f'x1: {x1.shape}, range=[{x1.min():.2f}, {x1.max():.2f}]')
print(f'latent_mask: {latent_mask.shape}')

# --- Small model ---
model = TTSBackbone(
    vocab_size=tokenizer.vocab_size, latent_dim=512,
    d_model=384, num_heads=6, num_layers=6, d_ff=1536,
    max_audio_frames=max_frames * 2 + 10,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model: {n_params/1e6:.1f}M params')

criterion = FlowMatchingLoss(loss_type='mse')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

# --- Overfit loop ---
print('\nTraining (overfit 10 samples, 5K steps)...')
model.train()
for step in range(5001):
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)
    t_exp = t[:, None, None]
    x_t = (1 - t_exp) * x0 + t_exp * x1

    velocity = model(
        x_t, text_ids, t,
        text_padding_mask=text_mask,
        latent_padding_mask=(latent_mask == 0),
    )
    loss = criterion(velocity, x0, x1, mask=latent_mask)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 200 == 0:
        print(f'  step={step} loss={loss.item():.6f}')

# --- Inference: predict latents -> mimi.decoder -> waveform ---
print('\nInference on overfit samples...')
model.eval()
os.makedirs('/root/kzcalm/overfit_v2_out', exist_ok=True)

for i in range(min(3, B)):
    t_ids = text_ids[i:i+1]
    n_code_frames = int(codes_mask[i].sum().item())
    n_latent_frames = n_code_frames * 2  # 2x upsample

    with torch.no_grad():
        latents = sample_euler(
            model, t_ids, n_latent_frames,
            latent_dim=512, num_steps=50, device=device,
        )
        # Decode directly through Mimi decoder
        waveform = extractor.decode_latents(latents)

    wav = waveform.squeeze().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(f'/root/kzcalm/overfit_v2_out/overfit_v2_{i}.wav', wav, 24000)
    dur = wav.shape[-1] / 24000
    text_preview = samples[i]["text"][:60]
    print(f'  overfit_v2_{i}.wav: "{text_preview}" -> {dur:.1f}s')

# Also save ground truth for comparison
print('\nSaving ground truth (codes -> mimi.decode)...')
for i in range(min(3, B)):
    n_code_frames = int(codes_mask[i].sum().item())
    real_codes = codes[i:i+1, :, :n_code_frames]
    with torch.no_grad():
        gt_wav = extractor.mimi.decode(audio_codes=real_codes).audio_values
    gt_wav = gt_wav.squeeze().cpu()
    if gt_wav.dim() == 1:
        gt_wav = gt_wav.unsqueeze(0)
    torchaudio.save(f'/root/kzcalm/overfit_v2_out/ground_truth_{i}.wav', gt_wav, 24000)
    print(f'  ground_truth_{i}.wav: {gt_wav.shape[-1]/24000:.1f}s')

print('\nDone! Compare overfit_v2_*.wav vs ground_truth_*.wav')
