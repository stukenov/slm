"""Wrapper to run CosyVoice extract scripts with soundfile backend.

torchaudio 2.10 requires torchcodec which is broken on this server.
This script patches torchaudio.load to use soundfile instead.
"""
import sys
import os
import runpy

# Patch torchaudio.load before importing anything else
import torchaudio
import soundfile as sf
import torch


def _sf_load(path, **kwargs):
    data, sr = sf.read(str(path))
    if data.ndim == 1:
        data = data[None, :]
    else:
        data = data.T
    return torch.tensor(data, dtype=torch.float32), sr


torchaudio.load = _sf_load

# Now run the actual CosyVoice script
mode = sys.argv[1]
sys.argv = sys.argv[1:]  # remove mode arg

cosyvoice_dir = "/root/slm/CosyVoice"
sys.path.insert(0, cosyvoice_dir)

if mode == "embeddings":
    script = os.path.join(cosyvoice_dir, "tools/extract_embedding.py")
elif mode == "tokens":
    script = os.path.join(cosyvoice_dir, "tools/extract_speech_token.py")
else:
    print(f"Unknown mode: {mode}. Use 'embeddings' or 'tokens'")
    sys.exit(1)

sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
