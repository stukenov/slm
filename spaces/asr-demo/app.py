"""SozKZ -- Kazakh ASR Demo. Uses original model_v2.py from HF repo."""

import os
import spaces
import gradio as gr
import torch
import numpy as np
import librosa
import soundfile as sf
import time
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download, login

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Download and import original model code from HF repo
model_code_path = hf_hub_download("stukenov/sozkz-core-omniaudio-70m-kk-asr-v1", "src/model_v2.py")
import importlib.util
spec = importlib.util.spec_from_file_location("model_v2", model_code_path)
model_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_v2)

# Exact mel filterbank from torchaudio (pre-computed, diff=0.0)
MEL_FB = torch.load(
    hf_hub_download("stukenov/sozkz-core-omniaudio-70m-kk-asr-v2", "mel_filterbank.pt"),
    map_location="cpu", weights_only=True,
)
MEL_WINDOW = torch.hann_window(400)

def compute_mel(wav_np):
    wav = torch.from_numpy(wav_np).float()
    stft = torch.stft(wav, n_fft=400, hop_length=160, win_length=400,
                      window=MEL_WINDOW, center=True, pad_mode="reflect", return_complex=True)
    power = stft.abs().pow(2)
    mel = torch.matmul(MEL_FB.T, power)
    return torch.log(torch.clamp(mel, min=1e-10)).unsqueeze(0)


# Load models
ASR_MODELS = {
    "v2 (CTC+CE)": "stukenov/sozkz-core-omniaudio-70m-kk-asr-v2",
    "v1 (pure CE)": "stukenov/sozkz-core-omniaudio-70m-kk-asr-v1",
}
ENC_CFG = {"n_mels": 80, "d_model": 256, "n_heads": 4, "n_layers": 6, "n_conv": 2}
DEC_CFG = {"d_model": 512, "n_heads": 8, "n_layers": 8}

TOK_REPO = "stukenov/sozkz-core-gpt2-50k-kk-base-v1"
tok_file = hf_hub_download(TOK_REPO, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
tokenizer.eos_token = "<|endoftext|>"
tokenizer.eos_token_id = 0

loaded_asr = {}
for name, repo in ASR_MODELS.items():
    print(f"Loading {name} from {repo}...")
    mdl = model_v2.OmniAudioScratchModel(
        encoder_config=ENC_CFG, decoder_config=DEC_CFG, vocab_size=50257,
    )
    w = hf_hub_download(repo, "model.pt")
    sd = torch.load(w, map_location="cpu", weights_only=True)
    info = mdl.load_state_dict(sd, strict=False)
    # lm_head not in checkpoint — it's tied to embed_tokens
    mdl.lm_head.weight = mdl.embed_tokens.weight
    print(f"  missing: {len(info.missing_keys)}, unexpected: {len(info.unexpected_keys)}, lm_head tied")
    for k in info.missing_keys:
        if "rope" not in k and "inv_freq" not in k and "lm_head" not in k:
            print(f"  MISSING: {k}")
    mdl.requires_grad_(False)
    loaded_asr[name] = mdl
print("Ready.")


def generate_no_repeat(mdl, mel, max_new_tokens=100, eos_token_id=0,
                       repetition_penalty=1.5, no_repeat_ngram=4):
    """Generate with ngram blocking to prevent phrase repetition."""
    mdl.train(False)
    enc_out = mdl.encoder(mel)
    audio_embeds = mdl.projector(enc_out)
    generated = []
    combined = audio_embeds

    for _ in range(max_new_tokens):
        cos, sin = mdl.decoder_rope(combined.size(1))
        x = combined
        for layer in mdl.decoder_layers:
            x = layer(x, cos, sin)
        x = mdl.decoder_norm(x)
        logits = mdl.lm_head(x[:, -1:]).squeeze(0).squeeze(0)

        # Token repetition penalty
        if repetition_penalty != 1.0 and generated:
            for t in set(generated):
                if logits[t] > 0:
                    logits[t] /= repetition_penalty
                else:
                    logits[t] *= repetition_penalty

        # N-gram blocking
        if no_repeat_ngram > 0 and len(generated) >= no_repeat_ngram - 1:
            prefix = tuple(generated[-(no_repeat_ngram - 1):])
            for i in range(len(generated) - no_repeat_ngram + 1):
                if tuple(generated[i:i + no_repeat_ngram - 1]) == prefix:
                    logits[generated[i + no_repeat_ngram - 1]] = float("-inf")

        next_token = logits.argmax(dim=-1).item()
        if next_token == eos_token_id:
            break
        generated.append(next_token)
        token_embed = mdl.embed_tokens(torch.tensor([[next_token]], device=mel.device))
        combined = torch.cat([combined, token_embed], dim=1)

    return generated


@spaces.GPU
def transcribe(audio, model_name):
    if audio is None:
        return "No audio"
    t0 = time.perf_counter()

    # Load and resample to 16kHz mono
    if isinstance(audio, str):
        wav, sr = sf.read(audio)
        wav = np.array(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    elif isinstance(audio, tuple):
        sr, wav = audio
        wav = np.array(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav.mean(axis=0)
        if np.abs(wav).max() > 1.0:
            wav = wav / 32768.0
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    else:
        return "Unsupported format"

    wav = wav[:int(10.0 * 16000)]
    mel = compute_mel(wav)

    asr = loaded_asr.get(model_name, loaded_asr["v2 (CTC+CE)"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr.to(device)
    mel = mel.to(device)

    with torch.no_grad():
        tokens = generate_no_repeat(asr, mel, max_new_tokens=100, eos_token_id=0,
                                    repetition_penalty=2.0, no_repeat_ngram=3)

    asr.to("cpu")
    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    elapsed = time.perf_counter() - t0
    if not text:
        return f"(no speech detected, {elapsed:.1f}s)"
    return text + f"\n\n({elapsed:.1f}s)"


CSS = """
:root, :root.dark { color-scheme: light only !important; --body-background-fill: #fff !important; }
html, body { background: #fff !important; }
* { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif !important; }
.gradio-container[class] { max-width: 700px !important; margin: 0 auto !important; }
footer { display: none !important; }
#hero { padding: 28px 32px 16px; text-align: center; }
#hero h1 { font-size: 36px !important; font-weight: 700 !important; margin: 0 0 6px !important; }
#hero .sub { font-size: 16px; color: #86868b; margin: 0 0 12px; }
#hero .tag { font-size: 12px; color: #6e6e73; background: #f5f5f7; border: 1px solid #d2d2d7; padding: 4px 12px; border-radius: 100px; }
.output textarea { font-size: 20px !important; line-height: 1.6 !important; }
"""

theme = gr.themes.Base(primary_hue="blue")
theme.set(body_background_fill="#fff", block_background_fill="#fff", block_border_width="0px", block_shadow="none")

with gr.Blocks(css=CSS, theme=theme, title="SozKZ ASR") as demo:
    gr.HTML("""<div id="hero">
        <h1>SozKZ ASR</h1>
        <p class="sub">Kazakh Speech Recognition</p>
        <span class="tag">OmniAudio 70M</span>
    </div>""")

    model_sel = gr.Radio(list(ASR_MODELS.keys()), value="v2 (CTC+CE)", label="Model", interactive=True)
    audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload audio (WAV/MP3/FLAC, max 10s)")
    btn = gr.Button("Transcribe", variant="primary")
    output = gr.Textbox(label="Transcription", lines=4, elem_classes=["output"])
    btn.click(fn=transcribe, inputs=[audio_input, model_sel], outputs=output)

    gr.Examples(
        examples=[
            ["examples/wwww.wav"],
            ["examples/audio2.wav"],
            ["examples/audio3.wav"],
        ],
        inputs=[audio_input],
        label="Example audio",
    )

    gr.HTML("""<div style="text-align:center;padding:20px;font-size:12px;color:#aaa">
        <a href="https://huggingface.co/stukenov/sozkz-core-omniaudio-70m-kk-asr-v2" style="color:#888">v2 Model</a> |
        <a href="https://huggingface.co/stukenov/sozkz-core-omniaudio-70m-kk-asr-v1" style="color:#888">v1 Model</a> |
        <a href="https://huggingface.co/spaces/stukenov/sozkz-kazakh-llm-demo" style="color:#888">LLM Demo</a> |
        <a href="https://huggingface.co/stukenov" style="color:#888">stukenov</a>
    </div>""")

if __name__ == "__main__":
    demo.launch(ssr_mode=False, show_error=True)
