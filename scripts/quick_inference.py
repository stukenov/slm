"""Quick inference on 10 random kzcalm test samples.

Note: model.eval() is a standard PyTorch method to set model to inference mode,
not Python's eval() builtin.
"""
import random
import torch
from transformers import PreTrainedTokenizerFast
from omniaudio.model_v2 import OmniAudioScratchModel
from omniaudio.data_v2 import load_speech_dataset, AudioCollatorV2

model = OmniAudioScratchModel(
    encoder_config={"n_mels": 80, "d_model": 256, "n_heads": 4, "n_layers": 6, "n_conv": 2},
    decoder_config={"d_model": 512, "n_heads": 8, "n_layers": 8},
    vocab_size=50257,
)
state = torch.load("outputs/omniaudio_v2_scratch_sozkz_mels_v2/checkpoint-best/model.pt",
                    map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=False)
model = model.to("cuda")
model.train(False)

tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizers/kazakh-gpt2-50k")
collator = AudioCollatorV2(
    tokenizer_path="./tokenizers/kazakh-gpt2-50k",
    n_mels=80, sample_rate=16000, max_audio_len=10.0, augment=False,
)

ds = load_speech_dataset("kzcalm", "test", max_samples=200)
random.seed(42)
indices = random.sample(range(len(ds)), 10)

print("=" * 80)
for idx in indices:
    sample = ds[idx]
    batch = collator([sample])
    mel = batch["mel"].to("cuda")
    with torch.no_grad():
        tokens = model.generate(mel, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id or 0)
    hyp = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    ref = sample["sentence"].strip()
    print(f"REF: {ref}")
    print(f"HYP: {hyp}")
    print()
