"""
exp_013/adapter.py — Адаптеры: MLX SmolLM2-1.7B + HPLT + Marian

HPLT CTranslate2 float32: KK↔EN
Helsinki-NLP MarianMT: RU↔EN
SmolLM2-1.7B-Instruct: Math + Code + Err (MLX, Metal GPU)
"""

import ctranslate2
import sentencepiece as spm
import torch
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from mlx_lm import load, generate

HF_CACHE = Path.home() / ".cache/huggingface/hub"

HPLT_MODELS = {
    "kk_en": {
        "ct2_dir": "../exp_009/models/kk_en",
        "spm": str(HF_CACHE / "models--HPLT--translate-kk-en-v2.0-hplt_opus"
                   / "snapshots/8de65915b0d5d38e6f3216dc363574d337080045/model.kk-en.spm"),
    },
    "en_kk": {
        "ct2_dir": "../exp_009/models/en_kk",
        "spm": str(HF_CACHE / "models--HPLT--translate-en-kk-v2.0-hplt_opus"
                   / "snapshots/dff70263645f2f428068e8452ae2e1aa9cd6454d/model.en-kk.spm"),
    },
}


class HPLTTranslator:
    """HPLT CTranslate2 float32 для KK↔EN."""

    def __init__(self, direction: str):
        cfg = HPLT_MODELS[direction]
        base = Path(__file__).parent
        ct2_path = str(base / cfg["ct2_dir"])
        self.translator = ctranslate2.Translator(ct2_path, device="cpu", compute_type="float32")
        self.sp = spm.SentencePieceProcessor(cfg["spm"])

    def translate(self, text: str) -> str:
        tokens = self.sp.encode(text, out_type=str)
        result = self.translator.translate_batch([tokens], beam_size=4)
        return self.sp.decode(result[0].hypotheses[0])


class MarianTranslator:
    """Helsinki-NLP Opus-MT для RU↔EN."""

    def __init__(self, direction: str):
        model_map = {
            "ru_en": "Helsinki-NLP/opus-mt-ru-en",
            "en_ru": "Helsinki-NLP/opus-mt-en-ru",
        }
        model_id = model_map[direction]
        self.tokenizer = MarianTokenizer.from_pretrained(model_id)
        self.model = MarianMTModel.from_pretrained(model_id)
        self.model.eval()

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = self.model.generate(**inputs, num_beams=4, max_new_tokens=128)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class SmolLMAdapter:
    """SmolLM2-1.7B-Instruct через MLX (Metal GPU, 4bit)."""

    MODEL_ID = "mlx-community/SmolLM2-1.7B-Instruct"

    def __init__(self):
        self.model, self.tokenizer = load(self.MODEL_ID)

    def generate_response(self, system_prompt: str, user_msg: str, max_tokens: int = 60) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(
            self.model, self.tokenizer, prompt=prompt,
            max_tokens=max_tokens, verbose=False,
        )
        return response.strip()
