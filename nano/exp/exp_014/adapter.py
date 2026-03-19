"""
exp_014/adapter.py — Cascading LLM: Qwen2.5-0.5B → escalation → Qwen3-1.7B

Primary: Qwen2.5-0.5B-Instruct (MLX 4bit)
Escalation: Qwen3-1.7B (MLX 4bit)
Translation: HPLT KK↔EN + Marian RU↔EN
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


class QwenAdapter:
    """MLX адаптер для Qwen моделей."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)

    def generate_response(self, system_prompt: str, user_msg: str, max_tokens: int = 100) -> str:
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
