"""
exp_012/adapter.py — Адаптеры для переводчиков и SmolLM2

HPLT CTranslate2 float32: KK↔EN
Helsinki-NLP MarianMT: RU↔EN
SmolLM2-360M-Instruct: Think + Math + Code + Err
"""

import ctranslate2
import sentencepiece as spm
import torch
from pathlib import Path
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoModelForCausalLM, AutoTokenizer,
)

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
    """SmolLM2-360M-Instruct — одна модель на все задачи."""

    MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

    def __init__(self, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, dtype=torch.float32
        ).to(device)
        self.model.eval()
        self.device = device

    def generate(self, system_prompt: str, user_msg: str, max_new_tokens: int = 60) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
