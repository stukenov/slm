"""
exp_015/adapter.py — Гибридный переводчик + Qwen MLX

KK↔EN: HPLT CTranslate2 float32 (лучшее качество казахского)
Остальные языки: NLLB-200-distilled-600M CTranslate2
Qwen2.5-0.5B-Instruct: primary LLM (MLX 4bit)
Qwen3-1.7B: escalation LLM (MLX 4bit)
"""

import re
import ctranslate2
import sentencepiece as spm
from transformers import AutoTokenizer
from mlx_lm import load, generate
from pathlib import Path

HF_CACHE = Path.home() / ".cache/huggingface/hub"

# NLLB language codes (для не-KK языков)
NLLB_LANGS = {
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "tr": "tur_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "ja": "jpn_Jpan",
}

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
    """HPLT CTranslate2 float32 — KK↔EN специализированный переводчик."""

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


class NLLBTranslator:
    """NLLB-200-distilled-600M via CTranslate2 — универсальный переводчик."""

    MODEL_ID = "facebook/nllb-200-distilled-600M"
    CT2_DIR = Path(__file__).parent / "models" / "nllb"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.translator = ctranslate2.Translator(
            str(self.CT2_DIR), device="cpu", compute_type="float32"
        )

    def _translate_one(self, text: str, src_code: str, tgt_code: str) -> str:
        self.tokenizer.src_lang = src_code
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text, truncation=True, max_length=128)
        )
        results = self.translator.translate_batch(
            [tokens], target_prefix=[[tgt_code]], beam_size=2
        )
        output_tokens = results[0].hypotheses[0]
        if output_tokens and output_tokens[0] == tgt_code:
            output_tokens = output_tokens[1:]
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(output_tokens)
        )

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_code = NLLB_LANGS.get(src_lang, src_lang)
        tgt_code = NLLB_LANGS.get(tgt_lang, tgt_lang)

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return ""

        translated = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            translated.append(self._translate_one(sent, src_code, tgt_code))

        return " ".join(translated)


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
