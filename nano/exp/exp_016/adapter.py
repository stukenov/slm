"""
exp_016/adapter.py — Гибридный переводчик + NER + Qwen MLX

KK↔EN: HPLT CTranslate2 float32 + NER (xlm-roberta-base) для имён
RU↔EN: Helsinki-NLP/opus-mt CTranslate2 (быстрый, качественный)
Остальные языки: NLLB-200-distilled-600M CTranslate2 (fallback)
Qwen2.5-0.5B-Instruct: primary LLM (MLX 4bit)
Qwen3-1.7B: escalation LLM (MLX 4bit)
"""

import re
import ctranslate2
import sentencepiece as spm
from transformers import AutoTokenizer, MarianTokenizer, pipeline
from mlx_lm import load, generate
from pathlib import Path

HF_CACHE = Path.home() / ".cache/huggingface/hub"
MODELS_DIR = Path(__file__).parent / "models"

# NLLB language codes (fallback для редких языков)
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


# Казахская кириллица → латиница (практическая английская транслитерация)
KK_TRANSLIT = {
    "а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g",
    "д": "d", "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "қ": "k", "л": "l", "м": "m", "н": "n",
    "ң": "n", "о": "o", "ө": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ұ": "u", "ү": "u", "ф": "f", "х": "kh",
    "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "і": "i", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}


def translit_kk(text: str) -> str:
    """Транслитерация казахской кириллицы → латиница."""
    result = []
    for ch in text:
        lower = ch.lower()
        if lower in KK_TRANSLIT:
            tr = KK_TRANSLIT[lower]
            result.append(tr.capitalize() if ch.isupper() else tr)
        else:
            result.append(ch)
    return "".join(result)


class KazakhNER:
    """NER для казахского языка — xlm-roberta-base."""

    def __init__(self):
        self.ner = pipeline(
            "ner",
            model="Davlan/xlm-roberta-base-ner-hrl",
            aggregation_strategy="simple",
        )

    def extract_entities(self, text: str) -> list[dict]:
        """Возвращает список {word, entity_group, start, end, translit}."""
        entities = self.ner(text)
        for e in entities:
            e["translit"] = translit_kk(e["word"])
        return entities


class HPLTTranslator:
    """HPLT CTranslate2 float32 — KK↔EN с NER-защитой имён."""

    def __init__(self, direction: str, ner: KazakhNER | None = None):
        cfg = HPLT_MODELS[direction]
        base = Path(__file__).parent
        ct2_path = str(base / cfg["ct2_dir"])
        self.translator = ctranslate2.Translator(ct2_path, device="cpu", compute_type="float32")
        self.sp = spm.SentencePieceProcessor(cfg["spm"])
        self.ner = ner
        self.direction = direction

    def translate(self, text: str) -> str:
        # NER: заменяем имена на транслит ДО перевода
        protected = text
        entities = []
        if self.ner and self.direction == "kk_en":
            entities = self.ner.extract_entities(text)
            # Заменяем с конца чтобы не сбить индексы
            for e in sorted(entities, key=lambda x: x["start"], reverse=True):
                if e["entity_group"] in ("PER", "LOC", "ORG"):
                    protected = protected[:e["start"]] + e["translit"] + protected[e["end"]:]

        tokens = self.sp.encode(protected, out_type=str)
        result = self.translator.translate_batch([tokens], beam_size=4)
        return self.sp.decode(result[0].hypotheses[0])


class MarianCT2Translator:
    """Helsinki-NLP/opus-mt via CTranslate2 — RU↔EN переводчик."""

    def __init__(self, direction: str):
        model_map = {
            "ru_en": "Helsinki-NLP/opus-mt-ru-en",
            "en_ru": "Helsinki-NLP/opus-mt-en-ru",
        }
        self.tokenizer = MarianTokenizer.from_pretrained(model_map[direction])
        ct2_path = str(MODELS_DIR / direction)
        self.translator = ctranslate2.Translator(ct2_path, device="cpu", compute_type="float32")

    def translate(self, text: str) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text, truncation=True, max_length=128)
        )
        result = self.translator.translate_batch([tokens], beam_size=4)
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(result[0].hypotheses[0]),
            skip_special_tokens=True,
        )


class NLLBTranslator:
    """NLLB-200-distilled-600M via CTranslate2 — fallback для редких языков."""

    MODEL_ID = "facebook/nllb-200-distilled-600M"
    CT2_DIR = MODELS_DIR / "nllb"

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
        # Убираем <think>...</think> блоки (Qwen3)
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
        # Если <think> не закрыт (обрезан по max_tokens) — берём текст после него
        if "<think>" in response and "</think>" not in response:
            response = ""  # вся генерация ушла в think — пусто
        return response.strip()
