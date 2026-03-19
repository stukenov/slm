"""
exp_009/adapter.py — CTranslate2 INT8 адаптер для HPLT переводчиков

Оборачивает CTranslate2 модели в интерфейс совместимый с нашим pipeline.
"""

import ctranslate2
import sentencepiece as spm
from pathlib import Path

HF_CACHE = Path.home() / ".cache/huggingface/hub"

MODELS = {
    "kk_en": {
        "ct2_dir": "models/kk_en",
        "spm": str(HF_CACHE / "models--HPLT--translate-kk-en-v2.0-hplt_opus"
                   / "snapshots/8de65915b0d5d38e6f3216dc363574d337080045/model.kk-en.spm"),
    },
    "en_kk": {
        "ct2_dir": "models/en_kk",
        "spm": str(HF_CACHE / "models--HPLT--translate-en-kk-v2.0-hplt_opus"
                   / "snapshots/dff70263645f2f428068e8452ae2e1aa9cd6454d/model.en-kk.spm"),
    },
}


class TranslatorAdapter:
    """Адаптер: CTranslate2 INT8 модель + SentencePiece токенизатор."""

    def __init__(self, direction: str, base_dir: str = None):
        """direction: 'kk_en' or 'en_kk'"""
        cfg = MODELS[direction]
        base = Path(base_dir) if base_dir else Path(__file__).parent
        ct2_path = str(base / cfg["ct2_dir"])

        self.translator = ctranslate2.Translator(ct2_path, device="cpu", compute_type="int8")
        self.sp = spm.SentencePieceProcessor(cfg["spm"])
        self.direction = direction

    def translate(self, text: str) -> str:
        tokens = self.sp.encode(text, out_type=str)
        result = self.translator.translate_batch([tokens], beam_size=4)
        return self.sp.decode(result[0].hypotheses[0])


if __name__ == "__main__":
    print("=== KK→EN ===")
    kk2en = TranslatorAdapter("kk_en")
    for t in ["Астана Қазақстанның астанасы.", "бір қосу екі нешеге тең?", "сәлем, қалайсың?"]:
        print(f"  {t} → {kk2en.translate(t)}")

    print("\n=== EN→KK ===")
    en2kk = TranslatorAdapter("en_kk")
    for t in ["Astana is the capital of Kazakhstan.", "one plus two is three.", "hello, how are you?"]:
        print(f"  {t} → {en2kk.translate(t)}")
