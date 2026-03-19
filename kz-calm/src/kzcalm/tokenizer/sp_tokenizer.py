"""SentencePiece tokenizer wrapper for Kazakh TTS."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class KazakhTokenizer:
    """SentencePiece tokenizer for Kazakh text."""

    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self.sp.Encode(text)
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.Decode(ids)

    @staticmethod
    def train(
        input_file: str,
        model_prefix: str,
        vocab_size: int = 4096,
        character_coverage: float = 1.0,
        model_type: str = "bpe",
    ):
        """Train a new SentencePiece model."""
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
        )
