"""Character-level tokenizer for Kazakh TTS."""
from __future__ import annotations

# Kazakh Cyrillic alphabet + common punctuation + space
KAZAKH_CHARS = (
    " "  # space
    "абвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
    "АБВГҒДЕЁЖЗИЙКҚЛМНҢОӨПРСТУҰҮФХҺЦЧШЩЪЫІЬЭЮЯ"
    "әӘ"
    ".,!?;:-–—\"'()0123456789"
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class KazakhCharTokenizer:
    """Character-level tokenizer for Kazakh TTS.

    No BOS/EOS — every token maps to actual audio.
    """

    def __init__(self):
        self._char2id: dict[str, int] = {}
        self._id2char: dict[int, str] = {}

        # 0 = pad, 1 = unk
        self._char2id[PAD_TOKEN] = 0
        self._id2char[0] = PAD_TOKEN
        self._char2id[UNK_TOKEN] = 1
        self._id2char[1] = UNK_TOKEN

        for i, ch in enumerate(KAZAKH_CHARS, start=2):
            if ch not in self._char2id:
                self._char2id[ch] = i
                self._id2char[i] = ch

    @property
    def vocab_size(self) -> int:
        return len(self._char2id)

    @property
    def pad_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        text = text.lower()
        return [self._char2id.get(ch, 1) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id2char.get(i, "?") for i in ids if i > 1)
