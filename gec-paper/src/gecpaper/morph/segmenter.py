from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_segmentation_data(
    wordforms: list[str],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_path: Path | None = None,
    batch_size: int = 32,
) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logger.info("Loading %s for morpheme segmentation...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompt_template = (
        "Kazakh morpheme segmentation. Split the word into morphemes separated by |.\n"
        "Examples:\n"
        "балаларға -> бала|лар|ға\n"
        "мектептерде -> мектеп|тер|де\n"
        "оқушылардың -> оқушы|лар|дың\n"
        "Word: {word} -> "
    )

    results = []
    for i in range(0, len(wordforms), batch_size):
        batch = wordforms[i : i + batch_size]
        for word in batch:
            prompt = prompt_template.format(word=word)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                )
            decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            segmented = decoded.strip().split("\n")[0].strip()

            reconstructed = segmented.replace("|", "")
            if reconstructed != word:
                continue

            results.append({"word": word, "segmented": segmented})

        if (i + batch_size) % 500 == 0:
            logger.info("Processed %d/%d words", min(i + batch_size, len(wordforms)), len(wordforms))

    logger.info("Generated %d segmentations from %d words", len(results), len(wordforms))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


class CharSegmenter:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}
        if model_path:
            self.load(model_path)

    def build_vocab(self, words: list[str]) -> None:
        chars = set()
        for w in words:
            chars.update(w)
        chars.add("|")
        self.char2id = {"<pad>": 0, "<unk>": 1}
        for i, c in enumerate(sorted(chars), start=2):
            self.char2id[c] = i
        self.id2char = {v: k for k, v in self.char2id.items()}

    def segment(self, word: str) -> str:
        if not self.model:
            return word
        import torch
        chars = [self.char2id.get(c, 1) for c in word]
        inputs = torch.tensor([chars])
        with torch.no_grad():
            logits = self.model(inputs)
        preds = logits.argmax(dim=-1)[0]
        result = []
        for char, pred in zip(word, preds):
            if pred == 1 and result:
                result.append("|")
            result.append(char)
        return "".join(result)

    def segment_text(self, text: str) -> str:
        words = text.split()
        return " ".join(self.segment(w) for w in words)

    def save(self, path: str) -> None:
        import torch
        data = {
            "char2id": self.char2id,
            "model_state": self.model.state_dict() if self.model else None,
        }
        torch.save(data, path)

    def load(self, path: str) -> None:
        import torch
        data = torch.load(path, weights_only=False)
        self.char2id = data["char2id"]
        self.id2char = {v: k for k, v in self.char2id.items()}
