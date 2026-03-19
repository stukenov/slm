"""
exp_010/smollm_adapter.py — SmolLM2-135M-Instruct адаптер

Заменяет тойшную Math модель на реальный instruct LLM.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SmolLMAdapter:
    """Адаптер для SmolLM2-135M-Instruct."""

    MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

    def __init__(self, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()
        self.device = device

    def answer(self, question: str, max_new_tokens: int = 50) -> str:
        """Отправляет вопрос в instruct формате, возвращает ответ."""
        messages = [
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        # Декодируем только новые токены
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("Loading SmolLM2-135M-Instruct...")
    llm = SmolLMAdapter()
    tests = [
        "What is one plus two?",
        "What is ten minus five?",
        "What is six plus three?",
        "What is eight minus two?",
        "What is zero plus seven?",
    ]
    for q in tests:
        a = llm.answer(q)
        print(f"  Q: {q}")
        print(f"  A: {a}\n")
