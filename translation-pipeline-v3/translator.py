"""Translation via HuggingFace MarianMT on TPU with per-sentence confidence."""

import math
import os
import time
from dataclasses import dataclass

import torch

from config import (
    HF_MODEL_NAME,
    BATCH_SIZE,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    PAD_BUCKETS,
)


@dataclass
class TranslationResult:
    text: str
    confidence: float


def find_bucket(length: int) -> int:
    """Find smallest bucket that fits the given length."""
    for b in PAD_BUCKETS:
        if length <= b:
            return b
    return PAD_BUCKETS[-1]


class Translator:
    """MarianMT EN-to-KK translator for TPU (torch_xla) or GPU/CPU fallback."""

    def __init__(self, device=None):
        from transformers import MarianMTModel, MarianTokenizer

        self.tokenizer = MarianTokenizer.from_pretrained(HF_MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(HF_MODEL_NAME)

        if device is not None:
            self.device = device
        elif os.environ.get("PJRT_DEVICE") == "TPU":
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.half()
        self.model.eval()

        self.pad_id = self.tokenizer.pad_token_id or 0
        self.eos_id = self.tokenizer.eos_token_id or 0

        print(f"Translator ready on {self.device}", flush=True)

    @torch.no_grad()
    def translate_sentences(
        self,
        sentences: list[str],
        batch_size: int = BATCH_SIZE,
        verbose: bool = True,
    ) -> list[TranslationResult]:
        """Translate sentences, returning text + confidence for each."""
        if not sentences:
            return []

        # Sort by length for better batching
        indexed = sorted(enumerate(sentences), key=lambda x: len(x[1]))
        results: list[TranslationResult | None] = [None] * len(sentences)

        total_batches = (len(indexed) + batch_size - 1) // batch_size
        t0 = time.time()

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(indexed))
            batch = indexed[start:end]

            batch_indices = [b[0] for b in batch]
            batch_texts = [b[1] for b in batch]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
            )

            # Pad to bucket for XLA graph caching (avoid recompilation)
            seq_len = encoded["input_ids"].shape[1]
            bucket_len = find_bucket(seq_len)
            if seq_len < bucket_len:
                pad_size = bucket_len - seq_len
                encoded["input_ids"] = torch.nn.functional.pad(
                    encoded["input_ids"], (0, pad_size), value=self.pad_id,
                )
                encoded["attention_mask"] = torch.nn.functional.pad(
                    encoded["attention_mask"], (0, pad_size), value=0,
                )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_ids = output.sequences
            confidences = self._compute_confidences(output, generated_ids)
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, orig_idx in enumerate(batch_indices):
                results[orig_idx] = TranslationResult(
                    text=decoded[i],
                    confidence=min(max(confidences[i], 0.0), 1.0),
                )

            # Mark step for XLA graph execution
            if "xla" in str(self.device):
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            if verbose and ((batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1):
                elapsed = time.time() - t0
                done = end
                sps = done / elapsed if elapsed > 0 else 0
                eta = (len(sentences) - done) / sps if sps > 0 else 0
                print(f"  [{done}/{len(sentences)}] {sps:.0f} sents/sec, ETA {eta:.0f}s", flush=True)

        return results

    def _compute_confidences(self, output, generated_ids) -> list[float]:
        """Compute per-sentence confidence from generation scores."""
        if not output.scores:
            return [0.5] * generated_ids.shape[0]

        # scores: tuple of (batch_size, vocab_size) per decoding step
        log_probs_per_step = [
            torch.nn.functional.log_softmax(step_scores, dim=-1)
            for step_scores in output.scores
        ]
        # (num_steps, batch, vocab)
        all_log_probs = torch.stack(log_probs_per_step, dim=0)

        gen_tokens = generated_ids[:, 1:]  # skip decoder_start_token
        num_steps = min(all_log_probs.shape[0], gen_tokens.shape[1])

        # Gather log-probs of actually generated tokens
        token_indices = gen_tokens[:, :num_steps].T.unsqueeze(-1)  # (steps, batch, 1)
        selected = torch.gather(all_log_probs[:num_steps], 2, token_indices).squeeze(-1)

        # Mask out pad/eos tokens
        token_mask = gen_tokens[:, :num_steps].T  # (steps, batch)
        valid = (token_mask != self.pad_id) & (token_mask != self.eos_id)
        valid_f = valid.float()

        # Mean log-prob per sentence -> exp -> confidence
        masked_lp = selected * valid_f
        count = valid_f.sum(dim=0).clamp(min=1)
        mean_lp = masked_lp.sum(dim=0) / count
        confidences = torch.exp(mean_lp).cpu().tolist()

        return confidences
