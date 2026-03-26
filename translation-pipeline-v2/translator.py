"""Translation via CTranslate2 with per-sentence confidence scoring."""

import math
import os
import time
from dataclasses import dataclass

from config import (
    BATCH_SIZE,
    BEAM_SIZE,
    COMPUTE_TYPE,
    MAX_INPUT_LENGTH,
    MAX_DECODING_LENGTH,
)

BASE = os.path.dirname(os.path.abspath(__file__))
CT2_DIR = os.path.join(BASE, "model_ct2")
SPM_PATH = os.path.join(BASE, "model_cache", "model.en-kk.spm")


@dataclass
class TranslationResult:
    text: str
    confidence: float
    token_scores: list[float]


def compute_sentence_confidence(token_log_probs: list[float]) -> float:
    """Compute sentence confidence from token log-probabilities.

    Returns exp(mean(log_probs)) — a value in (0, 1].
    Higher = more confident.
    """
    if not token_log_probs:
        return 0.0
    mean_log_prob = sum(token_log_probs) / len(token_log_probs)
    return math.exp(mean_log_prob)


class Translator:
    """CTranslate2-based EN→KK translator with confidence scoring."""

    def __init__(self, device: str = "cuda", device_index: int = 0):
        import ctranslate2
        import sentencepiece as spm

        self.ct2 = ctranslate2.Translator(
            CT2_DIR, device=device, device_index=device_index,
            compute_type=COMPUTE_TYPE if device != "cpu" else "float32",
        )
        self.sp = spm.SentencePieceProcessor(SPM_PATH)

    def translate_sentences(
        self,
        sentences: list[str],
        batch_size: int = BATCH_SIZE,
        beam_size: int = BEAM_SIZE,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_decoding_length: int = MAX_DECODING_LENGTH,
        verbose: bool = True,
    ) -> list[TranslationResult]:
        """Translate a list of sentences, returning text + confidence for each.

        Batches are sorted by length for minimal padding, then results
        are reordered to match input order.
        """
        if not sentences:
            return []

        # Tokenize
        all_tokens = []
        for s in sentences:
            toks = self.sp.encode(s, out_type=str)
            if len(toks) > max_input_length:
                toks = toks[:max_input_length]
            all_tokens.append(toks)

        results: list[TranslationResult | None] = [None] * len(sentences)
        total_batches = (len(all_tokens) + batch_size - 1) // batch_size
        t0 = time.time()

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(all_tokens))

            # Sort by length for efficient batching
            batch_indices = list(range(start, end))
            batch_indices.sort(key=lambda i: len(all_tokens[i]))
            batch_tokens = [all_tokens[i] for i in batch_indices]

            ct2_results = self.ct2.translate_batch(
                batch_tokens,
                beam_size=beam_size,
                max_decoding_length=max_decoding_length,
                return_scores=True,
            )

            for local_idx, global_idx in enumerate(batch_indices):
                hyp = ct2_results[local_idx]
                translated_tokens = hyp.hypotheses[0]
                translated_text = self.sp.decode(translated_tokens)

                # Extract score — cumulative log-prob for the hypothesis
                token_scores = []
                if hasattr(hyp, 'scores') and hyp.scores:
                    score = float(hyp.scores[0])
                    num_tokens = max(len(translated_tokens), 1)
                    token_scores = [score / num_tokens] * num_tokens

                confidence = compute_sentence_confidence(token_scores)

                results[global_idx] = TranslationResult(
                    text=translated_text,
                    confidence=confidence,
                    token_scores=token_scores,
                )

            if verbose and ((batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1):
                elapsed = time.time() - t0
                done = end
                sps = done / elapsed if elapsed > 0 else 0
                eta = (len(all_tokens) - done) / sps if sps > 0 else 0
                print(f"  [{done}/{len(all_tokens)}] {sps:.0f} sents/sec, ETA {eta:.0f}s", flush=True)

        return results
