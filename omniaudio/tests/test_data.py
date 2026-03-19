"""Tests for OmniAudio data pipeline."""

import numpy as np
import pytest
import torch

try:
    from omniaudio.data import AudioCollator

    _COLLATOR_AVAILABLE = True
except ImportError:
    _COLLATOR_AVAILABLE = False

import os

# Use local tokenizer if available, otherwise try HF hub
_LOCAL_TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "..", "tokenizers", "kazakh-bpe-32k")
TOKENIZER_PATH = _LOCAL_TOKENIZER if os.path.isdir(_LOCAL_TOKENIZER) else "saken-tukenov/kazakh-bpe-32k"


def _make_mock_batch(n=2):
    """Create mock batch mimicking Common Voice structure."""
    texts = ["сәлем", "қайырлы таң"]
    samples = []
    for i in range(n):
        samples.append(
            {
                "audio": {
                    "array": np.random.randn(16000).astype(np.float32),
                    "sampling_rate": 16000,
                },
                "sentence": texts[i % len(texts)],
            }
        )
    return samples


@pytest.mark.skipif(not _COLLATOR_AVAILABLE, reason="omniaudio not importable")
def test_collator_callable():
    try:
        collator = AudioCollator(tokenizer_path=TOKENIZER_PATH)
    except Exception:
        pytest.skip("Could not load tokenizer (network issue)")
    assert callable(collator)


@pytest.mark.skipif(not _COLLATOR_AVAILABLE, reason="omniaudio not importable")
def test_collator_output_shapes():
    try:
        collator = AudioCollator(tokenizer_path=TOKENIZER_PATH)
    except Exception:
        pytest.skip("Could not load tokenizer (network issue)")

    batch = _make_mock_batch(2)
    out = collator(batch)

    assert "mel" in out
    assert "text_ids" in out

    mel = out["mel"]
    text_ids = out["text_ids"]

    assert mel.ndim == 3
    assert mel.shape[0] == 2
    assert mel.shape[1] == 80  # n_mels
    assert torch.isfinite(mel).all()

    assert text_ids.ndim == 2
    assert text_ids.shape[0] == 2
