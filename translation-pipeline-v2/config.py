"""Configuration constants for translation pipeline v2."""

# Translation model
CT2_MODEL_NAME = "HPLT/translate-en-kk-v2.0-hplt_opus"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 4096
BEAM_SIZE = 1
MAX_INPUT_LENGTH = 128
MAX_DECODING_LENGTH = 200

# Source dataset
SOURCE_DATASET = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample-10BT"
ROWS_PER_CHUNK = 1_000_000

# HuggingFace output
HF_REPO = "stukenov/sozkz-fineweb-edu-kk-v2"

# Pre-translation filters (per sentence)
MIN_WORDS_PER_SENTENCE = 3
MAX_SENTENCE_LENGTH = 512
NON_ALPHA_THRESHOLD = 0.3  # >30% non-alpha chars → skip

# Post-translation filters (per sentence)
DUPLICATE_SIMILARITY_THRESHOLD = 0.9  # translation ≈ original → skip
LENGTH_RATIO_MAX = 3.0  # translation too long vs original
LENGTH_RATIO_MIN = 0.3  # translation too short vs original
NGRAM_REPEAT_THRESHOLD = 3  # 3+ repeats of same n-gram → skip

# Testing
TEST_SAMPLE_SIZE = 100
VALIDATION_SAMPLE_SIZE = 1000

# Checkpoint
CHECKPOINT_EVERY = 100_000
