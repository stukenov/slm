"""Configuration constants for translation pipeline v3 (TPU)."""

# Translation model (HuggingFace MarianMT — no CTranslate2 on TPU)
HF_MODEL_NAME = "HPLT/translate-en-kk-v2.0-hplt_opus"

# TPU inference
BATCH_SIZE = 256          # sentences per batch (TPU likes large batches)
MAX_INPUT_LENGTH = 128    # tokens
MAX_OUTPUT_LENGTH = 200   # tokens
# Fixed bucket lengths for XLA graph caching (avoids recompilation)
PAD_BUCKETS = [16, 32, 64, 96, 128]

# Source dataset
SOURCE_DATASET = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample-10BT"
ROWS_PER_CHUNK = 1_000_000

# HuggingFace output
HF_REPO = "stukenov/sozkz-fineweb-edu-kk-v3"

# Pre-translation filters (per sentence)
MIN_WORDS_PER_SENTENCE = 3
MAX_SENTENCE_LENGTH = 512
NON_ALPHA_THRESHOLD = 0.3

# Post-translation filters (per sentence)
DUPLICATE_SIMILARITY_THRESHOLD = 0.9
LENGTH_RATIO_MAX = 3.0
LENGTH_RATIO_MIN = 0.3
NGRAM_REPEAT_THRESHOLD = 3

# Progress
PROGRESS_FILE = "progress.json"
LOG_EVERY_ROWS = 10_000
