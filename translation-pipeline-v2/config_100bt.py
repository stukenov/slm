"""Configuration constants for translation pipeline v2 — FineWeb-Edu sample-100BT."""

# Translation model
CT2_MODEL_NAME = "HPLT/translate-en-kk-v2.0-hplt_opus"
COMPUTE_TYPE = "int8_float16"
BATCH_SIZE = 4096
BEAM_SIZE = 1
MAX_INPUT_LENGTH = 128
MAX_DECODING_LENGTH = 200

# Source dataset
SOURCE_DATASET = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample-100BT"
ROWS_PER_CHUNK = 1_000_000

# HuggingFace output
HF_REPO = "stukenov/sozkz-fineweb-edu-100bt-en-kk"

# Pre-translation filters (per sentence)
MIN_WORDS_PER_SENTENCE = 3
MAX_SENTENCE_LENGTH = 512
NON_ALPHA_THRESHOLD = 0.3

# Post-translation filters (per sentence)
DUPLICATE_SIMILARITY_THRESHOLD = 0.9
LENGTH_RATIO_MAX = 3.0
LENGTH_RATIO_MIN = 0.3
NGRAM_REPEAT_THRESHOLD = 3

# Checkpointing
CHECKPOINT_EVERY = 50_000       # rows within a chunk (save to disk)
UPLOAD_VERIFY_ROWS = True       # verify uploaded shard row count

# Progress tracking
PROGRESS_FILE = "progress_100bt.json"
