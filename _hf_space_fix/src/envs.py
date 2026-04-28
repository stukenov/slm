import os

from huggingface_hub import HfApi

# clone / pull the lmeh eval data
H4_TOKEN = os.environ.get("H4_TOKEN", None)
REPO_ID = "stukenov/kaz-llm-lb"
QUEUE_REPO = "open-llm-leaderboard/requests"
DYNAMIC_INFO_REPO = "open-llm-leaderboard/dynamic_model_information"
RESULTS_REPO = "open-llm-leaderboard/results"

PRIVATE_QUEUE_REPO = "open-llm-leaderboard/private-requests"
PRIVATE_RESULTS_REPO = "open-llm-leaderboard/private-results"

IS_PUBLIC = bool(os.environ.get("IS_PUBLIC", True))

HF_HOME = os.getenv("HF_HOME", ".")
HF_TOKEN_PRIVATE = os.environ.get("H4_TOKEN")

# Check HF_HOME write access
if not os.access(HF_HOME, os.W_OK):
    HF_HOME = "."
    os.environ["HF_HOME"] = HF_HOME

DATA_PATH = os.path.join(HF_HOME, "data")
# DATA_ARENA_PATH = os.path.join(DATA_PATH, "arena-hard-v0.1")

RESET_JUDGEMENT_ENV = "RESET_JUDGEMENT"

API = HfApi(token=H4_TOKEN)

# useless env
EVAL_REQUESTS_PATH = os.path.join(HF_HOME, "data/eval-queue")
PATH_TO_COLLECTION = "open-llm-leaderboard/llm-leaderboard-best-models-652d6c7965a4619fb5c27a03"

# Rate limit variables
RATE_LIMIT_PERIOD = 7
RATE_LIMIT_QUOTA = 5
HAS_HIGHER_RATE_LIMIT = ["TheBloke"]
