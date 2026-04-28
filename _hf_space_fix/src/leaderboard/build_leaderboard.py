import json
import logging
import os
import time

import pandas as pd
from huggingface_hub import snapshot_download

from src.envs import DATA_PATH, HF_TOKEN_PRIVATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def time_diff_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        diff = time.time() - start_time
        logging.info("Time taken for %s: %.1fs", func.__name__, diff)
        return result
    return wrapper


@time_diff_wrapper
def download_dataset(repo_id, local_dir, repo_type="dataset"):
    """Download dataset with caching (no force re-download)."""
    os.makedirs(local_dir, exist_ok=True)
    try:
        logging.info("Downloading %s to %s", repo_id, local_dir)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type=repo_type,
            tqdm_class=None,
            token=HF_TOKEN_PRIVATE,
            etag_timeout=10,
        )
        logging.info("Download successful")
    except Exception as e:
        logging.error("Error downloading %s: %s", repo_id, e)


def download_openbench():
    download_dataset("stukenov/kaz-llm-lb-metainfo", DATA_PATH)
    download_dataset("stukenov/s-openbench-eval", "m_data")


# ─── Cached DataFrame ───
_leaderboard_cache = None

def build_leadearboard_df(force=False):
    global _leaderboard_cache
    if _leaderboard_cache is not None and not force:
        return _leaderboard_cache.copy()

    initial_file_path = f"{os.path.abspath(DATA_PATH)}/leaderboard.json"
    logging.info("Reading leaderboard from: %s", initial_file_path)
    with open(initial_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame.from_records(data)

    cols = ['model', 'mmlu_translated_kk', 'kk_constitution_mc', 'kk_dastur_mc',
            'kazakh_and_literature_unt_mc', 'kk_geography_unt_mc',
            'kk_world_history_unt_mc', 'kk_history_of_kazakhstan_unt_mc',
            'kk_english_unt_mc', 'kk_biology_unt_mc',
            'kk_human_society_rights_unt_mc', 'model_dtype', 'ppl']
    leaderboard_df = df[cols].copy()

    score_cols = cols[1:11]
    leaderboard_df['avg'] = leaderboard_df[score_cols].mean(axis=1)
    leaderboard_df.sort_values(by='avg', ascending=False, inplace=True)

    numeric_cols = leaderboard_df.select_dtypes(include=['number']).columns
    leaderboard_df[numeric_cols] = leaderboard_df[numeric_cols].round(3)

    ordered = ['model', 'avg'] + score_cols + ['model_dtype', 'ppl']
    leaderboard_df = leaderboard_df[ordered]

    _leaderboard_cache = leaderboard_df
    return leaderboard_df.copy()


def invalidate_cache():
    global _leaderboard_cache
    _leaderboard_cache = None
