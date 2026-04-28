"""Monitor ALL HuggingFace gated models & datasets and notify via Telegram.
Auto-discovers gated repos — no hardcoded list needed.
"""
import json
import time
import requests
from pathlib import Path
from huggingface_hub import HfApi

# Config
TELEGRAM_TOKEN = "8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g"
TELEGRAM_CHAT_ID = "47474471"
STATE_FILE = Path("/root/slm/logs/hf_access_state.json")
CHECK_INTERVAL = 300  # 5 minutes
AUTHOR = "stukenov"


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def discover_gated_repos(api, token):
    """Find all gated models and datasets under AUTHOR."""
    gated_repos = []

    # Check models
    for model in api.list_models(author=AUTHOR, token=token):
        try:
            info = api.model_info(model.id, token=token)
            if getattr(info, "gated", None):
                gated_repos.append(("model", model.id))
        except Exception:
            pass

    # Check datasets
    for ds in api.list_datasets(author=AUTHOR, token=token):
        try:
            info = api.dataset_info(ds.id, token=token)
            if getattr(info, "gated", None):
                gated_repos.append(("dataset", ds.id))
        except Exception:
            pass

    return gated_repos


def check_access():
    token = open("/root/.cache/huggingface/token").read().strip()
    api = HfApi()
    state = load_state()

    gated_repos = discover_gated_repos(api, token)

    if not gated_repos:
        return

    for repo_type, repo_id in gated_repos:
        known = set(state.get(repo_id, []))

        try:
            accepted = api.list_accepted_access_requests(repo_id, token=token)
        except Exception:
            continue

        new_users = []
        all_usernames = []
        for req in accepted:
            username = req.username
            all_usernames.append(username)
            if username not in known:
                new_users.append(req)

        if new_users:
            short_name = repo_id.split("/")[-1]
            for user in new_users:
                msg = (
                    f"HF Access ({repo_type}): {short_name}\n"
                    f"User: {user.username}\n"
                    f"Full name: {getattr(user, 'fullname', 'N/A')}\n"
                    f"Time: {getattr(user, 'timestamp', 'N/A')}"
                )
                send_telegram(msg)
                print(f"New access: {user.username} -> {repo_id}")

        state[repo_id] = all_usernames

    save_state(state)


def main():
    token = open("/root/.cache/huggingface/token").read().strip()
    api = HfApi()
    gated = discover_gated_repos(api, token)

    print(f"HF Access Monitor started. Gated repos: {len(gated)}. Checking every {CHECK_INTERVAL}s")
    print("Will only notify on NEW access requests.")

    while True:
        try:
            check_access()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
