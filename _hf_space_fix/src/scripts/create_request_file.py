import json
import os
import pprint
from datetime import datetime, timezone

import click
from colorama import Fore
from huggingface_hub import HfApi, snapshot_download

from src.display.utils import ModelType, WeightType
from src.submission.check_validity import get_model_size

EVAL_REQUESTS_PATH = "eval-queue"
QUEUE_REPO = "open-llm-leaderboard/requests"

precisions = ("float16", "bfloat16", "8bit (LLM.int8)", "4bit (QLoRA / FP4)", "GPTQ")
model_types = [e.name for e in ModelType]
weight_types = [e.name for e in WeightType]


def main():
    api = HfApi()
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot_download(repo_id=QUEUE_REPO, revision="main", local_dir=EVAL_REQUESTS_PATH, repo_type="dataset")

    model_name = click.prompt("Enter model name")
    revision = click.prompt("Enter revision", default="main")
    precision = click.prompt("Enter precision", default="float16", type=click.Choice(precisions))
    model_type = click.prompt("Enter model type", type=click.Choice(model_types))
    weight_type = click.prompt("Enter weight type", default="Original", type=click.Choice(weight_types))
    base_model = click.prompt("Enter base model", default="")
    status = click.prompt("Enter status", default="FINISHED")

    try:
        model_info = api.model_info(repo_id=model_name, revision=revision)
    except Exception as e:
        print(f"{Fore.RED}Could not find model info for {model_name} on the Hub\n{e}{Fore.RESET}")
        return 1

    model_size = get_model_size(model_info=model_info, precision=precision)

    try:
        license = model_info.cardData["license"]
    except Exception:
        license = "?"

    eval_entry = {
        "model": model_name,
        "base_model": base_model,
        "revision": model_info.sha,  # force to use the exact model commit
        "private": False,
        "precision": precision,
        "weight_type": weight_type,
        "status": status,
        "submitted_time": current_time,
        "model_type": model_type,
        "likes": model_info.likes,
        "params": model_size,
        "license": license,
    }

    user_name = ""
    model_path = model_name
    if "/" in model_name:
        user_name = model_name.split("/")[0]
        model_path = model_name.split("/")[1]

    pprint.pprint(eval_entry)

    if click.confirm("Do you want to continue? This request file will be pushed to the hub"):
        click.echo("continuing...")

        out_dir = f"{EVAL_REQUESTS_PATH}/{user_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{model_path}_eval_request_{False}_{precision}_{weight_type}.json"

        with open(out_path, "w") as f:
            f.write(json.dumps(eval_entry))

        api.upload_file(
            path_or_fileobj=out_path,
            path_in_repo=out_path.split(f"{EVAL_REQUESTS_PATH}/")[1],
            repo_id=QUEUE_REPO,
            repo_type="dataset",
            commit_message=f"Add {model_name} to eval queue",
        )
    else:
        click.echo("aborting...")


if __name__ == "__main__":
    main()
