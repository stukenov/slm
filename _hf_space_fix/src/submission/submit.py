from src.display.formatting import styled_message
# from src.leaderboard.filter_models import DO_NOT_SUBMIT_MODELS
# from src.submission.check_validity import (
#     already_submitted_models,
#     check_model_card,
#     get_model_size,
#     get_model_tags,
#     is_model_on_hub,
#     user_submission_permission,
# )

REQUESTED_MODELS = None
USERS_TO_SUBMISSION_DATES = None


def add_new_eval(
    model: str,
):
    # global REQUESTED_MODELS
    # global USERS_TO_SUBMISSION_DATES
    # if not REQUESTED_MODELS:
    #     REQUESTED_MODELS, USERS_TO_SUBMISSION_DATES = already_submitted_models(EVAL_REQUESTS_PATH)

    # user_name = ""
    # model_path = model
    # if "/" in model:
    #     user_name = model.split("/")[0]
    #     model_path = model.split("/")[1]

    # # precision = precision.split(" ")[0]
    # current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # if model_type is None or model_type == "":
    #     return styled_error("Please select a model type.")

    # # Is the user rate limited?
    # if user_name != "":
    #     user_can_submit, error_msg = user_submission_permission(
    #         user_name, USERS_TO_SUBMISSION_DATES, RATE_LIMIT_PERIOD, RATE_LIMIT_QUOTA
    #     )
    #     if not user_can_submit:
    #         return styled_error(error_msg)

    # Did the model authors forbid its submission to the leaderboard?
    # if model in DO_NOT_SUBMIT_MODELS or base_model in DO_NOT_SUBMIT_MODELS:
    #     return styled_warning("Model authors have requested that their model be not submitted on the leaderboard.")

    # if model == "CohereForAI/c4ai-command-r-plus":
    #     return styled_warning(
    #         "This model cannot be submitted manually on the leaderboard before the transformers release."
    #     )

    # # Does the model actually exist?
    # if revision == "":
    #     revision = "main"

    # # Is the model on the hub?
    # if weight_type in ["Delta", "Adapter"]:
    #     base_model_on_hub, error, _ = is_model_on_hub(
    #         model_name=base_model, revision=revision, token=H4_TOKEN, test_tokenizer=True
    #     )
    #     if not base_model_on_hub:
    #         return styled_error(f'Base model "{base_model}" {error}')

    # architecture = "?"
    # downloads = 0
    # created_at = ""
    # if not weight_type == "Adapter":
    #     model_on_hub, error, model_config = is_model_on_hub(model_name=model, revision=revision, test_tokenizer=True)
    #     if not model_on_hub or model_config is None:
    #         return styled_error(f'Model "{model}" {error}')
    #     if model_config is not None:
    #         architectures = getattr(model_config, "architectures", None)
    #         if architectures:
    #             architecture = ";".join(architectures)
    #         downloads = getattr(model_config, "downloads", 0)
    #         created_at = getattr(model_config, "created_at", "")

    # Is the model info correctly filled?
    # try:
    #     model_info = API.model_info(repo_id=model, revision=revision)
    # except Exception:
    #     return styled_error("Could not get your model information. Please fill it up properly.")

    # model_size = get_model_size(model_info=model_info, precision=precision)

    # Were the model card and license filled?
    # try:
    #     license = model_info.cardData["license"]
    # except Exception:
    #     return styled_error("Please select a license for your model")

    # modelcard_OK, error_msg, model_card = check_model_card(model)
    # if not modelcard_OK:
    #     return styled_error(error_msg)

    # tags = get_model_tags(model_card, model)

    # # Seems good, creating the eval
    # print("Adding new eval")

    # eval_entry = {
    #     "model": model,
    #     # "base_model": base_model,
    #     # "revision": model_info.sha, # force to use the exact model commit
    #     # "private": private,
    #     # "precision": precision,
    #     # "params": model_size,
    #     # "architectures": architecture,
    #     # "weight_type": weight_type,
    #     "status": "PENDING",
    #     # "submitted_time": current_time,
    #     # "model_type": model_type,
    #     "job_id": -1,
    #     "job_start_time": None,
    # }

    # supplementary_info = {
    #     "likes": model_info.likes,
    #     "license": license,
    #     "still_on_hub": True,
    #     "tags": tags,
    #     "downloads": downloads,
    #     "created_at": created_at,
    # }

    # # Check for duplicate submission
    # if f"{model}_{revision}_{precision}" in REQUESTED_MODELS:
    #     return styled_warning("This model has been already submitted.")

    # print("Creating eval file")
    # OUT_DIR = f"{EVAL_REQUESTS_PATH}/{user_name}"
    # os.makedirs(OUT_DIR, exist_ok=True)
    # out_path = f"{OUT_DIR}/{model_path}_eval_request_{private}_{precision}_{weight_type}.json"

    # with open(out_path, "w") as f:
    #     f.write(json.dumps(eval_entry))

    # print("Uploading eval file")
    # API.upload_file(
    #     path_or_fileobj=out_path,
    #     path_in_repo=out_path.split("eval-queue/")[1],
    #     repo_id=QUEUE_REPO,
    #     repo_type="dataset",
    #     commit_message=f"Add {model} to eval queue",
    # )

    # We want to grab the latest version of the submission file to not accidentally overwrite it
    # snapshot_download(
    #     repo_id=DYNAMIC_INFO_REPO, local_dir=DYNAMIC_INFO_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30
    # )

    # with open(DYNAMIC_INFO_FILE_PATH) as f:
    #     all_supplementary_info = json.load(f)

    # # all_supplementary_info[model] = supplementary_info
    # with open(DYNAMIC_INFO_FILE_PATH, "w") as f:
    #     json.dump(all_supplementary_info, f, indent=2)

    # API.upload_file(
    #     path_or_fileobj=DYNAMIC_INFO_FILE_PATH,
    #     path_in_repo=DYNAMIC_INFO_FILE_PATH.split("/")[-1],
    #     repo_id=DYNAMIC_INFO_REPO,
    #     repo_type="dataset",
    #     commit_message=f"Add {model} to dynamic info queue",
    # )

    # # Remove the local file
    # os.remove(out_path)

    return styled_message("Your request has been submitted to the evaluation queue!\nPlease wait for up to an hour.")
