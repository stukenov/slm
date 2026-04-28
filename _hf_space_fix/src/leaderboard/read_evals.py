import json
from pathlib import Path
from json import JSONDecodeError
import logging
import math

from dataclasses import dataclass, field
from typing import Optional, Dict, List

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np

from src.display.formatting import make_clickable_model
from src.display.utils import AutoEvalColumn, ModelType, Precision, Tasks, WeightType, parse_datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class EvalResult:
    # Also see src.display.utils.AutoEvalColumn for what will be displayed.
    eval_name: str  # org_model_precision (uid)
    full_model: str  # org/model (path on hub)
    org: Optional[str]
    model: str
    revision: str  # commit hash, "" if main
    results: Dict[str, float]
    precision: Precision = Precision.Unknown
    model_type: ModelType = ModelType.Unknown  # Pretrained, fine tuned, ...
    weight_type: WeightType = WeightType.Original
    architecture: str = "Unknown"  # From config file
    license: str = "?"
    likes: int = 0
    num_params: int = 0
    date: str = ""  # submission date of request file
    still_on_hub: bool = True
    is_merge: bool = False
    not_flagged: bool = False
    status: str = "FINISHED"
    # List of tags, initialized to a new empty list for each instance to avoid the pitfalls of mutable default arguments.
    tags: List[str] = field(default_factory=list)

    @classmethod
    def init_from_json_file(cls, json_filepath: str) -> "EvalResult":
        with open(json_filepath, "r") as fp:
            data = json.load(fp)

        config = data.get("config_general", {})
        precision = Precision.from_str(config.get("model_dtype", "unknown"))
        org_and_model = config.get("model_name", "").split("/", 1)
        org = org_and_model[0] if len(org_and_model) > 1 else None
        model = org_and_model[-1]
        if len(org_and_model) == 1:
            org = None
            model = org_and_model[0]
            result_key = f"{model}_{precision.value.name}"
        else:
            org = org_and_model[0]
            model = org_and_model[1]
            result_key = f"{org}_{model}_{precision.value.name}"
        full_model = "/".join(org_and_model)

        results = cls.extract_results(data)  # Properly call the method to extract results

        return cls(
            eval_name=result_key,
            full_model=full_model,
            org=org,
            model=model,
            results=results,
            precision=precision,
            revision=config.get("model_sha", ""),
        )

    @staticmethod
    def extract_results(data: Dict) -> Dict[str, float]:
        """
        Extract and process benchmark results from a given dict.

        Parameters:
        - data (Dict): A dictionary containing benchmark data. This dictionary must
        include 'versions' and 'results' keys with respective sub-data.

        Returns:
        - Dict[str, float]: A dictionary where keys are benchmark names and values
        are the processed average scores as percentages.

        Notes:
        - The method specifically checks for certain benchmark names to skip outdated entries.
        - Handles NaN values by setting the corresponding benchmark result to 0.0.
        - Averages scores across metrics for benchmarks found in the data, in a percentage format.
        """
        results = {}
        for task in Tasks:
            task = task.value
            # We skip old mmlu entries
            if task.benchmark == "hendrycksTest":
                for mmlu_k in ["harness|hendrycksTest-abstract_algebra|5", "hendrycksTest-abstract_algebra"]:
                    if mmlu_k in data["versions"] and data["versions"][mmlu_k] == 0:
                        continue

            # Some benchamrk values are NaNs, mostly truthfulQA
            # Would be more optimal (without the whole dict itertion) if benchmark name was same as key in results
            # e.g. not harness|truthfulqa:mc|0 but truthfulqa:mc
            for k, v in data["results"].items():
                if task.benchmark in k:
                    if math.isnan(float(v[task.metric])):
                        results[task.benchmark] = 0.0
                        continue

            # We average all scores of a given metric (mostly for mmlu)
            accs = np.array([v.get(task.metric, None) for k, v in data["results"].items() if task.benchmark in k])
            if accs.size == 0 or any([acc is None for acc in accs]):
                continue

            mean_acc = np.mean(accs) * 100.0
            results[task.benchmark] = mean_acc

        return results

    def update_with_request_file(self, requests_path):
        """Finds the relevant request file for the current model and updates info with it."""
        try:
            request_file = get_request_file_for_model(requests_path, self.full_model, self.precision.value.name)
            if request_file is None:
                logging.warning(f"No request file for {self.org}/{self.model}")
                self.status = "FAILED"
                return

            with open(request_file, "r") as f:
                request = json.load(f)

            self.model_type = ModelType.from_str(request.get("model_type", "Unknown"))
            self.weight_type = WeightType[request.get("weight_type", "Original")]
            self.num_params = int(request.get("params", 0))  # Ensuring type safety
            self.date = request.get("submitted_time", "")
            self.architecture = request.get("architectures", "Unknown")
            self.status = request.get("status", "FAILED")

        except FileNotFoundError:
            self.status = "FAILED"
            logging.error(f"Request file: {request_file} not found for {self.org}/{self.model}")
        except JSONDecodeError:
            self.status = "FAILED"
            logging.error(f"Error decoding JSON from the request file for {self.org}/{self.model}")
        except KeyError as e:
            self.status = "FAILED"
            logging.error(f"Key error {e} in processing request file for {self.org}/{self.model}")
        except Exception as e:  # Catch-all for any other unexpected exceptions
            self.status = "FAILED"
            logging.error(f"Unexpected error {e} for {self.org}/{self.model}")

    def update_with_dynamic_file_dict(self, file_dict):
        """Update object attributes based on the provided dictionary, with error handling for missing keys and type validation."""
        # Default values set for optional or potentially missing keys.
        self.license = file_dict.get("license", "?")
        self.likes = int(file_dict.get("likes", 0))  # Ensure likes is treated as an integer
        self.still_on_hub = file_dict.get("still_on_hub", False)  # Default to False if key is missing
        self.tags = file_dict.get("tags", [])

        # Calculate `flagged` only if 'tags' is not empty and avoid calculating each time
        self.not_flagged = not (any("flagged" in tag for tag in self.tags))

    def to_dict(self):
        """Converts the Eval Result to a dict compatible with our dataframe display"""
        average = sum([v for v in self.results.values() if v is not None]) / len(Tasks)
        data_dict = {
            "eval_name": self.eval_name,  # not a column, just a save name,
            AutoEvalColumn.precision.name: self.precision.value.name,
            AutoEvalColumn.model_type.name: self.model_type.value.name,
            AutoEvalColumn.model_type_symbol.name: self.model_type.value.symbol,
            AutoEvalColumn.weight_type.name: self.weight_type.value.name,
            AutoEvalColumn.architecture.name: self.architecture,
            AutoEvalColumn.model.name: make_clickable_model(self.full_model),
            AutoEvalColumn.fullname.name: self.full_model,
            AutoEvalColumn.revision.name: self.revision,
            AutoEvalColumn.average.name: average,
            AutoEvalColumn.license.name: self.license,
            AutoEvalColumn.likes.name: self.likes,
            AutoEvalColumn.params.name: self.num_params,
            AutoEvalColumn.still_on_hub.name: self.still_on_hub,
            AutoEvalColumn.merged.name: not ("merge" in self.tags if self.tags else False),
            AutoEvalColumn.moe.name: not (
                ("moe" in self.tags if self.tags else False) or "moe" in self.full_model.lower()
            ),
            AutoEvalColumn.not_flagged.name: self.not_flagged,
        }

        for task in Tasks:
            data_dict[task.value.col_name] = self.results[task.value.benchmark]

        return data_dict


def get_request_file_for_model(requests_path, model_name, precision):
    """Selects the correct request file for a given model. Only keeps runs tagged as FINISHED"""
    requests_path = Path(requests_path)
    pattern = f"{model_name}_eval_request_*.json"

    # Using pathlib to find files matching the pattern
    request_files = list(requests_path.glob(pattern))

    # Sort the files by name in descending order to mimic 'reverse=True'
    request_files.sort(reverse=True)

    # Select the correct request file based on 'status' and 'precision'
    request_file = None
    for request_file in request_files:
        with request_file.open("r") as f:
            req_content = json.load(f)
            if req_content["status"] == "FINISHED" and req_content["precision"] == precision.split(".")[-1]:
                request_file = str(request_file)

    # Return empty string if no file found that matches criteria
    return request_file


def get_raw_eval_results(results_path: str, requests_path: str, dynamic_path: str) -> list[EvalResult]:
    """From the path of the results folder root, extract all needed info for results"""
    with open(dynamic_path) as f:
        dynamic_data = json.load(f)

    results_path = Path(results_path)
    model_files = list(results_path.rglob("results_*.json"))
    model_files.sort(key=lambda file: parse_datetime(file.stem.removeprefix("results_")))

    eval_results = {}
    # Wrap model_files iteration with tqdm for progress display
    for model_result_filepath in tqdm(model_files, desc="Processing model files"):
        # Creation of result
        eval_result = EvalResult.init_from_json_file(model_result_filepath)
        with logging_redirect_tqdm():
            eval_result.update_with_request_file(requests_path)

        if eval_result.full_model in dynamic_data:
            eval_result.update_with_dynamic_file_dict(dynamic_data[eval_result.full_model])
            # Hardcoding because of gating problem
            if any([org in eval_result.full_model for org in ["meta-llama/", "google/", "tiiuae/"]]):
                eval_result.still_on_hub = True

        # Store results of same eval together
        eval_name = eval_result.eval_name
        if eval_name in eval_results.keys():
            eval_results[eval_name].results.update({k: v for k, v in eval_result.results.items() if v is not None})
        else:
            eval_results[eval_name] = eval_result

    results = []
    for k, v in eval_results.items():
        try:
            if v.status == "FINISHED":
                v.to_dict()  # we test if the dict version is complete
                results.append(v)
        except KeyError as e:
            logging.error(f"Error while checking model {k} {v.date} json, no key: {e}")  # not all eval values present
            continue

    return results
