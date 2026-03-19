#!/usr/bin/env python3
"""Self-contained remote MC benchmark script for vast.ai.
Runs autonomously: evaluates models, uploads results to HF, self-destructs.
"""
import argparse
import json
import os
import sys
import subprocess
import traceback

MAPPING = {
    "Kazakhstan": "kk_history_of_kazakhstan_unt_mc",
    "biology": "kk_biology_unt_mc",
    "const": "kk_constitution_mc",
    "dastur": "kk_dastur_mc",
    "english": "kk_english_unt_mc",
    "geography": "kk_geography_unt_mc",
    "history": "kk_world_history_unt_mc",
    "kazakh": "kazakh_and_literature_unt_mc",
    "mmlu": "mmlu_translated_kk",
    "right": "kk_human_society_rights_unt_mc",
}

HF_REPO = "stukenov/s-openbench-eval"


def run_single_model(model_id, dtype="bfloat16"):
    """Run mc-benchmark for one model, return flat dict or None."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_id}")
    print(f"{'='*60}", flush=True)

    os.makedirs("/tmp/results", exist_ok=True)
    sanitized = model_id.replace("/", "_")

    r = subprocess.run(
        ["python", "/tmp/scripts/mc-eval-simplified-inference.py",
         "--model_id", model_id,
         "--output_path", "/tmp/results",
         "--dtype", dtype],
        cwd="/tmp/scripts",
        timeout=7200,
    )

    if r.returncode != 0:
        print(f"ERROR: benchmark failed for {model_id} (exit {r.returncode})")
        return None

    csv_path = f"/tmp/results/final-{sanitized}.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}")
        return None

    scores = {}
    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4 and parts[0] in MAPPING:
                scores[MAPPING[parts[0]]] = float(parts[3])

    if len(scores) != 10:
        print(f"WARNING: Only {len(scores)} benchmarks found")

    flat = {"model": model_id, "model_dtype": f"torch.{dtype}", "ppl": 0}
    flat.update(scores)

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"  Average: {avg:.4f}")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    return flat


def upload_to_hf(flat, model_id):
    """Upload flat JSON to HF leaderboard dataset."""
    from huggingface_hub import HfApi
    import io

    sanitized = model_id.replace("/", "__")
    local_path = f"/tmp/results/{sanitized}_flat.json"
    with open(local_path, "w") as f:
        json.dump(flat, f)

    api = HfApi()
    buf = io.BytesIO(json.dumps(flat).encode("utf-8"))
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"model_data/external/{sanitized}.json",
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=f"Add {model_id} benchmark results (automated)",
    )
    print(f"  Uploaded to {HF_REPO}: {sanitized}.json", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--instance-id", default="")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        from huggingface_hub import login
        login(token=args.hf_token)

    results = {}
    for model_id in args.models:
        try:
            flat = run_single_model(model_id, args.dtype)
            if flat:
                upload_to_hf(flat, model_id)
                results[model_id] = "OK"
            else:
                results[model_id] = "FAILED"
        except Exception as e:
            print(f"ERROR for {model_id}: {e}")
            traceback.print_exc()
            results[model_id] = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, status in results.items():
        print(f"  {model}: {status}")

    with open("/tmp/results/summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Self-destruct
    if args.instance_id:
        print(f"\nDestroying instance {args.instance_id}...")
        subprocess.run(
            ["python", "-c",
             f"import subprocess,time; time.sleep(5); "
             f"subprocess.run(['vastai','destroy','instance','{args.instance_id}'])"],
            start_new_session=True,
            timeout=10,
        )


if __name__ == "__main__":
    main()
