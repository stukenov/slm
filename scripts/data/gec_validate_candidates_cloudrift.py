#!/usr/bin/env python3
"""Strict validation for synthetic Kazakh GEC candidate pairs.

Reads candidate JSONL rows and filters them with:
  - deterministic validation
  - CloudRift Qwen judge
  - round-trip correction check

Usage:
    python3 scripts/data/gec_validate_candidates_cloudrift.py \
      --input /path/to/candidates.jsonl \
      --output_dir ./gec_validated
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.data.gec_generate_qwen_cloudrift import (
    TRACKER,
    append_jsonl,
    normalize_text,
    save_progress,
    validate_pair,
    verify_with_judge,
    verify_with_roundtrip,
)


def load_rows(path: str, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def process_row(row: dict) -> tuple[dict | None, str]:
    incorrect = normalize_text(row.get("input", ""))
    correct = normalize_text(row.get("output", ""))
    ok, reason = validate_pair(incorrect, correct)
    if not ok:
        return None, reason

    judge_ok, judge_meta, judge_reason = verify_with_judge(incorrect, correct)
    if not judge_ok:
        return None, judge_reason

    rt_ok, repaired, rt_reason = verify_with_roundtrip(incorrect, correct)
    if not rt_ok:
        return None, rt_reason

    validated = dict(row)
    meta = dict(validated.get("meta", {}))
    meta["judge"] = judge_meta
    meta["roundtrip"] = {"repaired": repaired, "status": rt_reason}
    validated["meta"] = meta
    return validated, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Candidate JSONL file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.input, args.limit)
    print(f"Loaded {len(rows)} candidate rows", flush=True)

    accepted_path = os.path.join(args.output_dir, "accepted.jsonl")
    rejected_path = os.path.join(args.output_dir, "rejected.jsonl")

    accepted: list[dict] = []
    rejected: list[dict] = []
    failures: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        future_map = {ex.submit(process_row, row): row for row in rows}
        for idx, future in enumerate(as_completed(future_map), start=1):
            row = future_map[future]
            try:
                validated, reason = future.result()
            except Exception as exc:
                validated, reason = None, str(exc)

            if validated:
                accepted.append(validated)
            else:
                bad = dict(row)
                bad["reject_reason"] = reason
                rejected.append(bad)
                failures[reason] = failures.get(reason, 0) + 1

            if idx % 10 == 0 or idx == len(rows):
                print(
                    f"  done={idx}/{len(rows)} accepted={len(accepted)} "
                    f"rejected={len(rejected)} [{TRACKER.summary()}]",
                    flush=True,
                )

    append_jsonl(accepted_path, accepted)
    append_jsonl(rejected_path, rejected)
    progress = {
        "input_rows": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "failures": failures,
        "tracker": TRACKER.summary(),
        "spent_usd": TRACKER.cost_usd(),
    }
    save_progress(args.output_dir, progress)

    print("\nValidation done.", flush=True)
    print(f"  accepted: {len(accepted)}", flush=True)
    print(f"  rejected: {len(rejected)}", flush=True)
    print(f"  failures: {failures}", flush=True)
    print(f"  spend:    ${TRACKER.cost_usd():.4f}", flush=True)


if __name__ == "__main__":
    main()
