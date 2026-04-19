from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from gecpaper.scoring.metrics import (
    char_error_rate,
    compute_gleu,
    compute_word_f05,
    multi_ref_cer,
    multi_ref_exact_match,
    multi_ref_gleu,
    multi_ref_word_f05,
)
from gecpaper.taxonomy.schema import parse_annotation

logger = logging.getLogger(__name__)


def run_assessment(
    predict_fn: Callable[[str], str],
    test_data: list[dict],
    multi_ref: bool = False,
) -> dict:
    results = []
    per_l1: dict[str, list[float]] = defaultdict(list)
    per_l2: dict[str, list[float]] = defaultdict(list)

    total_em = 0
    total_cer = 0.0
    total_f05 = 0.0
    total_gleu = 0.0
    identity_correct = 0
    identity_total = 0

    t0 = time.time()
    for i, item in enumerate(test_data):
        source = item["input"]
        prediction = predict_fn(source)

        if multi_ref and "references" in item:
            refs = item["references"]
            target = refs[0]
            f05 = multi_ref_word_f05(source, prediction, refs)
            cer = multi_ref_cer(prediction, refs)
            em = multi_ref_exact_match(prediction, refs)
            gleu = multi_ref_gleu(source, prediction, refs)
        else:
            target = item["target"]
            f05_result = compute_word_f05(source, prediction, target)
            f05 = f05_result["f05"]
            cer = char_error_rate(prediction, target)
            em = prediction.strip() == target.strip()
            gleu = compute_gleu(source, prediction, target)

        is_identity = source.strip() == (target if not multi_ref else item.get("target", target)).strip()
        if is_identity:
            identity_total += 1
            if prediction.strip() == source.strip():
                identity_correct += 1

        total_em += int(em)
        total_cer += cer
        total_f05 += f05
        total_gleu += gleu

        tag = item.get("error_tag", "")
        if tag:
            ann = parse_annotation(tag)
            per_l1[ann.l1.value].append(f05)
            per_l2[ann.l2.value].append(f05)

        results.append({
            "source": source,
            "prediction": prediction,
            "target": target,
            "f05": f05,
            "cer": cer,
            "em": int(em),
            "gleu": gleu,
        })

        if (i + 1) % 100 == 0:
            logger.info("Assessed %d/%d", i + 1, len(test_data))

    n = len(test_data)
    elapsed = time.time() - t0

    metrics = {
        "n": n,
        "exact_match": total_em / n if n else 0,
        "cer": total_cer / n if n else 0,
        "word_f05": total_f05 / n if n else 0,
        "gleu": total_gleu / n if n else 0,
        "identity_preservation": identity_correct / identity_total if identity_total else None,
        "elapsed_seconds": round(elapsed, 1),
        "per_l1": {k: round(sum(v) / len(v), 4) for k, v in per_l1.items()},
        "per_l2": {k: round(sum(v) / len(v), 4) for k, v in per_l2.items()},
    }
    return {"metrics": metrics, "results": results}


def print_metrics(metrics: dict, model_name: str = "model") -> None:
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  N = {metrics['n']}")
    print(f"  Word F0.5    : {metrics['word_f05']:.4f}")
    print(f"  GLEU         : {metrics['gleu']:.4f}")
    print(f"  CER          : {metrics['cer']:.4f}")
    print(f"  Exact Match  : {metrics['exact_match']:.4f}")
    if metrics.get("identity_preservation") is not None:
        print(f"  Identity Pres: {metrics['identity_preservation']:.4f}")
    print(f"  Time         : {metrics['elapsed_seconds']}s")

    if metrics.get("per_l1"):
        print(f"\n  Per-L1 F0.5:")
        for k, v in sorted(metrics["per_l1"].items()):
            print(f"    {k:20s}: {v:.4f}")

    if metrics.get("per_l2"):
        print(f"\n  Per-L2 F0.5 (top 10):")
        sorted_l2 = sorted(metrics["per_l2"].items(), key=lambda x: x[1])
        for k, v in sorted_l2[:10]:
            print(f"    {k:20s}: {v:.4f}")
    print(f"{'=' * 60}\n")
