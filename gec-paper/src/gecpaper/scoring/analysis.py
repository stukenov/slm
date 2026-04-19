from __future__ import annotations

import random
from collections import defaultdict

from gecpaper.scoring.metrics import compute_word_f05
from gecpaper.taxonomy.schema import parse_annotation


def per_category_breakdown(
    test_data: list[dict],
    predictions: list[str],
    level: str = "l2",
) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)

    for item, pred in zip(test_data, predictions):
        tag = item.get("error_tag", "")
        if not tag or tag == "identity":
            continue
        ann = parse_annotation(tag)
        key = getattr(ann, level, ann.l2).value if hasattr(getattr(ann, level, None), "value") else ann.l2.value

        source = item["input"]
        target = item["target"]
        result = compute_word_f05(source, pred, target)
        em = int(pred.strip() == target.strip())
        groups[key].append({"f05": result["f05"], "precision": result["precision"],
                            "recall": result["recall"], "em": em})

    breakdown = {}
    for key, items in sorted(groups.items()):
        n = len(items)
        breakdown[key] = {
            "n": n,
            "f05": sum(r["f05"] for r in items) / n,
            "precision": sum(r["precision"] for r in items) / n,
            "recall": sum(r["recall"] for r in items) / n,
            "em": sum(r["em"] for r in items) / n,
        }
    return breakdown


def bootstrap_significance(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    assert len(scores_a) == len(scores_b), "Score lists must have equal length"
    n = len(scores_a)
    rng = random.Random(seed)

    observed_diff = sum(scores_a) / n - sum(scores_b) / n
    count_extreme = 0

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        mean_a = sum(scores_a[i] for i in indices) / n
        mean_b = sum(scores_b[i] for i in indices) / n
        boot_diff = mean_a - mean_b
        if abs(boot_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_bootstrap
    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }
