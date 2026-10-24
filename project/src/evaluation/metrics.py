"""Standard top-K ranking metrics used throughout the evaluation suite.

All metrics take:
    recommended: list[str]   property_ids in ranked order (best first)
    relevant:    set[str]    property_ids the user actually interacted with
                              in the held-out TEST split (ground truth)
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def precision_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    relevant = set(relevant)
    hits = sum(1 for p in top_k if p in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for p in top_k if p in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    top_k = set(recommended[:k])
    return 1.0 if (top_k & relevant) else 0.0


def average_precision_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = 0
    sum_precisions = 0.0
    for i, p in enumerate(top_k, start=1):
        if p in relevant:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    return sum_precisions / min(len(relevant), k)


def ndcg_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    top_k = recommended[:k]
    dcg = 0.0
    for i, p in enumerate(top_k, start=1):
        if p in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


METRIC_FUNCS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "map": average_precision_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
}


def evaluate_recommendations(recommended: Sequence[str], relevant: Iterable[str], k_values: List[int]) -> dict:
    """Compute every metric at every K for one user's recommendation list."""
    out = {}
    for k in k_values:
        for name, func in METRIC_FUNCS.items():
            out[f"{name}@{k}"] = func(recommended, relevant, k)
    return out
